import contextlib
import logging

import numpy as np
import pandas as pd

from .. import settings
from ..core import FitResult, fit_metrics, padded_stage_grid
from ..data.zero_flow import ZeroFlowEstimate, estimate_zero_flow

log = logging.getLogger(__name__)

#: Column names ratingcurve's own table uses, mapped to this package's names. Both of
#: its summaries are of the posterior *predictive*, so both are labelled as such;
#: ``discharge_cfs`` is added separately from the rating itself.
_TABLE_RENAME = {"stage": "stage_ft",
                 "discharge": "discharge_predictive_mean_cfs",
                 "median": "discharge_predictive_median_cfs"}

#: Standard-normal quantile for a 95% interval.
_Z95 = 1.959963984540054

#: The form :meth:`BayesianRating.equation` returns parameters for, written out so a
#: caller reading ``a``, ``b`` and ``hs`` cannot mistake which notation they belong to.
#: This is ratingcurve's own documented form. ASCII deliberately: it goes into CSVs,
#: terminals and report text, where a Greek sigma or a subscript may not survive.
POWER_LAW_FORM = "ln(q) = a + sum(b[i] * ln(max(h - hs[i], 0) + ho[i]))"

#: Significant figures in the substituted equation string. Six matches what the
#: parameters are worth: the posterior mean of a breakpoint is not known to more.
_EQUATION_PRECISION = 6


def _substituted_form(intercept, exponents, breakpoints, offsets) -> str:
    """:data:`POWER_LAW_FORM` with the fitted values in place of the symbols.

    Written to be read literally: every ``ho[i]`` is resolved to the ``+ 1`` it stands
    for, or dropped where it is zero, so the line needs no key to interpret.
    """
    precision = _EQUATION_PRECISION
    terms = []
    for exponent, breakpoint_, offset in zip(exponents, breakpoints, offsets):
        head = f"max(h - {breakpoint_:.{precision}g}, 0)"
        if offset:
            head += f" + {offset:.{precision}g}"
        terms.append(f"{exponent:.{precision}g} * ln({head})")
    return f"ln(q) = {intercept:.{precision}g} + " + " + ".join(terms)


#: Attribute value types NetCDF can store. Anything else has to be encoded.
_NETCDF_ATTR_TYPES = (str, bytes, int, float, complex, list, tuple, np.ndarray,
                      np.number)


@contextlib.contextmanager
def coerced_netcdf_attrs(idata):
    """Temporarily JSON-encode any posterior attributes NetCDF cannot store.

    NetCDF attributes must be strings, numbers or arrays. Samplers write richer
    things into the posterior's ``attrs`` dict, which NetCDF cannot store, so they
    are encoded as JSON instead.

    Parameters
    ----------
    idata : arviz.InferenceData or None
        The posterior whose attributes are coerced. ``None`` is accepted and does
        nothing, so callers need no guard.

    Yields
    ------
    arviz.InferenceData or None
        The same object, with coercible attributes replaced for the duration.
    """
    import json

    if idata is None:
        yield idata
        return
    saved = dict(getattr(idata, "attrs", {}) or {})
    unstorable = {key: value for key, value in saved.items()
                  if not isinstance(value, _NETCDF_ATTR_TYPES)}
    if not unstorable:
        yield idata
        return
    log.debug("JSON-encoding %d posterior attribute(s) NetCDF cannot store: %s",
              len(unstorable), ", ".join(sorted(unstorable)))
    try:
        idata.attrs = {key: (value if isinstance(value, _NETCDF_ATTR_TYPES)
                             else json.dumps(value, default=str))
                       for key, value in saved.items()}
        yield idata
    finally:
        idata.attrs = saved


@contextlib.contextmanager
def cleared_initial_values(model):
    """Temporarily drop a PyMC model's pinned initial values.

    The power-law family pins an initial value on its breakpoint prior, and
    ``pm.compute_log_likelihood`` rebuilds the model graph, which PyMC refuses for a
    model carrying non-default initial values. Clearing them for the duration
    sidesteps that: it affects graph rebuilding only, never the posterior already
    drawn. They are restored on exit, so a model is unchanged by having been scored.

    Parameters
    ----------
    model : pymc.Model
        The model whose initial values are cleared.

    Yields
    ------
    pymc.Model
        The same model, with its initial values cleared for the duration.
    """
    saved = dict(model.rvs_to_initial_values)
    try:
        for variable in model.rvs_to_initial_values:
            model.rvs_to_initial_values[variable] = None
        yield model
    finally:
        model.rvs_to_initial_values.update(saved)


def available_algorithms() -> tuple:
    """Every ratingcurve estimator family this wrapper can construct.

    Returns
    -------
    tuple of str
        Names accepted as the `algorithm` argument: ``"power_law"`` and
        ``"spline"``. ratingcurve's experimental parameterizations (Reitan, Le Coz,
        ISO, and the two broken power laws) are not offered here.
    """
    return tuple(_algorithm_classes())


def _algorithm_classes() -> dict:
    """Map algorithm name to the ratingcurve class, imported on demand."""
    from ratingcurve import ratings
    return {
        "power_law": ratings.PowerLawRating,
        "spline": ratings.SplineRating,
    }


class BayesianRating:
    """One ratingcurve model, fittable and evaluable through a uniform interface.

    Most callers want :func:`limnotech_rating_curves.fit_rating` instead; this is
    the layer underneath it, for when you need the ratingcurve model object
    itself.

    Parameters
    ----------
    algorithm : str, default 'power_law'
        Which estimator family (see :func:`available_algorithms`).
    **model_kwargs
        Forwarded to the underlying model's constructor. The relevant knob per
        family: ``segments`` (int) and ``prior`` (dict) for the power-law family,
        ``df`` (interior knots), ``mean`` and ``sd`` for the spline. A keyword a
        family ignores is harmless.

    Attributes
    ----------
    algorithm : str
        The family this instance was built for.
    model : object
        The underlying ratingcurve model.
    idata : arviz.InferenceData or None
        The posterior, once fitted.
    meta : dict
        Record of how it was configured and fitted.
    """

    def __init__(self, algorithm: str = "power_law", **model_kwargs):
        classes = _algorithm_classes()
        if algorithm not in classes:
            raise ValueError(f"algorithm must be one of {tuple(classes)}, "
                             f"got {algorithm!r}")
        self.algorithm = algorithm
        self.model_kwargs = model_kwargs
        self.model = classes[algorithm](**model_kwargs)
        self.idata = None
        self.meta = {"algorithm": algorithm, **model_kwargs}

    def fit(self, stage, discharge, *, discharge_sigma=None, method: str = "nuts",
            seed: int = settings.SEED, progressbar: bool = False,
            **sampler_kwargs) -> "BayesianRating":
        """Fit the model to measured stage and discharge.

        Parameters
        ----------
        stage, discharge : array-like
            Gage height (feet) and discharge (cfs).
        discharge_sigma : array-like, optional
            One-sigma discharge measurement error (cfs), one per measurement.
            Supplying it meaningfully sharpens the posterior, because the model
            can then separate measurement error from rating error.
        method : {'nuts', 'advi'}, default 'nuts'
            Full MCMC, or the faster variational approximation.
        seed : int
            RNG seed.
        progressbar : bool, default False
            Show the sampler's progress bar.
        **sampler_kwargs
            Passed through to the sampler (``draws``, ``tune``, ``chains``,
            ``cores``, ``target_accept``, ``nuts_sampler``, ``n`` for ADVI).
            ratingcurve's own defaults apply to anything not named here; the
            package-level :func:`fit` fills in ``target_accept`` from the sample
            size, so a short record does not need it set by hand.

        Returns
        -------
        BayesianRating
            Self, fitted.
        """
        stage = np.asarray(stage, float).ravel()
        discharge = np.asarray(discharge, float).ravel()
        if discharge_sigma is not None:
            discharge_sigma = np.asarray(discharge_sigma, float).ravel()
        self.idata = self.model.fit(h=stage, q=discharge, q_sigma=discharge_sigma,
                                   method=method, random_seed=seed,
                                   progressbar=progressbar, **sampler_kwargs)
        self.meta.update({"method": method, "n": int(stage.size)})
        return self

    def predict(self, stage) -> np.ndarray:
        """Posterior-mean discharge at the given stage(s).

        Parameters
        ----------
        stage : array-like
            Gage height (feet).

        Returns
        -------
        numpy.ndarray
            Discharge (cfs), one value per requested stage.
        """
        stages = np.atleast_1d(np.asarray(stage, float)).ravel()
        # The spline builds a basis matrix per prediction and hands it to
        # pm.set_data, which requires the same rank as at fit time. numpy collapses a
        # one-row basis to 1-D, so a single-stage prediction raises "New values for
        # 'B' must have 2 dimensions". Predicting a duplicated pair keeps it 2-D and
        # costs one extra row; the answer is unchanged.
        padded = np.repeat(stages, 2) if stages.size == 1 else stages
        predicted = np.asarray(self.model.predict(padded), float).ravel()
        return predicted[:1] if stages.size == 1 else predicted

    def posterior_draws(self, stage) -> np.ndarray:
        """Draws of the fitted rating itself, without gaging scatter.

        ``predict_posterior`` draws from the likelihood, so each of its draws is the
        rating *plus* a simulated field-measurement error. These are draws of the
        rating alone: the model's mean function evaluated at each posterior draw of
        the parameters.

        The two differ once discharge leaves log space. A measurement error that is
        symmetric in logs is not symmetric in cfs, so averaging ``predict_posterior``
        gives the rating multiplied by ``exp(sigma ** 2 / 2)`` rather than the rating.
        Their medians agree, which is what
        ``tests/test_posterior_mean.py`` checks. See ``posterior_clarification.md``.

        Parameters
        ----------
        stage : array-like
            Gage height (feet).

        Returns
        -------
        numpy.ndarray
            Discharge (cfs), shaped ``(stages, draws)`` as ``predict_posterior`` is.
            A draw whose stage of zero flow sits above the stage gives exactly zero,
            the same value ``predict_posterior`` reports there.
        """
        stages = np.atleast_1d(np.asarray(stage, float)).ravel()
        return self.model.q_transform.untransform(self._mean_log_z(stages))

    def posterior_mean(self, stage) -> np.ndarray:
        """Posterior-mean discharge of the fitted rating, ``E[exp(mu)]``.

        The mean of :meth:`posterior_draws`. Averages over everything the fit leaves
        uncertain about the curve, and nothing else - unlike :meth:`predict`, which
        also folds in the error of a hypothetical future gaging.

        Parameters
        ----------
        stage : array-like
            Gage height (feet).

        Returns
        -------
        numpy.ndarray
            Discharge (cfs), one value per requested stage.
        """
        return np.asarray(self.posterior_draws(stage), float).mean(axis=1)

    def _mean_log_z(self, stages) -> np.ndarray:
        """The model's mean function at `stages`, in the standardized log space it is
        sampled in, shaped ``(stages, draws)``.

        Mirrors the mean each algorithm hands to its likelihood, so the two cannot
        drift apart: the power law's ``a + dot(b, X)`` and the spline's ``dot(B, w)``.
        """
        posterior = self.idata.posterior
        flatten = lambda name: (posterior[name]
                                .stack(sample=("chain", "draw")).values)

        if self.algorithm == "spline":
            # The design matrix is the same for every draw, so it is built once from
            # the very Dmatrix the fit used rather than reassembled here.
            basis = np.asarray(self.model.d_transform(stages), float)
            return basis @ flatten("w")

        # (segments, draws) for both; hs carries a trailing length-1 axis
        exponents = flatten("b").reshape(self.model.segments, -1)
        breakpoints = flatten("hs").reshape(self.model.segments, -1)
        intercept = flatten("a")
        offsets = np.asarray(self.model.ho, float).reshape(-1, 1, 1)

        # log(max(h - hs, 0) + ho), the model's own X, per draw
        depth = np.clip(stages[None, None, :] - breakpoints[:, :, None], 0, None)
        with np.errstate(divide="ignore"):
            basis = np.log(depth + offsets)
        # sum over segments -> (draws, stages), then to (stages, draws)
        return (intercept[:, None]
                + np.einsum("sd,sdx->dx", exponents, basis)).T

    def table(self, stage=None, step: float = 0.01, extend: float = 1.1) -> pd.DataFrame:
        """The fitted rating as a tidy table with its uncertainty.

        Parameters
        ----------
        stage : array-like, optional
            Stages to evaluate at. Defaults to a fine grid spanning the observed
            range times `extend`.
        step : float, default 0.01
            Grid spacing (feet) when `stage` is not given.
        extend : float, default 1.1
            How far past the observed stage range the default grid runs.

        Returns
        -------
        pandas.DataFrame
            Columns ``stage_ft``; ``discharge_cfs``, the posterior mean of the rating
            (:meth:`posterior_mean`); ``discharge_posterior_median_cfs``,
            ``posterior_lower`` and ``posterior_upper``, the median and 95% interval
            of the rating itself (:meth:`posterior_draws`);
            ``discharge_predictive_mean_cfs`` and
            ``discharge_predictive_median_cfs``, the two summaries ratingcurve reports
            of the posterior *predictive*; ``gse`` (geometric standard error); and
            ``lower`` and ``upper`` (the 95% predictive interval,
            ``median * gse**±1.96``).

            The two interval pairs answer different questions: ``posterior_lower`` /
            ``posterior_upper`` say where the rating is, and ``lower`` / ``upper`` say
            where the next gaging would fall. The second is always the wider.

        Notes
        -----
        ``discharge_cfs`` is the rating, not the posterior predictive mean. The two
        differ by ``exp(sigma ** 2 / 2)`` - negligible where sigma is small, and
        unbounded where it is not. The predictive mean is still reported, in its own
        column. See ``posterior_clarification.md``.

        ``ratingcurve.table`` drops every row whose mean discharge exceeds
        ``extend`` times the largest measured discharge, which cuts a curve off
        below the top of the stage grid it was asked for. When `stage` is given the
        table is built here from ``predict_posterior`` instead, so the requested
        stages all come back and a curve covers the same range whichever family
        fitted it.
        """
        if stage is None:
            table = self.model.table(step=step, extend=extend)
        else:
            stage = np.atleast_1d(np.asarray(stage, float)).ravel()
            if stage.size == 1:
                stage = np.repeat(stage, 2)   # see predict(): keeps the basis 2-D
            table = self._posterior_table(stage)
        table = table.rename(columns=_TABLE_RENAME).reset_index(drop=True)
        median = table["discharge_predictive_median_cfs"].to_numpy(float)
        gse = table["gse"].to_numpy(float)
        draws = self.posterior_draws(table["stage_ft"].to_numpy(float))
        table["discharge_cfs"] = draws.mean(axis=1)
        table["discharge_posterior_median_cfs"] = np.median(draws, axis=1)
        table["posterior_lower"], table["posterior_upper"] = np.percentile(
            draws, [2.5, 97.5], axis=1)
        table["lower"] = median * gse ** (-_Z95)
        table["upper"] = median * gse ** _Z95
        return table.sort_values("stage_ft").reset_index(drop=True)

    def _posterior_table(self, stage) -> pd.DataFrame:
        """The rating at the given stages, summarized as ratingcurve's table is.

        The same four columns ``ratingcurve.table`` builds from
        ``predict_posterior``, without its discharge ceiling.
        """
        posterior = np.asarray(
            self.model.predict_posterior(np.asarray(stage, float),
                                         extend_idata=False), float)
        # below the stage of zero flow a draw predicts exactly zero, whose log is
        # undefined; the geometric standard error is not defined there either
        positive = posterior > 0
        log_posterior = np.where(positive, np.log(np.where(positive, posterior, 1.0)),
                                 np.nan)
        with np.errstate(invalid="ignore"):
            gse = np.exp(np.std(log_posterior, axis=1))
        return pd.DataFrame({
            "stage": stage,
            "discharge": np.mean(posterior, axis=1),
            "median": np.median(posterior, axis=1),
            "gse": np.where(positive.all(axis=1), gse, np.nan)})

    def equation(self) -> dict:
        """The denormalized power-law parameters, in ratingcurve's own form.

        A passthrough to ``ratingcurve``'s :meth:`PowerLawRating.equation`, which
        undoes the standardization the model fits in and returns the parameters of

        .. math::

            \\ln q = a + \\sum_i b_i \\ln\\!\\big(\\max(h - hs_i,\\,0) + ho_i\\big)

        with :math:`ho_0 = 0` and :math:`ho_i = 1` for :math:`i > 0`.

        This is the log-space form the upstream package documents and tests, and the
        only form this package reports. It is exposed unchanged so a caller can quote
        the same parameters the library does. Exponentiating it gives the equivalent
        multiplicative form,

        .. math::

            Q = e^{a}\\,(h - hs_0)^{b_0}
                \\prod_{i>0} \\big(1 + \\max(h - hs_i,\\,0)\\big)^{b_i},

        which is left to the caller rather than returned as a second dict of numbers:
        one set of parameters, in one notation, cannot be quoted against the wrong
        form.

        Returns
        -------
        dict
            The parameters, exactly as upstream returns them:

            ``a``
                Float intercept.
            ``b``
                Exponent per segment.
            ``hs``
                Breakpoint stage per segment.

            and, so the numbers cannot be misread, the notation they belong to:

            ``ho``
                The offset per segment that the form refers to - ``0`` for the first
                segment and ``1`` thereafter. Returned rather than left to the reader
                so the dict can be evaluated without consulting this docstring.
            ``symbolic``
                The form itself, as text:
                ``"ln(q) = a + sum(b[i] * ln(max(h - hs[i], 0) + ho[i]))"``.
            ``numeric``
                The same with the fitted values substituted in, to six significant
                figures - the line to quote in a report.

            Empty for the spline, which has no such form, and before the fit.

        Notes
        -----
        These are posterior *mean* parameters - the equation evaluated at the average
        of the draws, which is not the average of the evaluated equation. The curve
        they describe tracks the ``discharge_predictive_median_cfs`` column of
        :meth:`table` more closely than ``discharge_cfs``, and it puts one sharp kink
        at the mean breakpoint where the draws smear theirs over a range of stages.
        Upstream's own test allows 15% for exactly this reason. Read a segmented curve
        near its kink off :meth:`table`, not off these parameters.
        """
        if self.algorithm != "power_law" or not self.fitted:
            return {}
        params = self.model.equation()
        intercept = float(params["a"])
        exponents = np.asarray(params["b"], float).ravel()
        breakpoints = np.asarray(params["hs"], float).ravel()
        offsets = np.ones(exponents.size)
        offsets[0] = 0.0
        return {"a": intercept, "b": exponents, "hs": breakpoints, "ho": offsets,
                "symbolic": POWER_LAW_FORM,
                "numeric": _substituted_form(intercept, exponents, breakpoints,
                                             offsets)}

    def summary(self, var_names=None) -> pd.DataFrame:
        """ArviZ posterior summary of the fitted parameters.

        Parameters
        ----------
        var_names : list of str, optional
            Restrict to these parameters.

        Returns
        -------
        pandas.DataFrame
            Mean, standard deviation, credible interval, R-hat and ESS per
            parameter.
        """
        return self.model.summary(var_names=var_names)

    def residuals(self) -> np.ndarray:
        """Log residuals at the measurements, ``log(observed) - log(predicted)``.

        Returns
        -------
        numpy.ndarray
        """
        return np.asarray(self.model.residuals(), float).ravel()

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run successfully.

        Read from the posterior rather than from ratingcurve's ``is_fitted_`` flag,
        which its rating models declare but never set - the same condition its own
        ``@is_fit`` decorator tests.
        """
        return self.idata is not None

    def save(self, path) -> str:
        """Write the fitted model to a NetCDF file that can be reloaded.

        ratingcurve's own serialization: the posterior plus the model and sampler
        configuration in the file's attributes, so the model class's ``load()`` can
        rebuild a working object.

        Parameters
        ----------
        path : path-like
            Destination ``.nc`` file.

        Returns
        -------
        str
            The path written.
        """
        with coerced_netcdf_attrs(self.idata):
            self.model.save(str(path))
        return str(path)


def breakpoint_prior(stage, discharge, segments: int, zero_flow=None) -> dict:
    """Build ratingcurve's normal prior on the power-law breakpoints.

    The power-law family places one breakpoint per segment, and the first of them
    is the **stage of zero flow** - the parameter a short record identifies worst.
    Centering it on an independent estimate is the single most useful prior you can
    supply, so that is what this does; any further breakpoints are spread evenly
    across the observed stage range, which is a deliberately uninformative choice.

    Parameters
    ----------
    stage, discharge : array-like
        Gage height (feet) and discharge (cfs).
    segments : int
        Number of power-law segments, hence of breakpoints.
    zero_flow : None or float or ZeroFlowEstimate or str, optional
        Where to center the first breakpoint.

        * ``None`` or ``"johnson"`` - estimate it with
          :func:`~limnotech_rating_curves.data.zero_flow.estimate_zero_flow`, which
          uses Johnson's three-point method and falls back to just below the
          lowest measurement.
        * a number or a
          :class:`~limnotech_rating_curves.data.zero_flow.ZeroFlowEstimate` - use that
          value. It must be below the lowest observed stage.
        * ``"infer"`` - do not use an informative estimate: center the first
          breakpoint just below the lowest measurement and let the data decide.

    Returns
    -------
    dict
        The ``prior`` argument for a ratingcurve power-law model:
        ``{"distribution": "normal", "mu": [...], "sigma": [...]}``, with the
        breakpoint means in feet of stage.

        The first breakpoint's ``sigma`` is the binding choice. The model evaluates
        :math:`(h - e)^{\\beta}`, so the prior must not put meaningful mass on an
        ``e`` at or above the lowest measurement - that makes the base negative and
        the log-density NaN, and the sampler fails outright rather than recovering.
        So it is capped at half the *headroom* between the estimate and the lowest
        measurement, as well as at half the observed stage range. When zero flow sits
        just below the lowest measurement that headroom is small, and a tight prior
        there is not over-confidence but the constraint the model itself imposes.
        Later breakpoints fall inside the observed range and are not constrained
        this way, so they keep the looser half-range sigma.

    Raises
    ------
    ValueError
        If an explicit `zero_flow` is not below the lowest observed stage, which
        would make ``h - e`` non-positive and the power law undefined there.
    """
    stage = np.asarray(stage, float).ravel()
    lowest, highest = float(np.nanmin(stage)), float(np.nanmax(stage))
    observed_range = (highest - lowest) or 1.0

    keyword = zero_flow.strip().lower() if isinstance(zero_flow, str) else None
    if keyword in ("infer", "none", "estimate"):
        # no informative estimate: just below the lowest measurement, which is the
        # loosest placement that still keeps (h - e) positive everywhere
        zero_flow_ft = lowest - 0.1 * observed_range
    elif zero_flow is None or keyword in ("johnson", "auto"):
        zero_flow_ft = float(estimate_zero_flow(stage, discharge).stage_ft)
    elif isinstance(zero_flow, ZeroFlowEstimate):
        zero_flow_ft = float(zero_flow.stage_ft)
    elif keyword is not None:
        raise ValueError(f"unknown zero_flow {zero_flow!r}; pass a number, "
                         f"'johnson', or 'infer'")
    else:
        zero_flow_ft = float(zero_flow)

    if zero_flow_ft >= lowest:
        raise ValueError(
            f"zero_flow = {zero_flow_ft:.4f} ft is not below the lowest observed "
            f"stage {lowest:.4f} ft, so (h - e) would be non-positive; the stage of "
            f"zero flow must sit below every measurement")

    loose = 0.5 * observed_range
    headroom = lowest - zero_flow_ft            # positive, checked above
    zero_flow_sigma = max(min(loose, 0.5 * headroom), 1e-3 * observed_range)

    means = [zero_flow_ft]
    sigmas = [zero_flow_sigma]
    for index in range(1, int(segments)):
        means.append(lowest + observed_range * index / segments)
        sigmas.append(loose)
    return {"distribution": "normal", "mu": means, "sigma": sigmas}


def adapt_config(algorithm: str, segments, min_points: int, n: int, *,
                 enforce_min_points: bool = True) -> tuple:
    """Decide whether a model can be fitted to `n` measurements, and how.

    Model complexity has to follow sample size: a two-piece rating needs more
    measurements than a single power law, and a spline cannot have more knots than
    it has points to place them between. Rather than fitting an over-parameterized
    model and reporting a meaningless score, a sample too small for a model is
    skipped with a reason.

    Parameters
    ----------
    algorithm : {'power_law', 'spline'}
        Which family.
    segments : int or None
        Power-law segment count; ``None`` for the spline.
    min_points : int
        Fewest measurements this model needs.
    n : int
        Measurements available.
    enforce_min_points : bool, default True
        When False, attempt the fit at any sample size and let only a genuine
        failure stop it. Cross-validation folds use this, since a fold is
        deliberately smaller than the sample the minimum was set for.

    Returns
    -------
    should_fit : bool
        Whether to attempt the fit.
    config : dict
        Model keywords - ``{"segments": n}`` or ``{"df": knots}``.
    reason : str
        Why it was skipped, when `should_fit` is False.
    """
    if enforce_min_points and n < min_points:
        what = (f"a {segments}-segment power law" if algorithm == "power_law"
                else "a natural spline")
        return False, {}, f"n={n} is below the {min_points} measurements {what} needs"
    if algorithm == "power_law":
        return True, {"segments": int(segments)}, ""
    knots = max(1, min(settings.DEFAULT_SPLINE_KNOTS, n - 2))
    return True, {"df": knots}, ""


def fit(sample, *, key: str, label: str, algorithm: str, segments=None,
        min_points: int = 3, method: str = "nuts", seed: int = settings.SEED,
        cores=None, advi_iters: int = settings.ADVI_ITERS,
        nuts_sampler: str = None, target_accept: float = None, zero_flow=None,
        enforce_min_points: bool = True, config_override=None,
        with_log_likelihood: bool = True, grid=None) -> FitResult:
    """Fit one ratingcurve model to one sample. Never raises.

    Any failure is captured in the returned result's ``status`` and ``reason``, so
    a batch run over many sites reports what went wrong per model instead of
    aborting.

    Parameters
    ----------
    sample : Sample or pandas.DataFrame
        The measurements.
    key, label : str
        Identifier and display name recorded on the result.
    algorithm : {'power_law', 'spline'}
        Which family to fit.
    segments : int, optional
        Power-law segment count. Ignored for the spline.
    min_points : int, default 3
        Fewest measurements this model needs (see :func:`adapt_config`).
    method : {'nuts', 'advi'}, default 'nuts'
        Full MCMC, or the fast variational fit. NUTS is required for a
        trustworthy ELPD and Pareto-k.
    seed : int
        RNG seed.
    cores : int, optional
        NUTS chains to run in parallel. ``None`` leaves PyMC's default; pass 1
        when this call is already inside a worker process, so the sampler does not
        spawn a nested pool.
    advi_iters : int
        ADVI optimization steps, when ``method="advi"``. ratingcurve installs
        convergence callbacks that can stop early, so this is a ceiling.
    nuts_sampler : str, optional
        Which NUTS implementation (see ``settings.NUTS_SAMPLERS``). Recorded on
        the result, because ratingcurve builds its own stored sampler config
        before this keyword is merged in and so never sees it.
    target_accept : float, optional
        NUTS target acceptance rate. ``None`` (the default) resolves it from the
        sample size with :func:`settings.target_accept_for`, which raises it for a
        short record; pass a number to sample at that rate regardless.
    zero_flow : None or float or ZeroFlowEstimate or str, optional
        Stage of zero flow to center the breakpoint prior on; see
        :func:`breakpoint_prior`. Power-law family only.
    enforce_min_points : bool, default True
        Apply the per-model minimum-sample rule.
    config_override : dict, optional
        Merged over the adapted config, so a caller can pin e.g. the spline's
        ``df`` instead of taking the automatic choice.
    with_log_likelihood : bool, default True
        Compute the pointwise log-likelihood immediately after sampling, while the
        model still holds the data it was fitted on. This has to happen here rather
        than on demand: ratingcurve's ``predict`` sets new stage data on the PyMC
        model and leaves it there, so a later
        :func:`~limnotech_rating_curves.evaluate.metrics.ensure_log_likelihood` would
        try to score the fitted observations against whatever stages were predicted
        last, and fail on the shape mismatch. Turn it off only to save the time and
        memory when the fit will never be scored.
    grid : array-like, optional
        Stage grid the fitted curve is tabulated on. Defaults to a padded grid
        over the observed range, the same one bdrc and the least-squares forms
        are given, so every family's curve covers the same stages.

    Returns
    -------
    FitResult
    """
    from ..core import Sample
    frame = sample.to_frame() if isinstance(sample, Sample) else pd.DataFrame(sample)
    stage = frame["stage_ft"].to_numpy(float)
    discharge = frame["discharge_cfs"].to_numpy(float)
    n = len(frame)
    nuts_sampler = settings.NUTS_SAMPLER if nuts_sampler is None else nuts_sampler
    target_accept = (settings.target_accept_for(n) if target_accept is None
                     else float(target_accept))

    should_fit, config, reason = adapt_config(
        algorithm, segments, min_points, n, enforce_min_points=enforce_min_points)
    if config_override:
        config = {**config, **config_override}

    result = FitResult(key=key, label=label, family="ratingcurve", n=n,
                       status="skipped", reason=reason, config=config)
    if not should_fit:
        return result

    try:
        if method == "advi":
            sampler_kwargs = {"draws": settings.ADVI_DRAWS, "n": advi_iters}
            result.config = {**result.config, "method": "advi", "advi_iters": advi_iters}
        else:
            sampler_kwargs = {"draws": settings.NUTS_DRAWS, "tune": settings.NUTS_TUNE,
                              "nuts_sampler": nuts_sampler,
                              "target_accept": target_accept}
            if cores is not None:
                sampler_kwargs["cores"] = cores
            result.config = {**result.config, "method": "nuts",
                             "nuts_sampler": nuts_sampler,
                             "target_accept": target_accept}

        model_kwargs = dict(config)
        if algorithm == "power_law":
            prior = breakpoint_prior(stage, discharge, model_kwargs["segments"],
                                     zero_flow=zero_flow)
            model_kwargs["prior"] = prior
            result.config = {**result.config, "zero_flow_prior_ft": prior["mu"][0]}
        else:
            model_kwargs.setdefault("df", settings.DEFAULT_SPLINE_KNOTS)

        rating = BayesianRating(algorithm, **model_kwargs)
        rating.fit(stage, discharge, method=method, seed=seed, **sampler_kwargs)

        # Before table() or predict(), both of which push new stage data onto the
        # model and leave it there. ADVI fits are not scored by policy, so there is
        # nothing to compute for them.
        if with_log_likelihood and method != "advi":
            from ..evaluate.metrics import ensure_log_likelihood
            try:
                ensure_log_likelihood(rating)
            except Exception as exc:  # noqa: BLE001 - scoring is not the fit
                log.info("%s: could not attach a log-likelihood (%s: %s)",
                         key, type(exc).__name__, exc)

        table = rating.table(stage=padded_stage_grid(stage) if grid is None
                             else np.asarray(grid, float))
        # the grid is padded below the measurements, and a power law has no discharge
        # below its stage of zero flow: those rows are dropped rather than tabulated
        # as zero or infinity, which is where bdrc's curve stops too
        drawable = np.isfinite(table["discharge_cfs"]) & (table["discharge_cfs"] > 0)
        table = table[drawable].reset_index(drop=True)
        predicted = rating.posterior_mean(stage)
        result.rating = rating
        result.idata = rating.idata
        result.curve = table[["stage_ft", "discharge_cfs",
                              "discharge_posterior_median_cfs",
                              "posterior_lower", "posterior_upper",
                              "discharge_predictive_mean_cfs",
                              "discharge_predictive_median_cfs", "lower", "upper"]]
        result.predicted = predicted
        result.metrics = fit_metrics(discharge, predicted)
        result.status = "ok"
        result.reason = ""
    except Exception as exc:  # noqa: BLE001 - a failed model must not sink the run
        result.status = "failed"
        result.reason = f"{type(exc).__name__}: {str(exc)[:160]}"
        log.debug("%s failed on n=%d: %s", key, n, result.reason)
    return result
