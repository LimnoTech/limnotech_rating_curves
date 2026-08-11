import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .models import catalog
from . import settings
from .core import Metrics, Sample

log = logging.getLogger(__name__)


def _quantile_z(level: float) -> float:
    """The standard-normal quantile bounding a central interval of `level`."""
    return NormalDist().inv_cdf((1 + float(level)) / 2)


class RatingModel:
    """Base class for the rating estimators - not used directly.

    Every estimator shares this interface, so swapping a power law for a spline or
    a bdrc model changes one line and nothing else. :class:`RatingSet` answers to
    the same names, one row or one series per model, so there is one set of method
    names to learn rather than two:

    ==========================  ================================================
    ``fit(data, ...)``          fit to measurements, returns self
    ``predict(stage)``          posterior-mean discharge
    ``interval(stage, level)``  credible interval for discharge
    ``curve(stage, level)``     the whole fitted rating as a table
    ``metrics``                 fit and predictive scores
    ``cross_validate()``        refit on subsets and score the held-out points
    ``summary()``               posterior summary of the parameters
    ``equation()``              the fitted curve written out, where it has a
                                closed form
    ``diagnostics()``           convergence diagnostics (R-hat, ESS)
    ``plot(ax)``                measurements, curve and band
    ``plot_residuals(ax)``      predicted / observed against stage
    ``save_posterior(dir)``     write the posterior to NetCDF
    ==========================  ================================================

    Attributes
    ----------
    sample : Sample or None
        The measurements this was fitted to.
    """

    #: The catalog entry this estimator wraps. Set by each subclass.
    entry: catalog.ModelEntry

    def __init__(self):
        self._fit = None
        self.sample: Sample | None = None
        self.fit_arguments: dict = {}
        self._config_override: dict | None = None

    # -- identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        """Stable model key, e.g. ``"power_law_2seg"``."""
        return self.entry.key

    @property
    def label(self) -> str:
        """Display name for legends and tables."""
        return self.entry.label

    @property
    def color(self) -> str:
        """The color this model is drawn in, everywhere."""
        return self.entry.color

    @property
    def family(self) -> str:
        """Which backend fits it: ``"ratingcurve"`` or ``"bdrc"``."""
        return self.entry.family

    @property
    def fitted(self) -> bool:
        """True once a successful fit has been made."""
        return self._fit is not None and self._fit.ok

    @property
    def result(self):
        """The underlying :class:`~limnotech_rating_curves.core.FitResult`."""
        return self._fit

    def _require_fit(self):
        if not self.fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted - "
                               f"call .fit(measurements) first")

    # -- fitting --------------------------------------------------------------

    def fit(self, data, discharge=None, *, stage=None, time=None, stage_datum=None,
            method: str = "nuts", seed: int = settings.SEED, zero_flow=None,
            cores=None, advi_iters: int = settings.ADVI_ITERS, nuts_sampler=None,
            with_log_likelihood: bool = True) -> "RatingModel":
        """Fit the model to measured stage and discharge.

        Parameters
        ----------
        data : Sample or pandas.DataFrame or array-like or pagaia Station
            The measurements. A DataFrame needs a stage column and a discharge
            column (see :meth:`Sample.of` for the names recognized); an array of
            stage values needs `discharge` as well.
        discharge : array-like or str, optional
            Discharge (cfs) when `data` is an array, or the discharge column name
            when `data` is a DataFrame.
        stage : str, optional
            Name of the stage column, when `data` is a DataFrame whose stage column
            is not one of the recognized names.
        time : array-like or str, optional
            Measurement timestamps, or the name of the timestamp column. Needed only
            when `stage_datum` is a timeseries of reference elevations.
        stage_datum : optional
            Convert stage onto a gage-height reference before fitting. Pass the
            elevation of gage height zero, ``"lowest"``, a timeseries of reference
            elevations, or a
            :class:`~limnotech_rating_curves.data.datum.StageDatum`. See
            :mod:`limnotech_rating_curves.data.datum`.
        method : {'nuts', 'advi'}, default 'nuts'
            ``"nuts"`` is full MCMC and is required for a trustworthy predictive
            score; ``"advi"`` is a fast variational approximation, for quick looks.
        seed : int
            RNG seed. The same seed gives the same fit.
        zero_flow : optional
            What to do about the stage of zero flow, the parameter a short record
            identifies worst:

            * ``None`` (default) - each family does what suits it. The power law
              centers its breakpoint prior on Johnson's three-point estimate; bdrc
              infers the parameter from the data under its own priors.
            * ``"johnson"`` - use Johnson's estimate explicitly. For the power law
              that is a prior mean; for bdrc it **fixes** the parameter.
            * a number or a
              :class:`~limnotech_rating_curves.data.zero_flow.ZeroFlowEstimate` - the
              same, with your value.
            * ``"infer"`` - no informative offset prior at all.

            See :mod:`limnotech_rating_curves.data.zero_flow`.
        cores : int, optional
            NUTS chains to run in parallel. Pass 1 when fitting inside a worker
            process, so the sampler does not spawn a nested process pool.
        advi_iters : int
            ADVI optimization steps, when ``method="advi"``.
        nuts_sampler : str, optional
            Which NUTS implementation walks the posterior (see
            ``settings.NUTS_SAMPLERS``). All of them sample the same posterior;
            only the wall clock changes.
        with_log_likelihood : bool, default True
            Keep the pointwise log-likelihood, which is what the predictive score
            needs. Turn it off only to save memory when you will not score the fit.

        Returns
        -------
        RatingModel
            Self, fitted.

        Raises
        ------
        ValueError
            If the sample is empty.
        RuntimeError
            If this model could not be fitted to this sample, with the backend's
            reason.
        """
        sample = _as_sample(data, discharge, stage=stage, time=time,
                            stage_datum=stage_datum)
        if len(sample) == 0:
            raise ValueError(f"no usable measurements to fit "
                             f"({sample.skipped or 'the sample is empty'})")
        self._fit = self.entry.fit(
            sample, method=method, seed=seed, zero_flow=zero_flow, cores=cores,
            advi_iters=advi_iters, nuts_sampler=nuts_sampler,
            config_override=self._config_override,
            with_log_likelihood=with_log_likelihood)
        if not self._fit.ok:
            raise RuntimeError(f"{self.name} could not be fitted: {self._fit.reason}")
        self.sample = sample
        # kept so a refit on a subset can reproduce this fit rather than a default one
        # (see limnotech_rating_curves.evaluate.exact_loo)
        self.fit_arguments = {"method": method, "seed": seed, "zero_flow": zero_flow,
                              "cores": cores, "advi_iters": advi_iters,
                              "nuts_sampler": nuts_sampler}
        return self

    # -- prediction -----------------------------------------------------------

    def predict(self, stage):
        """Posterior-mean discharge at the given stage.

        Parameters
        ----------
        stage : float or array-like
            Gage height (feet), on the same axis the model was fitted on.

        Returns
        -------
        float or numpy.ndarray
            Discharge (cfs) - a float for a scalar stage, else an array. Stages
            outside what the fitted curve covers come back NaN rather than being
            silently extrapolated by interpolation.
        """
        self._require_fit()
        scalar = np.ndim(stage) == 0
        stages = np.atleast_1d(np.asarray(stage, float))
        if self._fit.rating is not None:
            # a live ratingcurve model, or a fitted polynomial, evaluates its own
            # parameters exactly rather than interpolating a tabulated curve
            predicted = self._fit.rating.predict(stages)
        else:
            curve = self._fit.curve.sort_values("stage_ft")
            predicted = np.interp(stages, curve["stage_ft"], curve["discharge_cfs"],
                                  left=np.nan, right=np.nan)
        return float(predicted[0]) if scalar else predicted

    def interval(self, stage, level: float = 0.95):
        """The credible interval for discharge at the given stage.

        Parameters
        ----------
        stage : float or array-like
            Gage height (feet).
        level : float, default 0.95
            Interval width. Honored exactly for the ratingcurve family, whose
            posterior is summarized by a geometric standard error; bdrc reports a
            fixed 95% predictive band, so other levels are approximated from it.

        Returns
        -------
        tuple
            ``(lower, upper)`` - two floats for a scalar stage, two arrays
            otherwise. Discharge in cfs.
        """
        curve = self.curve(np.atleast_1d(np.asarray(stage, float)), level=level)
        lower = curve["lower"].to_numpy(float)
        upper = curve["upper"].to_numpy(float)
        if np.ndim(stage) == 0:
            return float(lower[0]), float(upper[0])
        return lower, upper

    def curve(self, stage=None, level: float = 0.95) -> pd.DataFrame:
        """The fitted rating as a table.

        Parameters
        ----------
        stage : array-like, optional
            Stages to evaluate at. Defaults to the padded grid the fit was
            tabulated on, which is the same grid for every family.
        level : float, default 0.95
            Width of the reported interval.

        Returns
        -------
        pandas.DataFrame
            Columns ``stage_ft``, ``discharge_cfs`` (posterior mean),
            ``discharge_median_cfs``, ``lower``, ``upper``.
        """
        self._require_fit()
        if stage is None and self._fit.curve is not None:
            stage = self._fit.curve["stage_ft"].to_numpy(float)
        if self.family in ("polynomial", "exponential"):
            # a least-squares fit evaluates its own coefficients and reports a
            # prediction interval, so it builds the whole table itself
            return self._fit.rating.table(stage=stage, level=level)
        if self._fit.rating is not None:
            grid = None if stage is None else np.atleast_1d(np.asarray(stage, float))
            table = self._fit.rating.table(stage=grid)
            median = table["discharge_median_cfs"].to_numpy(float)
            gse = table["gse"].to_numpy(float)
            z = _quantile_z(level)
            evaluated = pd.DataFrame({
                "stage_ft": table["stage_ft"].to_numpy(float),
                "discharge_cfs": table["discharge_cfs"].to_numpy(float),
                "discharge_median_cfs": median,
                "lower": median * gse ** (-z), "upper": median * gse ** z})
            # the backend pads a single-stage request to keep the spline basis 2-D
            if grid is not None and grid.size == 1:
                evaluated = evaluated.iloc[:1].reset_index(drop=True)
            return evaluated

        fitted = self._fit.curve.sort_values("stage_ft")
        if stage is None:
            return fitted.reset_index(drop=True)
        stages = np.asarray(stage, float)
        return pd.DataFrame({
            "stage_ft": stages,
            "discharge_cfs": np.interp(stages, fitted["stage_ft"],
                                       fitted["discharge_cfs"]),
            "discharge_median_cfs": np.interp(stages, fitted["stage_ft"],
                                              fitted["discharge_median_cfs"]),
            "lower": np.interp(stages, fitted["stage_ft"], fitted["lower"]),
            "upper": np.interp(stages, fitted["stage_ft"], fitted["upper"])})

    @property
    def zero_flow(self) -> float:
        """The fitted stage of zero flow, in feet on the sample's stage axis.

        The stage at which the rating predicts no discharge - the parameter written
        :math:`e` in :math:`Q = C(h-e)^{\\beta}` and ``c`` in bdrc. Subtracting it
        gives effective head, which is what straightens a power law on log-log axes.

        Returns
        -------
        float
            The posterior mean, the value it was fixed at if it was fixed, or NaN for
            a model that has no such parameter (a spline, or the spreadsheet forms).
        """
        self._require_fit()
        posterior = getattr(self._fit.idata, "posterior", None)
        if posterior is not None and "hs" in posterior:
            # ratingcurve: hs[0, 0] is the stage of zero flow
            return float(posterior["hs"].mean(dim=("chain", "draw"))[0, 0])
        config = self._fit.config
        for key in ("zero_flow_fitted_ft", "zero_flow_ft"):    # bdrc, drawn or fixed
            if config.get(key) is not None:
                return float(config[key])
        return float("nan")

    # -- scoring and diagnostics ---------------------------------------------

    @property
    def metrics(self) -> Metrics:
        """Fit and predictive scores for this model.

        The Bayesian block (ELPD and friends) is computed the first time it is
        asked for and then cached on the fit, because PSIS-LOO needs the posterior
        in memory and is not free.

        With ``settings.RELOO`` on, ``elpd_loo`` is then repaired by refitting the
        measurements whose Pareto-k says PSIS could not be trusted at them - which
        costs a NUTS fit apiece, and is why the flag is off by default. See
        :mod:`limnotech_rating_curves.evaluate.exact_loo`.

        Returns
        -------
        Metrics
        """
        self._require_fit()
        if not self._fit.bayes:
            self._fit.bayes = self.entry.bayes_metrics(self._fit)
            if settings.RELOO:
                from .evaluate.exact_loo import refine
                self._fit.bayes = refine(self._fit, self.sample, self.entry,
                                         fit_arguments=self.fit_arguments)
        return Metrics.from_fit(self._fit)

    def cross_validate(self, **kwargs):
        """Refit this model on subsets of its own sample and score the held-out points.

        The same call as :meth:`RatingSet.cross_validate`, restricted to this one
        model, and run through the same
        :func:`limnotech_rating_curves.cross_validate` - so a single model's sweep
        and a comparison's sweep are the same computation on the same folds.

        The ``zero_flow`` and ``nuts_sampler`` this model was fitted with carry over
        unless you pass them again, so the folds refit the model you have rather than
        a default one.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`limnotech_rating_curves.cross_validate` - ``scheme``,
            ``holdout``, ``n_splits``, ``n_train``, ``seed``, ``fold_method``,
            ``zero_flow``, ``nuts_sampler``.

        Returns
        -------
        CrossValidation
        """
        self._require_fit()
        return _cross_validate(self.sample, [self.name], self.fit_arguments, kwargs)

    def equation(self, precision: int = 6) -> str:
        """The fitted curve written out, for the families that have a closed form.

        Parameters
        ----------
        precision : int, default 6
            Significant figures per coefficient.

        Returns
        -------
        str
            The equation, or an empty string for a family whose curve cannot be
            written as one - the spline, whose shape is a basis matrix and its
            weights, and bdrc, whose exponent varies continuously with stage. Those
            two are read through :meth:`curve` and :meth:`predict` instead.
        """
        return ""

    def summary(self, var_names=None) -> pd.DataFrame:
        """Posterior summary of the fitted parameters.

        Parameters
        ----------
        var_names : list of str, optional
            Restrict to these parameters.

        Returns
        -------
        pandas.DataFrame
            Mean, standard deviation, credible interval, R-hat and effective
            sample size per parameter.
        """
        self._require_fit()
        from .evaluate import diagnostics
        return diagnostics.posterior_summary(self._fit, var_names=var_names)

    def diagnostics(self) -> pd.DataFrame:
        """Convergence diagnostics: R-hat and effective sample size per parameter.

        Returns
        -------
        pandas.DataFrame
            One row per parameter with ``r_hat``, ``ess_bulk``, ``ess_tail`` and a
            ``converged`` flag. See
            :mod:`limnotech_rating_curves.evaluate.diagnostics` for what the thresholds
            mean.
        """
        self._require_fit()
        from .evaluate import diagnostics
        return diagnostics.convergence(self._fit)

    def pareto_k(self) -> pd.DataFrame:
        """Per-measurement Pareto-k, the closest thing to an influence score.

        Returns
        -------
        pandas.DataFrame
            One row per measurement with its ``stage_ft``, ``discharge_cfs``,
            ``pareto_k`` and a ``reliable`` flag. Both are missing for a fit made
            with ``method="advi"``: PSIS-LOO needs several MCMC chains, and a
            variational fit has none, so ``reliable`` is ``pandas.NA`` rather than
            False - "not measured" is not the same as "unreliable".
        """
        self._require_fit()
        _ = self.metrics                      # ensure the Bayesian scores exist
        from .evaluate import metrics as metrics_module
        values = metrics_module.pareto_k_per_point(self._fit)
        reliable = pd.array(values <= settings.PARETO_K_GOOD, dtype="boolean")
        reliable[~np.isfinite(values)] = pd.NA
        return pd.DataFrame({
            "stage_ft": self.sample.stage_ft, "discharge_cfs": self.sample.discharge_cfs,
            "pareto_k": values, "reliable": reliable})

    def save_posterior(self, directory, sample_id=None):
        """Export the posterior to an ArviZ NetCDF file.

        Parameters
        ----------
        directory : path-like
            Destination directory.
        sample_id : str, optional
            Name used in the file name. Defaults to the sample's site id.

        Returns
        -------
        str or None
            The path written, or None if there was no posterior to write.
        """
        self._require_fit()
        name = sample_id or (self.sample.site_id if self.sample else "sample")
        return self.entry.save_posterior(self._fit, directory, name or "sample")

    # -- plotting -------------------------------------------------------------

    def plot(self, ax=None, *, level: float = 0.95, log_discharge: bool = True):
        """Plot the measurements, the fitted curve and its credible band.

        The curve is drawn over the whole padded grid it was fitted on, and the
        part outside the measured stage range is drawn in
        :data:`~limnotech_rating_curves.view.plots.EXTRAPOLATION_COLOR` rather than
        being cut off, so every family covers the same stages and the unsupported
        part of the curve is visible as such.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Created if not given.
        level : float, default 0.95
            Width of the band drawn.
        log_discharge : bool, default True
            Log-scale the discharge axis. A rating spans orders of magnitude, so
            this is normally what you want.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._require_fit()
        import matplotlib.pyplot as plt
        from .view import plots
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 5.5))
        measured_range = (self.sample.stage_range if self.sample is not None
                          else (-np.inf, np.inf))
        plots.draw_curve(ax, self.curve(level=level), measured_range,
                         color=self.color, label=self.label, level=level,
                         linewidth=2.4,
                         extrapolation_label=plots.EXTRAPOLATION_LABEL)
        if self.sample is not None:
            ax.scatter(self.sample.stage_ft, self.sample.discharge_cfs, s=45,
                       facecolor="white", edgecolor="black", zorder=5,
                       label="measurements")
        if log_discharge:
            ax.set_yscale("log")
        if self.sample is not None:
            plots.limit_discharge_axis(ax, self.sample.discharge_cfs, log_discharge)
        ax.set_xlabel(f"stage - {self.sample.stage_label if self.sample else 'ft'}")
        ax.set_ylabel("discharge (cfs)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize="small")
        return ax

    def plot_residuals(self, ax=None):
        """Predicted over observed discharge at every measurement, against stage.

        The same plot :meth:`RatingSet.plot_residuals` draws, with this model's series
        alone on it.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._require_fit()
        from .view import plots
        return plots.plot_residual_ratio(self, ax)

    def plot_check(self, *, zero_flow=None, level: float = 0.95, figsize=(13, 4.5)):
        """The two panels a rating is judged on: log-log shape, and residual ratio.

        Parameters
        ----------
        zero_flow : float, optional
            Stage of zero flow to subtract in the left panel. Defaults to this fit's
            own :attr:`zero_flow`.
        level : float, default 0.95
            Width of the credible band drawn.
        figsize : tuple, default (13, 4.5)
            Figure size.

        Returns
        -------
        numpy.ndarray of matplotlib.axes.Axes
        """
        self._require_fit()
        from .view import plots
        return plots.plot_fit_check(self, zero_flow=zero_flow, level=level,
                                    figsize=figsize)

    def __repr__(self):
        return f"{type(self).__name__}({self._config_repr()}, fitted={self.fitted})"

    def _config_repr(self) -> str:
        return ""


class PowerLaw(RatingModel):
    """Power-law rating :math:`Q = C(h - e)^{\\beta}`.

    The default model, and the one to reach for when a rating has to extrapolate:
    the power law *is* the theoretical form of flow over a control, so its behavior
    above the highest measurement is a physical extension rather than a guess.

    With ``segments > 1`` it becomes a segmented power law - a separate
    :math:`(C, \\beta)` above and below a fitted breakpoint. That describes a
    channel whose control changes with stage (in-bank flow giving way to overbank),
    and costs roughly three more measurements per segment to identify.

    Parameters
    ----------
    segments : int, default 1
        Number of power-law segments. Needs about ``3 * segments`` measurements.

    Examples
    --------
    >>> PowerLaw(segments=2).fit(measurements).predict(5.0)   # doctest: +SKIP
    """

    def __init__(self, segments: int = 1):
        super().__init__()
        self.segments = int(segments)
        key = "power_law" if self.segments == 1 else f"power_law_{self.segments}seg"
        try:
            self.entry = catalog.get(key)
        except KeyError:
            # a segment count the catalog does not list is still fittable
            self.entry = catalog.RatingCurveEntry(
                key, f"segmented power law ({self.segments} segments)",
                f"PL {self.segments}-seg", catalog.color("power_law"),
                "power_law", self.segments, max(3, 3 * self.segments))

    @property
    def coefficients(self) -> dict:
        """The closed-form coefficients, in discharge units.

        ``coefficient`` (:math:`C`), ``exponents`` (:math:`\\beta`, one per segment)
        and ``breakpoints`` (feet; the first is :attr:`zero_flow`), taken at the
        posterior mean. See
        :meth:`~limnotech_rating_curves.models.ratingcurve.BayesianRating.coefficients`
        for the exact form they combine in, and for why the curve through them
        tracks the median rather than the mean of :meth:`predict`.
        """
        self._require_fit()
        return self._fit.rating.coefficients()

    def equation(self, precision: int = 6) -> str:
        """The fitted power law, written out.

        Examples
        --------
        >>> PowerLaw().fit(measurements).equation()      # doctest: +SKIP
        'Q = 182.16 (h - 5.12323)^1.46971'
        """
        self._require_fit()
        return self._fit.rating.equation(precision=precision)

    def _config_repr(self):
        return f"segments={self.segments}"


class Spline(RatingModel):
    """Natural cubic spline rating, fitted in log space.

    Flexible and atheoretical. It will fit the measurements at least as well as any
    power law, which is the reason to be careful with it: the flexibility that
    helps between measurements is unconstrained beyond the highest one, so a spline
    should be read as an interpolator and not trusted to extrapolate. Compare its
    predictive score (``elpd_loo``) against a power law's before believing the
    better in-sample fit.

    Parameters
    ----------
    knots : int, optional
        Number of interior knots. Capped at ``n - 2``; ``None`` chooses
        automatically from the sample size.

    Examples
    --------
    >>> Spline().fit(measurements).interval(5.0)   # doctest: +SKIP
    """

    def __init__(self, knots: int | None = None):
        super().__init__()
        self.knots = knots
        self.entry = catalog.get("spline")
        self._config_override = {"df": int(knots)} if knots else None

    def _config_repr(self):
        return f"knots={self.knots if self.knots else 'auto'}"


class Bdrc(RatingModel):
    """Bayesian discharge rating curve - the bdrc generalized power law.

    A native PyMC port of the R ``bdrc`` package (Hrafnkelsson and others), so no R
    is required. Its distinguishing idea is that the power-law exponent varies
    smoothly with stage, modelled as a Gaussian process on :math:`\\beta(h)`. That
    makes it more flexible than a fixed-exponent power law while remaining a power
    law at every stage, and its priors are tuned to behave on very short records -
    which is why it is the model to try when a site has a handful of measurements.

    Parameters
    ----------
    variant : {'gplm0', 'gplm', 'plm0', 'plm'}, default 'gplm0'
        Which bdrc model:

        * ``gplm0`` - generalized power law, constant error variance (the default)
        * ``gplm`` - generalized power law, error variance varying with stage
        * ``plm0`` - plain power law, constant variance
        * ``plm`` - plain power law, variance varying with stage

    Examples
    --------
    >>> Bdrc("gplm0").fit(measurements).predict(5.0)   # doctest: +SKIP
    """

    VARIANTS = ("gplm0", "gplm", "plm0", "plm")

    def __init__(self, variant: str = "gplm0"):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {variant!r}")
        self.variant = variant
        self.entry = catalog.get(f"bdrc_{variant}")

    def _config_repr(self):
        return f"variant={self.variant!r}"


class Quadratic(RatingModel):
    """Least-squares quadratic in stage - the curve a field spreadsheet draws.

    ``Q = a h^2 + b h + c``, fitted by ordinary least squares on raw stage. This is
    what an Excel chart's order-2 polynomial trendline computes, and it is here so
    that curve can be reproduced and compared rather than retyped.

    It is not Bayesian and does not pretend to be: no priors, no posterior, no ELPD,
    and its band is a least-squares **prediction** interval. Two things follow.
    Compare it on NSE or R-squared, not on predictive score. And respect
    :attr:`effective_range`: a parabola turns around, so outside a limited band the
    fitted curve either falls as stage rises or goes negative, and
    :meth:`predict` returns NaN there instead of a confident wrong answer.

    Parameters
    ----------
    degree : int, default 2
        Polynomial degree. 2 is the spreadsheet case; 1 gives a straight line.
    stage_range : tuple of float, optional
        Override the effective range, when the curve is known to be usable over a
        band the measurements alone do not imply.

    Examples
    --------
    >>> rating = Quadratic().fit(measurements)          # doctest: +SKIP
    >>> rating.coefficients, rating.r_squared           # doctest: +SKIP
    >>> rating.equation()                               # doctest: +SKIP
    """

    def __init__(self, degree: int = 2, stage_range=None):
        super().__init__()
        self.degree = int(degree)
        self.stage_range = stage_range
        key = {1: "linear", 2: "quadratic"}.get(self.degree, f"poly{self.degree}")
        try:
            self.entry = catalog.get(key)
        except KeyError:
            self.entry = catalog.PolynomialEntry(
                key, f"least-squares degree-{self.degree} polynomial",
                f"poly {self.degree}", catalog.color("quadratic"), self.degree)
        if stage_range is not None:
            self._config_override = {"stage_range": tuple(stage_range)}

    @property
    def coefficients(self):
        """The fitted coefficients, highest power first, as Excel prints them."""
        self._require_fit()
        return self._fit.rating.coefficients

    @property
    def r_squared(self) -> float:
        """R-squared on discharge - the number beside the trendline equation."""
        self._require_fit()
        return self._fit.rating.r_squared

    @property
    def effective_range(self) -> tuple:
        """``(low, high)`` stage over which this curve may be used."""
        self._require_fit()
        return self._fit.rating.effective_range

    @property
    def turning_point(self):
        """Stage where the fitted parabola stops rising, or None for a line."""
        self._require_fit()
        return self._fit.rating.turning_point

    def equation(self, precision: int = 6) -> str:
        """The fitted polynomial, written out."""
        self._require_fit()
        return self._fit.rating.equation(precision=precision)

    def _config_repr(self):
        return f"degree={self.degree}"


class Exponential(RatingModel):
    """Least-squares exponential rating - Excel's ``trendlineType="exp"``, in code.

    ``Q = A exp(B h)``, fitted as ordinary least squares of ``log Q`` on stage, which
    is the computation Excel performs. Three MAGL workbooks (SBR-06, SBR-07, SBR-08)
    carry this form rather than a polynomial, and it is here so those sensors can be
    compared against the form their own field curve uses.

    Not Bayesian: no priors, no posterior, no ELPD. Its band is a least-squares
    prediction interval computed on log discharge and exponentiated, so it is
    asymmetric in cfs. And read :attr:`r_squared` (on discharge) rather than the
    log-space R² Excel prints, which flatters the fit.

    Examples
    --------
    >>> rating = Exponential().fit(measurements)         # doctest: +SKIP
    >>> rating.equation(), rating.rate                   # doctest: +SKIP
    """

    def __init__(self):
        super().__init__()
        self.entry = catalog.get("exponential")

    @property
    def amplitude(self) -> float:
        """``A`` - the fitted discharge at stage zero, which is an extrapolation."""
        self._require_fit()
        return self._fit.rating.amplitude

    @property
    def rate(self) -> float:
        """``B`` - log-discharge gradient per foot of stage."""
        self._require_fit()
        return self._fit.rating.rate

    @property
    def r_squared(self) -> float:
        """R-squared on discharge - not the log-space one Excel prints."""
        self._require_fit()
        return self._fit.rating.r_squared

    @property
    def r_squared_log(self) -> float:
        """R-squared on log discharge, the number Excel shows beside the trendline."""
        self._require_fit()
        return self._fit.rating.r_squared_log

    @property
    def effective_range(self) -> tuple:
        """``(low, high)`` stage over which this curve may be used."""
        self._require_fit()
        return self._fit.rating.effective_range

    def equation(self, precision: int = 6) -> str:
        """The fitted curve, written out."""
        self._require_fit()
        return self._fit.rating.equation(precision=precision)


def rating_model(model) -> RatingModel:
    """Build an estimator from a model name.

    Parameters
    ----------
    model : str or RatingModel
        A model key from the catalog (``"power_law"``, ``"power_law_2seg"``,
        ``"spline"``, ``"bdrc_gplm0"``, ``"quadratic"``, ...). An estimator instance
        is returned unchanged.

    Returns
    -------
    RatingModel
        An unfitted estimator.

    Raises
    ------
    KeyError
        On an unknown model name.
    """
    if isinstance(model, RatingModel):
        return model
    entry = catalog.get(model)
    if entry.family == "bdrc":
        return Bdrc(entry.variant)
    if entry.family == "exponential":
        return Exponential()
    if entry.family == "polynomial":
        return Quadratic(entry.degree)
    if entry.algorithm == "spline":
        return Spline()
    return PowerLaw(entry.segments)


def resolve_models(models=None) -> list:
    """Turn a ``models=`` argument into a list of unfitted ratings.

    Parameters
    ----------
    models : None or str or sequence, optional
        Model keys, group names (``"all"``, ``"bdrc"``, ``"ratingcurve"``) or
        estimator instances. ``None`` gives the curated default set.

    Returns
    -------
    list of RatingModel
    """
    if isinstance(models, RatingModel):
        return [models]
    if models is not None and not isinstance(models, str):
        instances = [m for m in models if isinstance(m, RatingModel)]
        if instances and len(instances) == len(list(models)):
            return instances
    return [rating_model(entry.key) for entry in catalog.select(models)]


def _as_sample(data, discharge=None, *, stage=None, time=None,
               stage_datum=None) -> Sample:
    """Coerce any accepted input to a :class:`Sample`.

    Accepts what :meth:`Sample.of` accepts, and additionally a pagaia ``Station``,
    which is routed through
    :func:`limnotech_rating_curves.data.pagaia.station_sample` - a station supplies
    its own stage record, so `stage` does not apply to it.
    """
    if isinstance(data, Sample):
        return data
    if _looks_like_pagaia_station(data):
        from .data import pagaia
        if stage is not None:
            raise TypeError("stage= names a column of a DataFrame; a pagaia station "
                            "supplies its own stage record. Use variable= instead "
                            "(see pagaia.station_sample).")
        return pagaia.station_sample(data, discharge=discharge,
                                            stage_datum=stage_datum)
    return Sample.of(data, discharge, stage=stage, time=time,
                     stage_datum=stage_datum)


def _looks_like_pagaia_station(obj) -> bool:
    """True for a pagaia ``Station`` without importing pagaia to find out."""
    return (hasattr(obj, "station_id") and hasattr(obj, "get_timeseries_data")
            and hasattr(obj, "latlon"))


def fit_rating(data, discharge=None, *, model="power_law", **kwargs) -> RatingModel:
    """Fit one rating model to stage-discharge measurements.

    The one-line entry point. For several models at once, use :func:`compare`.

    Parameters
    ----------
    data : Sample or pandas.DataFrame or array-like or pagaia Station
        The measurements.
    discharge : array-like or str, optional
        Discharge (cfs) when `data` is an array of stage values, or the discharge
        column name when `data` is a DataFrame.
    model : str or RatingModel, default 'power_law'
        Which model to fit.
    **kwargs
        Forwarded to :meth:`RatingModel.fit` - ``stage``, ``time``,
        ``stage_datum``, ``method``, ``seed``, ``zero_flow``, ``cores``,
        ``advi_iters``, ``nuts_sampler``.

    Returns
    -------
    RatingModel
        The fitted estimator.

    Examples
    --------
    >>> rating = fit_rating(measurements)                          # doctest: +SKIP
    >>> rating = fit_rating(measurements, model="bdrc_gplm0")      # doctest: +SKIP
    >>> rating = fit_rating(elevations, flows, stage_datum=612.34)  # doctest: +SKIP
    >>> rating = fit_rating(frame, stage="WSE (ft)", discharge="Q")  # doctest: +SKIP
    """
    return rating_model(model).fit(data, discharge, **kwargs)


def _cross_validate(sample, model_keys, fit_arguments, kwargs):
    """The one cross-validation call both :class:`RatingModel` and :class:`RatingSet`
    make, so ``.cross_validate()`` means the same thing on either.

    The fit arguments that change what is being fitted - the stage of zero flow and
    the NUTS implementation - are carried into the folds, so a sweep refits the model
    that was fitted rather than a default one. Anything passed explicitly wins.
    """
    from .evaluate import crossval
    kwargs.setdefault("models", list(model_keys))
    for name in ("zero_flow", "nuts_sampler"):
        if fit_arguments.get(name) is not None:
            kwargs.setdefault(name, fit_arguments[name])
    return crossval.cross_validate(sample, **kwargs)


def _stack_per_model(models, per_model) -> pd.DataFrame:
    """One per-parameter table per model, under leading ``model`` and ``parameter``
    columns. The parameter names are in the index of what the per-model methods
    return, so they are moved into a column before the blocks are concatenated.
    """
    blocks = []
    for model in models:
        block = per_model(model)
        if block is None or block.empty:
            continue
        block = block.rename_axis("parameter").reset_index()
        block.insert(0, "model", model.name)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


@dataclass
class RatingSet:
    """Several models fitted to one sample, for comparison.

    Index it by name (``comparison["spline"]``), take :attr:`best`, read
    :attr:`metrics`, or :meth:`plot` them all on one axis.

    It answers to the same method names as a single
    :class:`RatingModel`, with one row or one series per model, so moving between
    one model and several needs no second vocabulary:

    ==========================  ================================================
    ``predict(stage)``          posterior-mean discharge, per model
    ``interval(stage, level)``  credible interval, per model
    ``curve(stage, level)``     every fitted rating as one long table
    ``metrics``                 fit and predictive scores, one row per model
    ``cross_validate()``        refit on subsets and score the held-out points
    ``summary()``               posterior summaries, one block per model
    ``equation()``              each fitted curve written out, one row per model
    ``diagnostics()``           convergence diagnostics, one block per model
    ``plot(ax)``                every curve on one axis
    ``plot_residuals(ax)``      predicted / observed against stage, per model
    ``save_posterior(dir)``     write every posterior to NetCDF
    ==========================  ================================================

    On top of those, :meth:`ranking` and :meth:`plot_ranking` say which model to
    keep, which is a question a single model cannot be asked.

    Attributes
    ----------
    sample : Sample
        The measurements every model was fitted to.
    models : list of RatingModel
        The models that fitted successfully.
    failures : dict
        ``{model key: reason}`` for models that did not fit, so nothing disappears
        silently.
    reference : dict or None
        An externally published rating to compare against, as
        ``{"label", "curve", "metrics"}``.
    """

    sample: Sample
    models: list = field(default_factory=list)
    failures: dict = field(default_factory=dict)
    reference: dict | None = None

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, key) -> RatingModel:
        """Get one model by key, label or position."""
        if isinstance(key, int):
            return self.models[key]
        for model in self.models:
            if key in (model.name, model.label):
                return model
        raise KeyError(f"no model {key!r} here; have "
                       f"{[model.name for model in self.models]}")

    @property
    def best(self) -> "RatingModel | None":
        """The model with the highest predictive score.

        Ranked on ``elpd_loo``, with ``nse`` breaking ties (and standing in when no
        ELPD is available). Note that "best" here means best *out-of-sample*
        predictive density on this sample - read :meth:`table` and the standard
        error before treating a small margin as a decision.

        Returns
        -------
        RatingModel or None
            None if nothing fitted.
        """
        if not self.models:
            return None

        def score(model):
            """Rank key: predictive score first, in-sample NSE as the tiebreak."""
            scores = model.metrics
            elpd = scores.elpd_loo if np.isfinite(scores.elpd_loo) else -np.inf
            nse = scores.nse if np.isfinite(scores.nse) else -np.inf
            return (elpd, nse)

        return max(self.models, key=score)

    # -- prediction -----------------------------------------------------------

    def predict(self, stage) -> pd.DataFrame:
        """Posterior-mean discharge at the given stage, from every model.

        :meth:`RatingModel.predict` for each model in turn, side by side.

        Parameters
        ----------
        stage : float or array-like
            Gage height (feet), on the axis the models were fitted on.

        Returns
        -------
        pandas.DataFrame
            Indexed by stage, one column per model key. Discharge in cfs; NaN where
            a model's fitted curve does not reach.
        """
        stages = np.atleast_1d(np.asarray(stage, float))
        return pd.DataFrame({model.name: model.predict(stages)
                             for model in self.models},
                            index=pd.Index(stages, name="stage_ft"))

    def interval(self, stage, level: float = 0.95) -> pd.DataFrame:
        """The credible interval for discharge at the given stage, from every model.

        Parameters
        ----------
        stage : float or array-like
            Gage height (feet).
        level : float, default 0.95
            Interval width. See :meth:`RatingModel.interval` for how exactly each
            family honors it.

        Returns
        -------
        pandas.DataFrame
            Columns ``model``, ``stage_ft``, ``lower``, ``upper`` - one block of rows
            per model. Discharge in cfs.
        """
        stages = np.atleast_1d(np.asarray(stage, float))
        blocks = []
        for model in self.models:
            lower, upper = model.interval(stages, level=level)
            blocks.append(pd.DataFrame({"model": model.name, "stage_ft": stages,
                                        "lower": lower, "upper": upper}))
        return (pd.concat(blocks, ignore_index=True) if blocks else
                pd.DataFrame(columns=["model", "stage_ft", "lower", "upper"]))

    def curve(self, stage=None, level: float = 0.95) -> pd.DataFrame:
        """Every fitted rating as one long table.

        Parameters
        ----------
        stage : array-like, optional
            Stages to evaluate at. Defaults to each fit's own padded grid, which is
            the same grid for every family.
        level : float, default 0.95
            Width of the reported interval.

        Returns
        -------
        pandas.DataFrame
            ``model`` and ``label``, then the columns
            :meth:`RatingModel.curve` returns.
        """
        blocks = []
        for model in self.models:
            block = model.curve(stage=stage, level=level)
            block.insert(0, "model", model.name)
            block.insert(1, "label", model.label)
            blocks.append(block)
        return (pd.concat(blocks, ignore_index=True) if blocks else
                pd.DataFrame(columns=["model", "label", "stage_ft", "discharge_cfs",
                                      "discharge_median_cfs", "lower", "upper"]))

    # -- scoring and diagnostics ---------------------------------------------

    @property
    def metrics(self) -> pd.DataFrame:
        """Fit and predictive scores, one row per model, best predictive score first.

        The table form of :attr:`RatingModel.metrics`.

        Returns
        -------
        pandas.DataFrame
            Columns ``model``, ``label``, ``n``, ``nse``, ``rmse``, ``r2_log``,
            ``elpd_loo``, ``se_loo``, ``p_loo``, ``pareto_k_max``. Any published
            reference curve is appended at the end (with no Bayesian scores, since
            it has no posterior). See
            :data:`limnotech_rating_curves.evaluate.metrics.METRIC_GLOSSARY` for what each
            column means.
        """
        rows = []
        for model in self.models:
            scores = model.metrics
            rows.append({"model": model.name, "label": model.label, "n": scores.n,
                         "nse": scores.nse, "rmse": scores.rmse,
                         "r2_log": scores.r2_log, "elpd_loo": scores.elpd_loo,
                         "se_loo": scores.se_loo, "p_loo": scores.p_loo,
                         "pareto_k_max": scores.pareto_k_max})
        table = pd.DataFrame(rows)
        if not table.empty:
            table = (table.sort_values("elpd_loo", ascending=False, na_position="last")
                     .reset_index(drop=True))
        if self.reference:
            scores = self.reference["metrics"]
            table = pd.concat([table, pd.DataFrame([{
                "model": "published_reference", "label": self.reference["label"],
                "n": scores.get("n"), "nse": scores.get("nse"),
                "rmse": scores.get("rmse"), "r2_log": scores.get("r2_log")}])],
                ignore_index=True)
        return table

    def ranking(self) -> pd.DataFrame:
        """Rank the models by predictive score, with differences from the best.

        This is the table to read when deciding between models. ``elpd_diff`` is
        each model's ELPD minus the best model's, so it is 0 for the winner and
        negative for the rest; ``dse`` is the standard error of that difference. A
        model within about two ``dse`` of the top is not distinguishable from it by
        this sample, and the more parsimonious of the two is the better choice.

        Returns
        -------
        pandas.DataFrame
            Columns ``model``, ``elpd_loo``, ``se_loo``, ``elpd_diff``, ``dse``,
            ``p_loo``, ``pareto_k_max``, ``warning`` - the last flagging fits whose
            Pareto-k makes the estimate unreliable.
        """
        rows = []
        for model in self.models:
            scores = model.metrics
            rows.append({"model": model.name, "label": model.label,
                         "elpd_loo": scores.elpd_loo, "se_loo": scores.se_loo,
                         "p_loo": scores.p_loo,
                         "pareto_k_max": scores.pareto_k_max})
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        table = table.sort_values("elpd_loo", ascending=False,
                                  na_position="last").reset_index(drop=True)
        best = table["elpd_loo"].iloc[0]
        table["elpd_diff"] = table["elpd_loo"] - best
        # A conservative standard error on the difference: the two models' errors
        # added in quadrature. The exact quantity uses the paired pointwise
        # differences, which needs both log-likelihood matrices aligned; this is the
        # upper bound on it and is what a ranking decision should be held to.
        best_se = table["se_loo"].iloc[0]
        table["dse"] = np.sqrt(table["se_loo"] ** 2 + best_se ** 2)
        table.loc[0, "dse"] = 0.0
        table["warning"] = np.where(
            table["pareto_k_max"] > settings.PARETO_K_GOOD,
            f"pareto_k > {settings.PARETO_K_GOOD}: LOO unreliable", "")
        return table[["model", "label", "elpd_loo", "se_loo", "elpd_diff", "dse",
                      "p_loo", "pareto_k_max", "warning"]]

    def plot(self, ax=None, *, level: float | None = None, log_discharge: bool = True):
        """Overlay every fitted model on the measurements.

        Every family is drawn over the same padded stage grid, and the part of each
        curve outside the measured range is dotted. Model color is kept there
        rather than a shared extrapolation color, because on an overlay that is the
        only thing saying which curve is whose.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.
        level : float, optional
            Draw each model's credible band at this level. ``None`` draws curves
            only, which is what makes a multi-model comparison readable.
        log_discharge : bool, default True
            Log-scale the discharge axis.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from .view import plots
        if ax is None:
            _, ax = plt.subplots(figsize=(8.5, 6))
        ax.scatter(self.sample.stage_ft, self.sample.discharge_cfs, s=45,
                   facecolor="white", edgecolor="black", zorder=5,
                   label="measurements")
        measured_range = self.sample.stage_range
        extrapolated = False
        for model in self.models:
            curve = model.curve(level=level or 0.95)
            plots.draw_curve(ax, curve, measured_range, color=model.color,
                             label=model.label, level=level, band_alpha=0.12,
                             extrapolation_color=None)
            extrapolated = extrapolated or bool(
                (curve["stage_ft"] < measured_range[0]).any()
                or (curve["stage_ft"] > measured_range[1]).any())
        if extrapolated:
            ax.plot([], [], color="0.4", ls=":", lw=2.2,
                    label=plots.EXTRAPOLATION_LABEL)
        if self.reference:
            reference_curve = self.reference["curve"].sort_values("stage_ft")
            ax.plot(reference_curve["stage_ft"], reference_curve["discharge_cfs"],
                    color="black", ls="--", lw=2.0, label=self.reference["label"])
        if log_discharge:
            ax.set_yscale("log")
        plots.limit_discharge_axis(ax, self.sample.discharge_cfs, log_discharge)
        ax.set_xlabel(f"stage - {self.sample.stage_label}")
        ax.set_ylabel("discharge (cfs)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize="small")
        return ax

    def plot_residuals(self, ax=None):
        """Predicted over observed discharge per model, against stage.

        Where the families actually differ. Overlaid curves sit on top of each other
        over the measured range; the disagreement is in the residuals and at the ends.
        Any reference curve is drawn alongside as crosses.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .view import plots
        return plots.plot_residual_ratio(self, ax)

    def plot_ranking(self, ax=None):
        """The predictive-score ranking as a plot, with error bars on each difference.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.

        Returns
        -------
        matplotlib.axes.Axes or None
            None when no model has a predictive score to rank.
        """
        from .view import plots
        return plots.plot_ranking(self, ax)

    def summary(self, var_names=None) -> pd.DataFrame:
        """Posterior summaries of every model's fitted parameters, stacked.

        Parameters
        ----------
        var_names : list of str, optional
            Restrict to these parameters.

        Returns
        -------
        pandas.DataFrame
            What :meth:`RatingModel.summary` returns, with a leading ``model``
            column. Models whose backend has no posterior summary are skipped.
        """
        return _stack_per_model(self.models, lambda model: model.summary(var_names))

    def diagnostics(self) -> pd.DataFrame:
        """Convergence diagnostics for every model, stacked.

        Returns
        -------
        pandas.DataFrame
            What :meth:`RatingModel.diagnostics` returns, with a leading ``model``
            column.
        """
        return _stack_per_model(self.models, lambda model: model.diagnostics())

    def equation(self, precision: int = 6) -> pd.DataFrame:
        """Every fitted curve written out, one row per model.

        Parameters
        ----------
        precision : int, default 6
            Significant figures per coefficient.

        Returns
        -------
        pandas.DataFrame
            Columns ``model``, ``label``, ``equation``. The families with no closed
            form are listed with an empty ``equation`` rather than dropped, so the
            table says which models are readable this way and which are not.
        """
        return pd.DataFrame(
            [{"model": model.name, "label": model.label,
              "equation": model.equation(precision=precision)}
             for model in self.models],
            columns=["model", "label", "equation"])

    def cross_validate(self, **kwargs):
        """Refit exactly these models on subsets of this sample and score the rest.

        The same call as :meth:`RatingModel.cross_validate`, over every model here.
        Every model is scored on the *same* folds, because
        :func:`limnotech_rating_curves.cross_validate` computes the partition from the
        measurements and the seed alone.

        The ``zero_flow`` and ``nuts_sampler`` these models were fitted with carry
        over unless you pass them again.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`limnotech_rating_curves.cross_validate` - ``scheme``,
            ``holdout``, ``n_splits``, ``n_train``, ``seed``, ``fold_method``,
            ``zero_flow``, ``nuts_sampler``.

        Returns
        -------
        CrossValidation
        """
        fit_arguments = self.models[0].fit_arguments if self.models else {}
        return _cross_validate(self.sample, [model.name for model in self.models],
                               fit_arguments, kwargs)

    def save_posterior(self, directory=None) -> list:
        """Export every fitted posterior to NetCDF.

        Parameters
        ----------
        directory : path-like, optional
            Destination. Defaults to ``settings.POSTERIOR_DIR``.

        Returns
        -------
        list of str
            The paths written.
        """
        directory = Path(settings.POSTERIOR_DIR if directory is None else directory)
        written = [model.save_posterior(directory, self.sample.site_id or "sample")
                   for model in self.models]
        return [path for path in written if path]

    def __repr__(self):
        return (f"RatingSet({len(self.models)} models on "
                f"{self.sample.site_id or 'a sample'}, n={len(self.sample)})")


def compare(data, discharge=None, *, models=None, stage=None, time=None,
            stage_datum=None, method: str = "nuts", seed: int = settings.SEED,
            zero_flow=None, cores=None, advi_iters: int = settings.ADVI_ITERS,
            nuts_sampler=None, reference=None) -> RatingSet:
    """Fit several rating models to the same measurements and compare them.

    Parameters
    ----------
    data : Sample or pandas.DataFrame or array-like or pagaia Station
        The measurements.
    discharge : array-like or str, optional
        Discharge (cfs) when `data` is an array, or the column name when it is a
        DataFrame.
    stage : str, optional
        Name of the stage column, when `data` is a DataFrame whose stage column is
        not one of the recognized names.
    time : array-like or str, optional
        Measurement timestamps, or the name of the timestamp column.
    models : None or str or sequence, optional
        Which models to fit - keys, group names (``"all"``, ``"bdrc"``,
        ``"ratingcurve"``) or estimator instances. ``None`` fits the curated
        default set.
    stage_datum : optional
        Convert stage onto a gage-height reference first; see
        :mod:`limnotech_rating_curves.data.datum`.
    method : {'nuts', 'advi'}, default 'nuts'
        Fitting method for every model.
    seed : int
        RNG seed.
    zero_flow : optional
        Stage-of-zero-flow handling; see :meth:`RatingModel.fit`.
    cores : int, optional
        Parallel NUTS chains.
    advi_iters : int
        ADVI optimization steps.
    nuts_sampler : str, optional
        Which NUTS implementation.
    reference : dict, optional
        A published rating to show alongside, as ``{"label", "curve", "metrics"}``.

    Returns
    -------
    RatingSet
        The models that fitted, plus the reasons for any that did not. A model
        failing does not stop the others.

    Examples
    --------
    >>> comparison = compare(measurements, models="all")   # doctest: +SKIP
    >>> comparison.metrics                                  # doctest: +SKIP
    >>> comparison.ranking()                                # doctest: +SKIP
    """
    sample = _as_sample(data, discharge, stage=stage, time=time,
                        stage_datum=stage_datum)
    if len(sample) == 0:
        raise ValueError(f"no usable measurements to fit "
                         f"({sample.skipped or 'the sample is empty'})")

    fitted, failures = [], {}
    for estimator in resolve_models(models):
        try:
            fitted.append(estimator.fit(
                sample, method=method, seed=seed, zero_flow=zero_flow, cores=cores,
                advi_iters=advi_iters, nuts_sampler=nuts_sampler))
        except Exception as exc:  # noqa: BLE001 - one model must not sink the rest
            failures[estimator.name] = str(exc)
            log.warning("%s did not fit: %s", estimator.name, exc)
    return RatingSet(sample=sample, models=fitted, failures=failures,
                     reference=reference)
