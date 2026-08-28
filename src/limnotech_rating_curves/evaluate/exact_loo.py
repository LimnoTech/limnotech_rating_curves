import logging

import numpy as np
import pandas as pd

from .. import settings
from ..core import Sample
from ..models import bdrc, ratingcurve
from ..models.bdrc.defaults import CFS_TO_CMS, FT_TO_M
from ..models.bdrc.posterior import normal_logpdf
from . import metrics as metrics_module

log = logging.getLogger(__name__)

#: Families with a posterior, and so with an ELPD that can be refined.
ELIGIBLE_FAMILIES = ("bdrc", "ratingcurve")

#: Fields the returned scores carry beyond ``metrics.BAYES_FIELDS``.
RELOO_FIELDS = ("elpd_loo_psis", "n_refits", "n_flagged", "n_over_budget",
                "refit_index", "refit_budget", "k_threshold")


def refit_budget(n_points: int, fraction: float = None, max_refits: int = None) -> int:
    """How many refits this sample is allowed.

    Parameters
    ----------
    n_points : int
        Measurements in the sample.
    fraction : float, optional
        Share of the sample that may be refitted. Defaults to
        ``settings.RELOO_MAX_FRACTION``.
    max_refits : int, optional
        A hard ceiling, applied after the fraction.

    Returns
    -------
    int
        At least one. A sample small enough that the fraction rounds to zero is
        exactly the case where a single influential measurement matters most, so
        the budget never rounds down to nothing.
    """
    fraction = settings.RELOO_MAX_FRACTION if fraction is None else float(fraction)
    budget = max(1, int(round(fraction * int(n_points))))
    if max_refits is not None:
        budget = min(budget, int(max_refits))
    return budget


#: Column carrying each measurement's position in the sample as the caller supplied
#: it, so a refit can be reported against that rather than against a fitting order
#: the caller never sees.
POSITION_COLUMN = "sample_position"


def ordered_frame(sample, family: str) -> pd.DataFrame:
    """The measurements in the order this family's log-likelihood is indexed by.

    Column *i* of a pointwise log-likelihood - and therefore Pareto-k *i* - belongs
    to whichever measurement the backend put in position *i*, which is not
    necessarily the caller's row *i*. ``bdrc.fit`` sorts its measurements by stage
    before doing anything else, so its columns are in ascending stage; the
    ratingcurve family flattens the arrays it is given and leaves the order alone.
    Getting this wrong attributes a Pareto-k, and then a refit, to the wrong
    measurement - silently, because every index is still in range.

    Parameters
    ----------
    sample : Sample or pandas.DataFrame
        The measurements.
    family : {'bdrc', 'ratingcurve'}
        Which backend fitted them.

    Returns
    -------
    pandas.DataFrame
        Stage and discharge in the backend's order, plus ``sample_position``
        giving each measurement's row in `sample`.
    """
    frame = sample.to_frame() if isinstance(sample, Sample) else pd.DataFrame(sample)
    frame = (frame.dropna(subset=["stage_ft", "discharge_cfs"])
             .reset_index(drop=True))
    frame[POSITION_COLUMN] = np.arange(len(frame))
    if family == "bdrc":
        frame = frame.sort_values("stage_ft", kind="stable").reset_index(drop=True)
    return frame


def _psis_pointwise(fit, rating=None):
    """The pointwise PSIS-LOO result and the offset onto raw log discharge.

    The two families reach ArviZ differently - bdrc hands over a log-likelihood
    matrix and no posterior, so the relative efficiency has to be supplied, while a
    ratingcurve fit has its posterior in memory and ArviZ works it out. Both are
    already done once in :meth:`ModelEntry.bayes_metrics`; this repeats them only
    because that method keeps the summary dict and discards the ``ELPDData`` object
    :func:`arviz.reloo` needs.

    Parameters
    ----------
    fit : FitResult
    rating : BayesianRating, optional
        The live ratingcurve model, for that family.

    Returns
    -------
    tuple
        ``(ELPDData, log_offset)``.
    """
    import arviz as az

    if fit.family == "bdrc":
        log_likelihood = np.asarray(fit.log_likelihood, float)
        idata = az.from_dict(
            log_likelihood={"obs": log_likelihood[np.newaxis, :, :]})
        reff = metrics_module.relative_efficiency(log_likelihood)
        return az.loo(idata, reff=reff, pointwise=True), 0.0

    metrics_module.ensure_log_likelihood(rating)
    log_std = float(np.log(rating.model.q_transform.std_))
    return az.loo(rating.idata, pointwise=True), -log_std


class _RefitWrapper:
    """What :func:`arviz.reloo` calls to refit without one measurement.

    Deliberately not a subclass of :class:`arviz.SamplingWrapper`: that base class
    exists to supply a ``log_likelihood__i`` built from a plain Python density
    function and a list of posterior variable names, which neither family here has
    - bdrc evaluates its density off an interpolated posterior curve, and
    ratingcurve evaluates it by pushing the held-out point through its own PyMC
    model. Every method the base class requires is implemented below, which is all
    :func:`arviz.reloo` checks for.

    Parameters
    ----------
    frame : pandas.DataFrame
        Every measurement, in the order the all-data fit saw them.
    seed : int
        Base seed; the fold holding out measurement *i* is fitted with ``seed + 1 + i``,
        so a repair is reproducible and no fold repeats the all-data seed.
    """

    def __init__(self, frame: pd.DataFrame, seed: int):
        self.frame = frame
        self.seed = seed
        self.held_out = None

    def sel_observations(self, idx):
        """Split off measurement `idx`, keeping the rest as the training set."""
        index = int(np.ravel(idx)[0])
        self.held_out = index
        keep = np.delete(np.arange(len(self.frame)), index)
        return (self.frame.iloc[keep].reset_index(drop=True),
                self.frame.iloc[[index]].reset_index(drop=True))

    def get_inference_data(self, fitted_model):
        """The fitted object itself.

        :func:`arviz.reloo` only passes this on to :meth:`log_likelihood__i`, and
        both families need more of their fit than its ``InferenceData`` carries -
        bdrc needs the posterior rating curve, ratingcurve needs its live PyMC
        model and that fold's discharge transform.
        """
        return fitted_model

    def check_implemented_methods(self, methods) -> list:
        """Nothing is missing; every method :func:`arviz.reloo` requires is here."""
        return []

    def _fold_seed(self) -> int:
        return int(self.seed) + 1 + int(self.held_out)


class BdrcRefitWrapper(_RefitWrapper):
    """Refits a bdrc rating without one measurement and scores that measurement.

    Parameters
    ----------
    frame : pandas.DataFrame
        Every measurement.
    variant : {'gplm0', 'gplm', 'plm0', 'plm'}
        Which bdrc model, from the all-data fit's config.
    grid : array-like
        Stage grid spanning the **whole** sample. It has to span the whole sample
        rather than each fold's own range, because the measurement being scored is
        by construction absent from the fold, and bdrc refuses to predict above the
        stages it was given.
    zero_flow_ft : float or None
        Known stage of zero flow, as the all-data fit used it.
    nuts_sampler : str or None
        Which NUTS implementation.
    seed : int
        Base seed.
    """

    def __init__(self, frame, *, variant, grid, zero_flow_ft, nuts_sampler, seed):
        super().__init__(frame, seed)
        self.variant = variant
        self.grid = np.asarray(grid, float)
        self.zero_flow_ft = zero_flow_ft
        self.nuts_sampler = nuts_sampler

    def sample(self, modified_observed_data):
        """Fit bdrc on the training measurements, keeping the whole fit object."""
        output = bdrc.fit_predict(
            modified_observed_data["stage_ft"].to_numpy(float),
            modified_observed_data["discharge_cfs"].to_numpy(float),
            self.grid, model=self.variant, method="nuts",
            c_param_ft=self.zero_flow_ft, with_loglik=True,
            seed=self._fold_seed(), nuts_sampler=self.nuts_sampler)
        return output["fit"]

    def log_likelihood__i(self, excluded_obs, idata__i):
        """Per-draw log density of the held-out measurement under the fold.

        bdrc's likelihood is Gaussian on log discharge about the posterior mean
        curve, so the density anywhere is read off that curve. The fold's curve is
        evaluated at the held-out stage by linear interpolation, which is how
        :meth:`BdrcFit.predict` evaluates it anywhere off its own rows; against
        bdrc's own ``log_lik_i`` at the fitted measurements this reproduces it
        exactly.
        """
        import xarray as xr

        stage_m = float(excluded_obs["stage_ft"].iloc[0]) * FT_TO_M
        discharge_cms = float(excluded_obs["discharge_cfs"].iloc[0]) * CFS_TO_CMS
        return xr.DataArray(holdout_log_likelihood(idata__i, stage_m, discharge_cms))


def holdout_log_likelihood(fitted, stage_m: float, discharge_cms: float) -> np.ndarray:
    """One log-likelihood per posterior draw for a measurement a bdrc fit never saw.

    Parameters
    ----------
    fitted : BdrcFit
        A fit made without this measurement.
    stage_m : float
        Its stage, in metres - bdrc's own units.
    discharge_cms : float
        Its discharge, in cubic metres per second.

    Returns
    -------
    numpy.ndarray
        Shape ``(draws,)``, a density on raw log discharge.
    """
    with np.errstate(divide="ignore"):
        log_mean = np.log(fitted.rating_curve_mean_posterior)
    # rows at or below the stage of zero flow carry zero discharge in every draw
    usable = np.isfinite(log_mean).all(axis=1)
    rows, log_mean = fitted.h[usable], log_mean[usable]
    log_mu = np.array([np.interp(stage_m, rows, log_mean[:, draw])
                       for draw in range(log_mean.shape[1])])

    sigma = np.asarray(fitted.sigma_eps_posterior, float)
    if sigma.ndim == 2:
        sigma = sigma[usable]
        sigma = np.array([np.interp(stage_m, rows, sigma[:, draw])
                          for draw in range(sigma.shape[1])])
    return normal_logpdf(np.log(discharge_cms), log_mu, sigma)


class RatingCurveRefitWrapper(_RefitWrapper):
    """Refits a ratingcurve rating without one measurement and scores that measurement.

    Parameters
    ----------
    frame : pandas.DataFrame
        Every measurement.
    key, label : str
        Identifiers passed through to :func:`models.ratingcurve.fit`.
    algorithm : {'power_law', 'spline'}
        Which family.
    segments : int or None
        Power-law segment count.
    config_override : dict or None
        The all-data fit's configuration, so a fold is the same model.
    zero_flow : optional
        The zero-flow specification the all-data fit was given. Passed on rather
        than the resolved number, so a fold re-derives its breakpoint prior from
        its own measurements - leaving a measurement out of the fit and then
        centering the prior on it would not be leave-one-out.
    nuts_sampler : str or None
        Which NUTS implementation.
    seed : int
        Base seed.
    all_data_log_std : float
        ``log(std_)`` of the all-data discharge transform, which is the space the
        PSIS values this repairs are expressed in.
    """

    def __init__(self, frame, *, key, label, algorithm, segments, config_override,
                 zero_flow, nuts_sampler, seed, all_data_log_std):
        super().__init__(frame, seed)
        self.key = key
        self.label = label
        self.algorithm = algorithm
        self.segments = segments
        self.config_override = config_override
        self.zero_flow = zero_flow
        self.nuts_sampler = nuts_sampler
        self.all_data_log_std = float(all_data_log_std)

    def sample(self, modified_observed_data):
        """Fit on the training measurements, returning the live ratingcurve model."""
        fit = ratingcurve.fit(
            modified_observed_data, key=self.key, label=self.label,
            algorithm=self.algorithm, segments=self.segments, method="nuts",
            seed=self._fold_seed(), zero_flow=self.zero_flow,
            nuts_sampler=self.nuts_sampler, enforce_min_points=False,
            config_override=self.config_override)
        if not fit.ok or fit.rating is None:
            raise RuntimeError(f"fold without measurement {self.held_out} "
                               f"could not be fitted: {fit.reason}")
        return fit.rating

    def log_likelihood__i(self, excluded_obs, idata__i):
        """Per-draw log density of the held-out measurement under the fold.

        The fold's PyMC model is repointed at the held-out measurement and its
        log-likelihood recomputed against the posterior already drawn, which is
        what makes this one extra evaluation rather than one extra fit.
        """
        import pymc as pm
        import xarray as xr

        rating = idata__i
        # The spline hands PyMC a basis matrix and needs it two-dimensional, which a
        # single row is not; scoring the measurement twice keeps the rank and the two
        # columns are identical, so the first is the answer. Same workaround as
        # BayesianRating.predict.
        stage = np.repeat(excluded_obs["stage_ft"].to_numpy(float), 2)
        discharge = np.repeat(excluded_obs["discharge_cfs"].to_numpy(float), 2)

        with ratingcurve.cleared_initial_values(rating.model.model):
            rating.model._data_setter(stage, discharge)
            computed = pm.compute_log_likelihood(
                rating.idata, model=rating.model.model,
                extend_inferencedata=False, progressbar=False)

        observed_var = metrics_module.RATINGCURVE_OBSERVED_VAR
        values = computed[observed_var].values[..., 0].ravel()
        fold_log_std = float(np.log(rating.model.q_transform.std_))
        return xr.DataArray(values + self.all_data_log_std - fold_log_std)


def _wrapper_for(fit, entry, frame, fit_arguments):
    """The wrapper matching this fit's family, configured as the all-data fit was."""
    fit_arguments = fit_arguments or {}
    seed = int(fit_arguments.get("seed", settings.SEED))
    nuts_sampler = fit.config.get("nuts_sampler")

    if fit.family == "bdrc":
        stage = frame["stage_ft"].to_numpy(float)
        low, high = float(np.nanmin(stage)), float(np.nanmax(stage))
        pad = settings.GRID_PAD_FRACTION * ((high - low) or 1.0)
        return BdrcRefitWrapper(
            frame, variant=fit.config.get("variant"),
            grid=np.linspace(low - pad, high + pad, settings.GRID_POINTS),
            zero_flow_ft=fit.config.get("zero_flow_ft"),
            nuts_sampler=nuts_sampler, seed=seed)

    config = {key: value for key, value in (fit.config or {}).items()
              if key in ("segments", "df")}
    return RatingCurveRefitWrapper(
        frame, key=fit.key, label=fit.label, algorithm=entry.algorithm,
        segments=entry.segments, config_override=config or None,
        zero_flow=fit_arguments.get("zero_flow"), nuts_sampler=nuts_sampler,
        seed=seed,
        all_data_log_std=float(np.log(fit.rating.model.q_transform.std_)))


def reloo(rating, *, k_threshold: float = None, fraction: float = None,
          max_refits: int = None) -> dict:
    """Refit the measurements PSIS could not handle, and rescore them exactly.

    Parameters
    ----------
    rating : RatingModel
        A fitted Bayesian rating. Must have been fitted with ``method="nuts"``:
        PSIS-LOO's Pareto-k is meaningless for a variational fit, so there is no
        trustworthy list of measurements to repair.
    k_threshold, fraction, max_refits
        See :func:`refine`.

    Returns
    -------
    dict
        See :func:`refine`.

    Notes
    -----
    Costs one NUTS fit per refitted measurement, about twenty seconds each on these
    models. Read ``rating.pareto_k()`` first to see how many that will be.

    Examples
    --------
    >>> rating = lrc.fit_rating(measurements, model="bdrc_gplm0")   # doctest: +SKIP
    >>> refined = lrc.evaluate.reloo(rating)                         # doctest: +SKIP
    >>> refined["elpd_loo"], refined["elpd_loo_psis"], refined["n_refits"]
    """
    fit = rating.result
    if fit is None or not fit.ok:
        return metrics_module.unavailable("fit not ok")
    if not fit.bayes:
        fit.bayes = rating.entry.bayes_metrics(fit)
    return refine(fit, rating.sample, rating.entry,
                  fit_arguments=getattr(rating, "fit_arguments", None),
                  k_threshold=k_threshold, fraction=fraction,
                  max_refits=max_refits)


def refine(fit, sample, entry, *, fit_arguments: dict = None,
           k_threshold: float = None, fraction: float = None,
           max_refits: int = None) -> dict:
    """Repair a fit's ``elpd_loo`` by refitting its flagged measurements.

    The work behind :func:`reloo`, taking the pieces rather than a
    :class:`~limnotech_rating_curves.ratings.RatingModel`, so the batch report can
    call it with the fits and samples it already holds.

    Parameters
    ----------
    fit : FitResult
        A completed fit whose ``bayes`` scores have been computed.
    sample : Sample or pandas.DataFrame
        The measurements it was fitted to, in the order it saw them.
    entry : ModelEntry
        The catalog entry for `fit`, which carries the model configuration a refit
        has to reproduce.
    fit_arguments : dict, optional
        The keyword arguments the all-data fit was made with - ``seed`` and
        ``zero_flow`` are the two that matter. Defaults are used for anything
        missing, which reproduces a default fit exactly and a customized one only
        approximately.
    k_threshold : float, optional
        Pareto-k above which a measurement is refitted. Defaults to
        ``settings.PARETO_K_GOOD`` (0.7).
    fraction : float, optional
        Share of the sample that may be refitted. Defaults to
        ``settings.RELOO_MAX_FRACTION``.
    max_refits : int, optional
        A hard ceiling on refits, applied after `fraction`.

    Returns
    -------
    dict
        The fields in ``metrics.BAYES_FIELDS`` with ``elpd_loo`` repaired, plus:

        ==================  ====================================================
        ``elpd_loo_psis``   the unrepaired PSIS estimate, for comparison
        ``n_flagged``       measurements above `k_threshold`
        ``n_refits``        how many were actually refitted
        ``n_over_budget``   flagged measurements left approximate
        ``refit_index``     which measurements were refitted, as **rows of the
                            sample** - not positions in the backend's own order
                            (see :func:`ordered_frame`)
        ``refit_budget``    the cap that applied
        ``k_threshold``     the threshold used
        ``pareto_k``        the **original** per-measurement values, in the same
                            order the fit's PSIS scores already report them
        ==================  ====================================================

        ``elpd_waic`` and ``p_waic`` are carried through unchanged: WAIC does not
        reweight anything and so has nothing to repair.

    Notes
    -----
    Costs one NUTS fit per refitted measurement, about twenty seconds each on these
    models.

    A fit that is not eligible - a variational fit, a family with no posterior, a
    ratingcurve fit whose posterior did not survive a process boundary - is returned
    with its PSIS scores untouched rather than raising, so turning
    ``settings.RELOO`` on cannot break a run that contains one.
    """
    import arviz as az

    if fit is None or not fit.ok:
        return metrics_module.unavailable("fit not ok")
    scores = dict(fit.bayes or {})
    refused = metrics_module.refused_for_advi(fit)
    if refused is not None:
        return scores or refused
    if fit.family not in ELIGIBLE_FAMILIES:
        return scores or metrics_module.unavailable(
            f"{fit.family} ratings have no posterior to refit")
    if not np.isfinite(scores.get("elpd_loo", np.nan)):
        return scores

    if fit.family == "ratingcurve" and (fit.rating is None or fit.rating.idata is None):
        return {**scores, "note": "no posterior in memory to refit from"}
    if fit.family == "bdrc" and fit.log_likelihood is None:
        return {**scores, "note": "no pointwise log-likelihood to refit from"}

    frame = ordered_frame(sample, fit.family)
    loo_orig, log_offset = _psis_pointwise(fit, rating=fit.rating)
    pareto_k = np.asarray(loo_orig.pareto_k.values, float)
    threshold = settings.PARETO_K_GOOD if k_threshold is None else float(k_threshold)

    flagged = np.flatnonzero(pareto_k > threshold)
    budget = refit_budget(len(frame), fraction=fraction, max_refits=max_refits)
    # worst first, so a budget too small to cover every flagged measurement is spent
    # where the approximation is furthest from the truth
    ranked = flagged[np.argsort(-pareto_k[flagged])]
    selected = np.sort(ranked[:budget])
    over_budget = np.sort(ranked[budget:])

    position = frame[POSITION_COLUMN].to_numpy(int)
    extra = {"elpd_loo_psis": float(scores["elpd_loo"]),
             "n_flagged": int(flagged.size), "n_refits": int(selected.size),
             "n_over_budget": int(over_budget.size),
             "refit_index": position[selected].tolist(),
             "refit_budget": int(budget), "k_threshold": threshold}

    if selected.size == 0:
        log.info("%s: no measurement above Pareto-k %.2f; PSIS-LOO stands",
                 fit.key, threshold)
        return {**scores, **extra,
                "note": f"no Pareto-k above {threshold:g}; PSIS-LOO unchanged"}

    log.info("%s: refitting %d of %d measurement(s) above Pareto-k %.2f (budget %d)",
             fit.key, selected.size, flagged.size, threshold, budget)

    # reloo refits wherever the k it is handed exceeds the threshold, so the budget is
    # imposed by handing it a copy whose other values are already acceptable. The
    # originals are put back on the returned scores below.
    masked = loo_orig.copy()
    budgeted_k = np.zeros_like(pareto_k)
    budgeted_k[selected] = pareto_k[selected]
    masked.pareto_k.values[:] = budgeted_k

    wrapper = _wrapper_for(fit, entry, frame, fit_arguments)
    try:
        refined = az.reloo(wrapper, loo_orig=masked, k_thresh=threshold, verbose=False)
    except Exception as exc:  # noqa: BLE001 - a failed repair must leave PSIS standing
        log.warning("%s: reloo failed, keeping PSIS-LOO: %s", fit.key, exc)
        return {**scores, **extra, "n_refits": 0, "refit_index": [],
                "note": f"reloo failed ({type(exc).__name__}: {str(exc)[:80]}); "
                        f"elpd_loo is the PSIS estimate"}

    n = int(refined.n_data_points)
    note = ""
    if over_budget.size:
        note = (f"{over_budget.size} measurement(s) above Pareto-k {threshold:g} "
                f"left approximate by the {budget}-refit budget")
        log.warning("%s: %s", fit.key, note)

    return {**scores,
            "elpd_loo": float(refined.elpd_loo) + n * log_offset,
            "se_loo": float(refined.se),
            "p_loo": float(refined.p_loo),
            "pareto_k_max": float(np.nanmax(pareto_k)) if pareto_k.size else np.nan,
            "pct_k_high": (float(100 * np.mean(pareto_k > threshold))
                           if pareto_k.size else np.nan),
            "n_obs": n,
            "pareto_k": pareto_k.tolist(),
            **extra,
            "note": note}


# --- the hierarchical family ---------------------------------------------------
# Its refits are a different shape from the two above. There, one refit drops one
# measurement from one site's own model. Here every site shares a single model, so
# dropping one measurement means refitting *all* of them - the cost per flagged point
# is a whole joint fit, and the budget has to be read with that in mind.

def _held_out_log_likelihood(fit, site: str, stage_ft: float,
                             discharge_cfs: float) -> np.ndarray:
    """Per-draw log density of one measurement under a joint fit that never saw it.

    Evaluated from the refit's own posterior for that site, through the same two
    methods the likelihood is built from - :meth:`SitePosterior.log_discharge` for the
    mean and :meth:`SitePosterior.scatter` for the spread - so this cannot drift away
    from the model it is scoring.

    Parameters
    ----------
    fit : HierarchicalFit
        A fit made without this measurement.
    site : str
        Which site it belongs to. Must still be present in `fit`.
    stage_ft, discharge_cfs : float
        The held-out measurement.

    Returns
    -------
    numpy.ndarray
        Shape ``(draws,)``, a density on raw log discharge. A draw that puts its stage
        of zero flow above this measurement's stage carries no flow there, so it gives
        the measurement a density of zero, recorded as ``-inf`` rather than dropped -
        that draw genuinely says the measurement is impossible, and averaging it away
        would quietly flatter the fold. All ``-inf`` means every draw said so, which
        the caller reports rather than folding into a total.
    """
    posterior = fit._posteriors.get(site)
    if posterior is None:
        return np.full(1, -np.inf)
    stage = np.atleast_1d(float(stage_ft))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.asarray(posterior.log_discharge(stage), float)[:, 0]
        spread = np.asarray(posterior.scatter(stage), float)[:, 0]
        density = normal_logpdf(np.log(float(discharge_cfs)), mean, spread)
    # below a draw's stage of zero flow the mean is undefined, not merely small
    return np.where(np.isfinite(mean) & np.isfinite(spread) & (spread > 0),
                    density, -np.inf)


def refine_hierarchical(scores: dict, idata, observations, model, *,
                        seed: int = settings.SEED, nuts_sampler: str = None,
                        k_threshold: float = None, fraction: float = None,
                        max_refits: int = None) -> dict:
    """Repair each site's ``elpd_loo`` by refitting the joint model without its
    flagged measurements.

    Parameters
    ----------
    scores : dict
        ``{site: scores}`` from :func:`..models.hierarchical._site_scores`.
    idata : arviz.InferenceData
        The all-data joint posterior. Carried for the record; the repair works from
        `observations` and refits from scratch.
    observations : pandas.DataFrame
        One row per fitted measurement, in log-likelihood order, with ``site``,
        ``stage_ft`` and ``discharge_cfs``.
    model : pymc.Model
        The all-data model, for its site ordering.
    seed : int
        Base seed; the fold holding out measurement *i* uses ``seed + 1 + i``.
    nuts_sampler : str, optional
    k_threshold : float, optional
        Pareto-k above which a measurement is refitted. Defaults to
        ``settings.PARETO_K_GOOD``.
    fraction, max_refits
        Budget, as :func:`refit_budget` applies them. **Set ``max_refits``**: the
        default fraction allows a quarter of every measurement in the fit, and each
        one costs a full joint fit.

    Returns
    -------
    dict
        `scores`, with each refitted site's ``elpd_loo`` repaired and
        ``elpd_loo_psis``, ``n_refits`` and ``n_flagged`` recorded.
    """
    from ..models.hierarchical import fit_hierarchical

    threshold = settings.PARETO_K_GOOD if k_threshold is None else float(k_threshold)
    sites = np.asarray(observations["site"].to_numpy())
    # Scattered into place by site rather than concatenated in site order. The two
    # agree today, because `_observations` lays sites out in blocks - but a k attached
    # to the wrong measurement is still a valid float at a valid index, so the
    # alignment is made explicit instead of assumed.
    pareto_k = np.full(sites.size, np.nan)
    for site in dict.fromkeys(sites):
        columns = np.flatnonzero(sites == site)
        values = np.asarray((scores.get(site) or {}).get("pareto_k", []), float)
        if values.size != columns.size:
            log.warning("cannot reloo: %s has %d Pareto-k values for %d measurements",
                        site, values.size, columns.size)
            return scores
        pareto_k[columns] = values
    if not np.isfinite(pareto_k).all():
        log.warning("cannot reloo: %d measurement(s) have no Pareto-k",
                    int((~np.isfinite(pareto_k)).sum()))
        return scores

    flagged = np.flatnonzero(pareto_k > threshold)
    budget = refit_budget(sites.size, fraction=fraction, max_refits=max_refits)
    # Worst first: the budget should be spent where PSIS is furthest from the truth.
    order = flagged[np.argsort(-pareto_k[flagged])][:budget]
    log.warning("reloo: %d of %d measurement(s) above Pareto-k %g; refitting %d, "
                "each a full joint fit - expect roughly %d minute(s)",
                flagged.size, sites.size, threshold, order.size,
                max(1, round(order.size * 55 / 60)))
    if not order.size:
        return scores

    repaired = {site: dict(entry) for site, entry in scores.items()}
    for site, entry in repaired.items():
        entry.setdefault("elpd_loo_psis", entry.get("elpd_loo"))
        entry.setdefault("n_refits", 0)
        entry["n_flagged"] = int(np.sum(pareto_k[sites == site] > threshold))

    from scipy.special import logsumexp

    from ..core import Sample
    for position in order:
        index = int(position)
        site = str(sites[index])
        kept = observations.drop(observations.index[index])
        rebuilt = {name: Sample.of(rows, site_id=name, source="reloo")
                   for name, rows in kept.groupby("site", sort=False)}
        try:
            fold = fit_hierarchical(rebuilt, reference=False, seed=seed + 1 + index,
                                    nuts_sampler=nuts_sampler, progressbar=False)
            exact = _held_out_log_likelihood(
                fold, site, float(observations["stage_ft"].iloc[index]),
                float(observations["discharge_cfs"].iloc[index]))
        except Exception as exc:  # noqa: BLE001 - one bad fold must not sink the rest
            log.warning("reloo fold %d (%s) failed: %s: %s", index, site,
                        type(exc).__name__, exc)
            continue
        if not np.isfinite(exact).any():
            # Every draw gave this measurement no flow. At a two-measurement site that
            # is the usual outcome of dropping the lower point: the refit re-places the
            # stage of zero flow against the one point left, above the held-out stage.
            # The exact ELPD is then -inf, which is a true statement about the model
            # and not a number to add into a total, so PSIS's estimate is left alone.
            log.warning("reloo fold %d (%s, stage %.2f ft): every draw puts the stage "
                        "of zero flow above it, so its exact density is zero; keeping "
                        "the PSIS estimate for this measurement", index, site,
                        float(observations["stage_ft"].iloc[index]))
            continue
        # log mean exp over draws: the exact leave-one-out predictive density
        exact_elpd = float(logsumexp(exact) - np.log(exact.size))
        entry = repaired[site]
        columns = np.flatnonzero(sites == site)
        within = int(np.flatnonzero(columns == index)[0])
        approximate = _pointwise_elpd(scores[site], within)
        entry["elpd_loo"] = float(entry["elpd_loo"]) - approximate + exact_elpd
        # keep the pointwise terms summing to the total, so a second pass subtracts
        # what is actually there rather than the estimate it already replaced
        pointwise = list(entry["elpd_loo_i"])
        pointwise[within] = exact_elpd
        entry["elpd_loo_i"] = pointwise
        entry["n_refits"] = int(entry.get("n_refits", 0)) + 1

    for site, entry in repaired.items():
        refits, flagged_here = entry.get("n_refits", 0), entry.get("n_flagged", 0)
        if not refits:
            continue
        n = entry.get("n_obs") or 0
        entry["elpd_per_obs"] = entry["elpd_loo"] / n if n else np.nan
        remaining = flagged_here - refits
        if remaining > 0:
            # PSIS is still carrying some of this site's points, so the estimate is
            # still not one to quote - a partial repair is not a repair
            entry["note"] = (f"{refits} of {flagged_here} flagged measurement(s) "
                             f"refitted exactly; {remaining} still approximate")
        else:
            # every point PSIS could not handle was refitted, so what made the
            # estimate unusable is gone even though the original Pareto-k remain
            entry["reliable"] = True
            entry["note"] = (f"all {refits} flagged measurement(s) refitted exactly; "
                             f"Pareto-k shown are the original PSIS values")
    return repaired


def _pointwise_elpd(entry: dict, position: int) -> float:
    """The PSIS pointwise ELPD this site's `position`-th measurement contributed.

    Recorded by :func:`..models.hierarchical._site_scores` when it slices the joint
    score, so a repair can subtract exactly what it is replacing rather than an
    average standing in for it.
    """
    values = entry.get("elpd_loo_i")
    if values is None:
        raise KeyError("elpd_loo_i is required to repair a single measurement")
    return float(np.asarray(values, float).ravel()[position])
