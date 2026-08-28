import logging

import numpy as np

from .. import settings

log = logging.getLogger(__name__)

#: The observed random variable in ratingcurve's PyMC model; its log-likelihood
#: group carries one value per measurement.
RATINGCURVE_OBSERVED_VAR = "model_q"

#: Pareto-k above this means PSIS-LOO is unreliable at that observation.
PARETO_K_GOOD = settings.PARETO_K_GOOD

#: Share of a group's observations whose Pareto-k may exceed ``PARETO_K_GOOD`` before
#: :func:`elpd_by_group` marks that group's estimate unusable. The same threshold
#: ``evaluate.logo`` applies to blocks, and for the same reason: a few bad points out
#: of many is tolerable, a quarter of them is not a number to quote.
MAX_PCT_K_HIGH = 5.0

#: The Bayesian comparison fields, in table order.
BAYES_FIELDS = ("elpd_loo", "se_loo", "p_loo", "elpd_waic", "p_waic",
                "pareto_k_max", "pct_k_high", "n_obs")


def unavailable(note: str = "") -> dict:
    """A Bayesian-metrics dict of NaNs, with a note saying why.

    Used instead of raising, so a model whose ELPD cannot be computed still
    appears in the comparison table with an explanation rather than vanishing.

    Parameters
    ----------
    note : str
        Why the scores are unavailable.

    Returns
    -------
    dict
        Every field in ``BAYES_FIELDS`` set to NaN, plus ``note``.
    """
    scores = {field: np.nan for field in BAYES_FIELDS}
    scores["note"] = note
    return scores


def refused_for_advi(fit) -> "dict | None":
    """The Bayesian scores for an ADVI fit: none, by policy.

    PSIS-LOO reweights a fitted posterior toward each leave-one-out posterior. An
    ADVI fit is a variational approximation, not a sample from the posterior, and
    it is systematically narrower than the target, so the importance ratios are
    heavy-tailed and the estimate is not trustworthy. This is a property of the
    approximation, not of the sample size, so a low ``pareto_k`` on an ADVI fit
    does not rescue it and must not be read as a licence to use the number.
    Refuse it outright and let the table say so.

    Parameters
    ----------
    fit : FitResult
        A completed fit, whose ``config['method']`` records how it was fitted.

    Returns
    -------
    dict or None
        NaN scores with a note when the fit used ADVI, otherwise None so the
        caller carries on.
    """
    if (fit.config or {}).get("method") == "advi":
        return unavailable("ELPD not computed for ADVI fits (refit with NUTS)")
    return None


def ensure_log_likelihood(rating) -> None:
    """Attach a pointwise log-likelihood group to a ratingcurve fit's posterior.

    ratingcurve's fit does not request one, so ArviZ would otherwise have nothing
    to score. This is a no-op when the group is already present.

    The ratingcurve model pins an initial value on its breakpoint prior, and
    ``pm.compute_log_likelihood`` rebuilds the model graph, which PyMC refuses for
    models carrying non-default initial values. Clearing them for the duration of
    the computation sidesteps that - it only affects graph rebuilding, not the
    already-sampled posterior being scored - and they are restored afterwards.

    Parameters
    ----------
    rating : BayesianRating
        A fitted ratingcurve wrapper, whose ``idata`` is extended in place.
    """
    import pymc as pm

    idata = rating.idata
    if idata is not None and "log_likelihood" in idata.groups():
        return
    model = rating.model.model
    saved = dict(model.rvs_to_initial_values)
    try:
        for variable in model.rvs_to_initial_values:
            model.rvs_to_initial_values[variable] = None
        pm.compute_log_likelihood(idata, model=model, extend_inferencedata=True,
                                  progressbar=False)
    finally:
        model.rvs_to_initial_values.update(saved)


def _summarize(loo, waic, log_offset: float = 0.0) -> dict:
    """Pack ArviZ LOO and WAIC results into this package's comparison fields.

    Parameters
    ----------
    loo, waic : arviz.ELPDData
        The two ELPD estimates.
    log_offset : float, default 0.0
        Added to every pointwise log-likelihood (see ``docs/evaluate/metrics.md``): it
        shifts the ELPD point estimates by ``n * log_offset`` and nothing else.

    Returns
    -------
    dict
        The fields in ``BAYES_FIELDS``, plus an empty ``note``.
    """
    n = int(loo.n_data_points)
    pareto_k = np.asarray(loo.pareto_k.values, float)
    return {
        "elpd_loo": float(loo.elpd_loo) + n * log_offset,
        "se_loo": float(loo.se),
        "p_loo": float(loo.p_loo),
        "elpd_waic": float(waic.elpd_waic) + n * log_offset,
        "p_waic": float(waic.p_waic),
        "pareto_k_max": float(np.nanmax(pareto_k)) if pareto_k.size else np.nan,
        "pct_k_high": (float(100 * np.mean(pareto_k > PARETO_K_GOOD))
                       if pareto_k.size else np.nan),
        "n_obs": n,
        "pareto_k": pareto_k.tolist(),
        "note": "",
    }


def elpd_from_idata(idata, log_offset: float = 0.0) -> dict:
    """PSIS-LOO and WAIC for an InferenceData that carries a log-likelihood group.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must already have a ``log_likelihood`` group (see
        :func:`ensure_log_likelihood`).
    log_offset : float, default 0.0
        Constant added to every pointwise log-likelihood to move it into a common
        observation space. For ratingcurve fits, pass
        ``-log(model.q_transform.std_)``.

    Returns
    -------
    dict
        The fields in ``BAYES_FIELDS`` plus per-observation ``pareto_k``, or NaNs
        with a note if ArviZ could not score it.
    """
    import arviz as az
    try:
        loo = az.loo(idata)
        waic = az.waic(idata)
    except Exception as exc:  # noqa: BLE001
        return unavailable(f"{type(exc).__name__}: {str(exc)[:120]}")
    return _summarize(loo, waic, log_offset=log_offset)


def relative_efficiency(log_likelihood) -> float:
    """Relative MCMC efficiency of a pointwise log-likelihood matrix.

    ArviZ needs this scalar when the InferenceData carries no posterior group and
    it therefore cannot work out how autocorrelated the draws were. It is
    effective sample size over draw count, averaged across observations.

    Parameters
    ----------
    log_likelihood : array-like
        Shape ``(draws, observations)``.

    Returns
    -------
    float
        The efficiency, or 1.0 (independent draws) if it cannot be computed.
    """
    import arviz as az
    log_likelihood = np.asarray(log_likelihood, float)
    draws, observations = log_likelihood.shape
    try:
        per_observation = [float(az.ess(log_likelihood[np.newaxis, :, j]))
                           for j in range(observations)]
        efficiency = float(np.nanmean(per_observation)) / draws
        return efficiency if np.isfinite(efficiency) and efficiency > 0 else 1.0
    except Exception:  # noqa: BLE001
        return 1.0


def elpd_from_log_likelihood(log_likelihood) -> dict:
    """PSIS-LOO and WAIC from a raw pointwise log-likelihood matrix.

    Used for bdrc, which produces its log-likelihood directly and already in raw
    log-discharge space, so no offset is needed. The synthetic InferenceData has no
    posterior group, so the relative efficiency is supplied explicitly.

    Parameters
    ----------
    log_likelihood : array-like
        Shape ``(draws, observations)``.

    Returns
    -------
    dict
        The fields in ``BAYES_FIELDS`` plus per-observation ``pareto_k``, or NaNs
        with a note.
    """
    import arviz as az
    log_likelihood = np.asarray(log_likelihood, float)
    if log_likelihood.ndim != 2 or log_likelihood.size == 0:
        return unavailable("empty log-likelihood")
    idata = az.from_dict(log_likelihood={"obs": log_likelihood[np.newaxis, :, :]})
    try:
        loo = az.loo(idata, reff=relative_efficiency(log_likelihood))
        waic = az.waic(idata)
    except Exception as exc:  # noqa: BLE001
        return unavailable(f"{type(exc).__name__}: {str(exc)[:120]}")
    return _summarize(loo, waic)


def elpd_by_group(idata, groups, log_offset: float = 0.0,
                  observed_var: str = None) -> dict:
    """PSIS-LOO and WAIC for each group of observations within one joint fit.

    A hierarchical fit has one posterior over every site at once, and one pointwise
    log-likelihood spanning every measurement. This splits that single score into a
    per-group contribution, which is what puts an ELPD next to each site on the map.

    The split is exact rather than an approximation, and the reason is worth stating:
    PSIS fits its generalized Pareto to each observation's importance ratios
    *independently*, so ``loo_i`` and ``pareto_k`` for one measurement do not depend on
    which other measurements are in the call. Scoring once and slicing therefore gives
    the same numbers as scoring each group alone would - and unlike per-group calls, it
    cannot silently re-weight anything. Do not replace this with a loop over
    :func:`elpd_from_log_likelihood`.

    What the resulting number *means* needs care. It is group `g`'s contribution to the
    joint predictive score: how well the model predicts one more observation in `g`
    having seen the rest of `g` **and every other group**. It is not the score a model
    fitted to `g` alone would earn.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must carry a ``log_likelihood`` group (see :func:`ensure_log_likelihood`) and a
        ``posterior``, so ArviZ can work out the relative efficiency itself.
    groups : array-like
        One label per observation, in the order the log-likelihood is indexed by.
        Getting this order wrong misattributes every Pareto-k while leaving each index
        in range, which is why the caller is expected to have built it from the same
        frame the model was given.
    log_offset : float, default 0.0
        Added to every pointwise log-likelihood, to move it into a common observation
        space (see :func:`elpd_from_idata`). It shifts each group's ELPD by
        ``n_group * log_offset`` and leaves ``p_loo`` and ``se_loo`` alone.
    observed_var : str, optional
        Which variable in the log-likelihood group to score. Defaults to the only one
        present, and raises if there is more than one.

    Returns
    -------
    dict
        ``{group_label: scores}``, each holding the fields in ``BAYES_FIELDS`` plus
        per-observation ``pareto_k`` and ``elpd_loo_i``, and ``elpd_per_obs``,
        ``reliable`` and ``note``. On failure every group gets the same
        :func:`unavailable` dict, so a caller can still index by group.
    """
    import arviz as az
    from scipy.special import logsumexp

    groups = np.asarray(groups).ravel()
    try:
        available = list(idata.log_likelihood.data_vars)
    except AttributeError:
        return {label: unavailable("no log-likelihood group")
                for label in dict.fromkeys(groups)}
    if observed_var is None:
        if len(available) != 1:
            return {label: unavailable(
                        f"log-likelihood has {len(available)} variables; name one")
                    for label in dict.fromkeys(groups)}
        observed_var = available[0]

    values = np.asarray(idata.log_likelihood[observed_var].values, float)
    matrix = values.reshape(-1, values.shape[-1])          # (draws, observations)
    if matrix.shape[-1] != groups.size:
        return {label: unavailable(
                    f"{groups.size} group labels for {matrix.shape[-1]} observations")
                for label in dict.fromkeys(groups)}

    try:
        loo = az.loo(idata, pointwise=True, var_name=observed_var)
        waic = az.waic(idata, pointwise=True, var_name=observed_var)
    except Exception as exc:  # noqa: BLE001
        return {label: unavailable(f"{type(exc).__name__}: {str(exc)[:120]}")
                for label in dict.fromkeys(groups)}

    loo_i = np.asarray(loo.loo_i.values, float).ravel()
    waic_i = np.asarray(waic.waic_i.values, float).ravel()
    pareto_k = np.asarray(loo.pareto_k.values, float).ravel()
    # the log pointwise predictive density, which both effective parameter counts are
    # the gap between: p_loo = lppd - elpd_loo, p_waic = lppd - elpd_waic
    lppd_i = logsumexp(matrix, axis=0) - np.log(matrix.shape[0])

    scores = {}
    for label in dict.fromkeys(groups):            # first-seen order, not sorted
        columns = np.flatnonzero(groups == label)
        n = int(columns.size)
        elpd_loo = float(loo_i[columns].sum())
        elpd_waic = float(waic_i[columns].sum())
        lppd = float(lppd_i[columns].sum())
        k = pareto_k[columns]
        pct_high = float(100 * np.mean(k > PARETO_K_GOOD)) if n else np.nan
        # the standard error of a sum of n pointwise terms, as ArviZ computes it
        se_loo = float(np.sqrt(n) * np.std(loo_i[columns])) if n > 1 else np.nan
        scores[label] = {
            "elpd_loo": elpd_loo + n * log_offset,
            "se_loo": se_loo,
            "p_loo": lppd - elpd_loo,
            "elpd_waic": elpd_waic + n * log_offset,
            "p_waic": lppd - elpd_waic,
            "pareto_k_max": float(np.nanmax(k)) if n else np.nan,
            "pct_k_high": pct_high,
            "n_obs": n,
            "pareto_k": k.tolist(),
            # the per-measurement terms this group's elpd_loo is the sum of. Kept so a
            # reloo repair can subtract exactly the point it replaces rather than an
            # average standing in for it - see evaluate.exact_loo.refine_hierarchical.
            "elpd_loo_i": (loo_i[columns] + log_offset).tolist(),
            "elpd_per_obs": (elpd_loo + n * log_offset) / n if n else np.nan,
            "reliable": bool(n and np.isfinite(pct_high)
                             and pct_high <= MAX_PCT_K_HIGH),
            "note": "",
        }
        if not scores[label]["reliable"] and np.isfinite(pct_high):
            scores[label]["note"] = (
                f"PSIS unreliable: {pct_high:.0f}% of measurements above "
                f"k={PARETO_K_GOOD}")
    return scores


def pointwise_log_likelihood(fit):
    """The ``(draws, observations)`` log-likelihood behind a fit, in log-discharge space.

    The two Bayesian families reach it by different routes - bdrc carries the matrix
    on the fit, ratingcurve computes it from the live posterior and keeps it in a
    standardized space - so anything that needs the raw matrix (grouped ELPD, for
    one) goes through here rather than special-casing the backend again.

    Parameters
    ----------
    fit : FitResult
        A completed fit.

    Returns
    -------
    tuple of (numpy.ndarray, float) or None
        The matrix and the constant to add to every entry to put it in raw
        log-discharge space, or None when the fit carries no usable posterior
        (a least-squares family, an ADVI fit, or a fit that crossed a process
        boundary and left its PyMC model behind).
    """
    if not fit.ok:
        return None
    if (fit.config or {}).get("method") == "advi":
        return None
    if fit.log_likelihood is not None:
        # bdrc: already in raw log-discharge space
        return np.asarray(fit.log_likelihood, float), 0.0
    rating = getattr(fit, "rating", None)
    if rating is None or getattr(rating, "idata", None) is None:
        return None
    try:
        ensure_log_likelihood(rating)
        log_offset = -float(np.log(rating.model.q_transform.std_))
        values = rating.idata.log_likelihood[RATINGCURVE_OBSERVED_VAR].values
    except Exception as exc:  # noqa: BLE001
        # the message, not just the type: a bare "(ValueError)" in a log is the
        # difference between a diagnosable failure and a mystery
        log.info("no pointwise log-likelihood for %s (%s: %s)",
                 fit.key, type(exc).__name__, exc)
        return None
    # (chains, draws, observations) -> (draws, observations)
    matrix = np.asarray(values, float).reshape(-1, values.shape[-1])
    return matrix, log_offset


def pareto_k_per_point(fit) -> np.ndarray:
    """The per-measurement Pareto-k values from a scored fit.

    Pareto-k is the closest thing this package has to a per-measurement influence
    score: a high value means the PSIS-LOO reweighting could not remove that
    measurement, because the posterior depends on it too heavily. That is worth
    showing next to the point itself, which is what the interactive map does.

    Parameters
    ----------
    fit : FitResult
        A fit whose ``bayes`` scores have been computed.

    Returns
    -------
    numpy.ndarray
        One value per measurement, or an array of NaN of the right length when
        the scores are unavailable.
    """
    values = (fit.bayes or {}).get("pareto_k")
    if values is None:
        return np.full(fit.n, np.nan)
    values = np.asarray(values, float).ravel()
    if values.size != fit.n:
        padded = np.full(fit.n, np.nan)
        padded[:min(values.size, fit.n)] = values[:min(values.size, fit.n)]
        return padded
    return values


def comparison_rows(site) -> list:
    """One row per curve for a site's comparison table.

    Merges the in-sample scores with the Bayesian comparison fields, then appends
    every curve the package did not fit - the USGS published rating, a field
    spreadsheet's typed equation. Those have no posterior, so their Bayesian fields
    are NaN, which is the honest representation rather than an omission.

    Each row carries three identity columns that must be read together, because they
    answer different questions and conflating them is exactly how this table has
    misled before:

    ==============  ===========================================================
    ``model``       the key: a catalog key, ``"published_reference"``, or
                    ``"spreadsheet"``
    ``family``      who produced it: ``ratingcurve`` / ``bdrc`` /
                    ``polynomial`` / ``exponential`` for a fit, or the external
                    curve's ``kind``
    ``role``        what claim it makes: ``model`` (a rating this package
                    offers), ``spreadsheet_form`` (a least-squares form refitted
                    here to match a workbook's trendline type), or ``external``
                    (a curve typed by somebody else and merely evaluated)
    ==============  ===========================================================

    Parameters
    ----------
    site : SiteRating
        The site, its fits, and any external curves.

    Returns
    -------
    list of dict
        Ready for ``pandas.DataFrame``. Successful fits first, then the external
        curves.

    Notes
    -----
    ``nse``, ``rmse`` and ``r2_log`` score the posterior mean of the rating,
    ``E[exp(mu)]``, not the mean of the posterior predictive - the latter carries a
    factor of ``exp(sigma ** 2 / 2)`` with no finite value where sigma is large.
    ``elpd_loo`` and ``elpd_waic`` are pointwise *predictive densities* by
    definition: they ask how probable an unseen measurement is, which cannot be
    answered without the gaging scatter, and they are computed from the model's
    log-likelihood rather than from any exponentiated summary, so the same pathology
    does not reach them. See ``posterior_clarification.md``.
    """
    from ..models import catalog

    rows = []
    for fit in site.fits:
        if not fit.ok:
            continue
        row = {"model": fit.key, "model_label": fit.label, "family": fit.family,
               "role": catalog.role(fit.key), "n": fit.n, "status": fit.status}
        row.update({name: fit.metrics.get(name)
                    for name in ("nse", "rmse", "r2_log")})
        bayes = fit.bayes or {}
        row.update({field: bayes.get(field, np.nan) for field in BAYES_FIELDS})
        row["note"] = bayes.get("note", "")
        # absent means the scorer made no reliability claim, which is not the same as
        # claiming the estimate is unusable
        row["reliable"] = bool(bayes.get("reliable", True))
        rows.append(row)

    for external in site.external_curves():
        row = {"model": external.key, "model_label": external.label,
               "family": external.kind, "role": "external",
               "n": external.metrics.get("n"), "status": external.kind}
        row.update({name: external.metrics.get(name)
                    for name in ("nse", "rmse", "r2_log")})
        row.update({field: np.nan for field in BAYES_FIELDS})
        row["note"] = ""
        row["reliable"] = True     # no ELPD to be unreliable about
        row.update({key: value for key, value in external.detail.items()
                    if key in ("form", "axis", "equation")})
        rows.append(row)
    return rows


#: What each column of a comparison table means, for a notebook, a report, or a
#: tooltip. :func:`limnotech_rating_curves.view.mapview.build_map` wires these onto the
#: score table's column headers, so a reader can find out what a column is without
#: leaving the map.
METRIC_GLOSSARY = {
    "curve": "which rating this row scores: a fitted model, a least-squares "
             "spreadsheet form refitted here, or a curve drawn by somebody else",
    "n": "measurements the model was fitted on",
    "nse": "Nash-Sutcliffe efficiency on discharge, scored against the posterior "
           "mean of the rating. 1 is perfect, 0 is no better than the mean observed "
           "discharge",
    "rmse": "root-mean-square error in cfs, against the posterior mean of the "
            "rating. It is in data units, so high flows dominate it",
    "r2_log": "R-squared on log discharge, against the posterior mean of the rating. "
              "Read this one for a rating, since it weighs proportional error equally "
              "at all flows",
    "elpd_loo": "expected log predictive density for an unseen measurement, by "
                "PSIS-LOO. Higher is better, and only differences within one sample "
                "mean anything",
    "se_loo": "standard error of elpd_loo. A gap under about 2 of these is not "
              "resolvable",
    "p_loo": "effective number of parameters implied by LOO. Far above the model's "
             "real parameter count signals misfit or an over-influential point",
    "elpd_waic": "the same predictive score by WAIC, as a cross-check on elpd_loo",
    "p_waic": "effective parameter count implied by WAIC",
    "pareto_k_max": f"worst per-measurement Pareto-k. Above {PARETO_K_GOOD} the LOO "
                    f"estimate is unreliable, because one measurement dominates it",
    "pct_k_high": "percent of measurements whose Pareto-k is above the threshold",
    "n_obs": "measurements the ELPD was computed over",
}

#: How to *use* each column, next to what it means. The glossary says what a number
#: is; this says what to do with it - which is the part a reader comparing eight
#: curves actually needs, and the difference between a table that informs and one that
#: merely reports.
METRIC_INTUITION = {
    "curve": "Solid lines are Bayesian ratings, dashed and dotted are least-squares "
             "forms, and black is a curve this package did not fit.",
    "n": "Under about eight measurements, a segmented power law or a spline cannot be "
         "identified however good the sampler is.",
    "nse": "Reads like R² on discharge, so the highest flows dominate it. A curve can "
           "score 0.99 here and still be a factor of two wrong at low flow.",
    "rmse": "Compare it within a site, not across sites. It is in cfs, so a big river "
            "always looks worse than a small one.",
    "r2_log": "Rank ratings on this when there is no ELPD. Proportional error counts "
              "the same at 5 cfs and 500 cfs, which is how rating error is normally "
              "judged.",
    "elpd_loo": "The generalization column. In-sample NSE always favours the most "
                "flexible curve. ELPD estimates how the curve would do on a "
                "measurement it never saw, so flexibility that bought nothing costs "
                "it. It is a density over a measurement, so it necessarily includes "
                "the gaging scatter - unlike NSE and R²log, which score the "
                "rating itself.",
    "se_loo": "This turns the ranking into a decision. Two models less than about two "
              "standard errors apart are tied, and a tie goes to the simpler one.",
    "p_loo": "Compare it with the model's real parameter count. Much larger means the "
             "fit is leaning on one or two measurements.",
    "elpd_waic": "A second estimate of the same thing. Distrust both if it disagrees "
                 "materially with elpd_loo.",
    "pareto_k_max": "A reliability flag on elpd_loo, not a score. High almost always "
                    "means the highest-flow measurement is carrying the curve on its "
                    "own - or, in a hierarchical fit, that the site has so few "
                    "measurements that dropping one leaves the population holding it "
                    "up. A ⚠ on elpd_loo means the fit's own diagnostics say not "
                    "to quote it.",
    "pct_k_high": "How widespread the problem is. One bad point is survivable; a "
                  "quarter of them means the LOO estimate should not be quoted.",
    "n_obs": "Should equal n. If it does not, some measurements were dropped from the "
             "predictive score.",
}


def metric_hover(column: str, header: str = None) -> str:
    """The tooltip for one metrics-table column: what it is, then what to do with it.

    Parameters
    ----------
    column : str
        A key of ``METRIC_GLOSSARY``.
    header : str, optional
        The column header as displayed, when it differs from `column`.

    Returns
    -------
    str
        HTML with ``<br>`` breaks, ready for a Plotly ``hovertemplate``. Empty for a
        column with no glossary entry, so a caller can tell there is nothing to say.
    """
    definition = METRIC_GLOSSARY.get(column, "")
    if not definition:
        return ""
    intuition = METRIC_INTUITION.get(column, "")
    title = header or column
    text = f"<b>{title}</b><br>{_wrap(definition)}"
    if intuition:
        text += f"<br><br><i>{_wrap(intuition)}</i>"
    return text


def _wrap(text: str, width: int = 62) -> str:
    """Soft-wrap a sentence with ``<br>``, so a tooltip is not one long line."""
    import textwrap
    return "<br>".join(textwrap.wrap(text, width=width))
