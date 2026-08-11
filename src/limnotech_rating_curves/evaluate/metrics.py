import logging

import numpy as np

from .. import settings

log = logging.getLogger(__name__)

#: The observed random variable in ratingcurve's PyMC model; its log-likelihood
#: group carries one value per measurement.
RATINGCURVE_OBSERVED_VAR = "model_q"

#: Pareto-k above this means PSIS-LOO is unreliable at that observation.
PARETO_K_GOOD = settings.PARETO_K_GOOD

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
    """
    from ..models import catalog

    rows = []
    for fit in site.fits:
        if not fit.ok:
            continue
        row = {"model": fit.key, "model_label": fit.label, "family": fit.family,
               "role": catalog.role(fit.key), "n": fit.n, "status": fit.status}
        row.update({name: fit.metrics.get(name)
                    for name in ("nse", "rmse", "pbias_pct", "r2_log")})
        bayes = fit.bayes or {}
        row.update({field: bayes.get(field, np.nan) for field in BAYES_FIELDS})
        row["note"] = bayes.get("note", "")
        rows.append(row)

    for external in site.external_curves():
        row = {"model": external.key, "model_label": external.label,
               "family": external.kind, "role": "external",
               "n": external.metrics.get("n"), "status": external.kind}
        row.update({name: external.metrics.get(name)
                    for name in ("nse", "rmse", "pbias_pct", "r2_log")})
        row.update({field: np.nan for field in BAYES_FIELDS})
        row["note"] = ""
        row.update({key: value for key, value in external.detail.items()
                    if key in ("form", "axis", "equation")})
        rows.append(row)
    return rows


#: What each column of a comparison table means, for a notebook, a report, or a
#: tooltip. :func:`limnotech_rating_curves.view.mapview.build_map` wires these onto the
#: score table's column headers, so a reader can find out what a column is without
#: leaving the map.
METRIC_GLOSSARY = {
    "curve": "which rating this row scores - a fitted model, a least-squares "
             "spreadsheet form refitted here, or an external curve somebody else drew",
    "n": "measurements the model was fitted on",
    "nse": "Nash-Sutcliffe efficiency on discharge; 1 is perfect, 0 is no better "
           "than the mean observed discharge",
    "rmse": "root-mean-square error in cfs; in data units, so dominated by high flow",
    "pbias_pct": "percent bias; positive means the rating over-predicts on balance",
    "r2_log": "R-squared on log discharge - the one to read for a rating, because "
              "it weighs proportional error equally at all flows",
    "elpd_loo": "expected log predictive density for an unseen measurement, by "
                "PSIS-LOO. Higher is better; only differences on the same sample "
                "are meaningful",
    "se_loo": "standard error of elpd_loo; a gap under ~2 of these is not resolvable",
    "p_loo": "effective number of parameters implied by LOO; far above the model's "
             "real parameter count signals misfit or an over-influential point",
    "elpd_waic": "the same predictive score by WAIC, as a cross-check on elpd_loo",
    "p_waic": "effective parameter count implied by WAIC",
    "pareto_k_max": f"worst per-measurement Pareto-k; above {PARETO_K_GOOD} the "
                    f"LOO estimate is unreliable because one measurement dominates",
    "pct_k_high": "percent of measurements whose Pareto-k is above the threshold",
    "n_obs": "measurements the ELPD was computed over",
}

#: How to *use* each column, next to what it means. The glossary says what a number
#: is; this says what to do with it - which is the part a reader comparing eight
#: curves actually needs, and the difference between a table that informs and one that
#: merely reports.
METRIC_INTUITION = {
    "curve": "Solid lines are Bayesian ratings; dashed and dotted are least-squares "
             "forms; black is a curve this package did not fit.",
    "n": "Fewer than about eight measurements cannot identify a segmented power law "
         "or a spline, however good the sampler is.",
    "nse": "Reads like R² on discharge, so it is dominated by the highest flows. A "
           "curve can score 0.99 here and be a factor of two wrong at low flow.",
    "rmse": "Compare within a site, never across sites - it is in cfs, so a big river "
            "always looks worse than a small one.",
    "pbias_pct": "Near zero does not mean accurate: equal over- and under-prediction "
                 "cancels. Read it beside RMSE, not instead of it.",
    "r2_log": "The one to rank ratings on when you have no ELPD. Proportional error "
              "counts the same at 5 cfs and 500 cfs, which is how rating error is "
              "normally judged.",
    "elpd_loo": "This is the generalization column. In-sample NSE always favours the "
                "most flexible curve; ELPD estimates how it would do on a "
                "measurement it never saw, so it penalizes flexibility that bought "
                "nothing.",
    "se_loo": "Turn the ranking into a decision with this: two models less than about "
              "two standard errors apart are tied, and the simpler one wins a tie.",
    "p_loo": "Compare it with the model's real parameter count. Much larger means the "
             "fit is leaning on one or two measurements.",
    "elpd_waic": "A second estimate of the same thing. If it disagrees materially "
                 "with elpd_loo, distrust both.",
    "pareto_k_max": "A reliability flag on elpd_loo, not a score. High almost always "
                    "means the highest-flow measurement is carrying the curve alone.",
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
