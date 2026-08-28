import logging

import numpy as np
import pandas as pd

from .. import settings

log = logging.getLogger(__name__)


def _idata(fit_or_model):
    """The InferenceData behind an estimator or a Fit."""
    fit = getattr(fit_or_model, "result", fit_or_model)
    idata = getattr(fit, "idata", None)
    if idata is None and getattr(fit, "rating", None) is not None:
        # the least-squares ratings have a rating object and no posterior at all, so
        # this is a plain absence rather than an error
        idata = getattr(fit.rating, "idata", None)
    return idata


def posterior_summary(fit_or_model, var_names=None) -> pd.DataFrame:
    """ArviZ posterior summary of a fit's parameters.

    Parameters
    ----------
    fit_or_model : RatingModel or Fit
        A completed fit.
    var_names : list of str, optional
        Restrict to these parameters.

    Returns
    -------
    pandas.DataFrame
        Mean, standard deviation, credible interval, R-hat and ESS per parameter.
        Empty when the fit kept no posterior - which happens whenever it crossed a
        process boundary, since neither a PyMC model nor an InferenceData survives
        that.
    """
    import arviz as az
    idata = _idata(fit_or_model)
    if idata is None:
        log.info("this fit kept no posterior, so there is nothing to summarize")
        return pd.DataFrame()
    return az.summary(idata, var_names=var_names, round_to=None)


def convergence(fit_or_model, var_names=None) -> pd.DataFrame:
    """R-hat and effective sample size per parameter, with a pass/fail flag.

    Parameters
    ----------
    fit_or_model : RatingModel or Fit
        A completed fit.
    var_names : list of str, optional
        Restrict to these parameters.

    Returns
    -------
    pandas.DataFrame
        Indexed by parameter, with ``r_hat``, ``ess_bulk``, ``ess_tail`` and
        ``converged`` - the last True when R-hat is at or below
        ``settings.R_HAT_GOOD`` (1.01) *and* bulk ESS is at or above
        ``settings.ESS_GOOD`` (400). Worst R-hat first, so a problem is the first
        thing you see.

    Examples
    --------
    >>> rating.diagnostics()                      # doctest: +SKIP
    >>> rating.diagnostics()["converged"].all()    # doctest: +SKIP
    True
    """
    summary = posterior_summary(fit_or_model, var_names=var_names)
    if summary.empty:
        return pd.DataFrame(columns=["r_hat", "ess_bulk", "ess_tail", "converged"])
    table = pd.DataFrame({
        "r_hat": summary.get("r_hat"),
        "ess_bulk": summary.get("ess_bulk"),
        "ess_tail": summary.get("ess_tail"),
    })
    table["converged"] = ((table["r_hat"] <= settings.R_HAT_GOOD)
                          & (table["ess_bulk"] >= settings.ESS_GOOD))
    return table.sort_values("r_hat", ascending=False)


def convergence_report(fit_or_model) -> str:
    """A one-line verdict on whether a fit's sampler converged.

    Parameters
    ----------
    fit_or_model : RatingModel or Fit
        A completed fit.

    Returns
    -------
    str
        Either a clean bill of health with the worst R-hat and lowest ESS, or a
        statement of how many parameters failed and which was worst.
    """
    table = convergence(fit_or_model)
    if table.empty:
        return "no posterior kept, so convergence cannot be checked"
    worst_r_hat = float(table["r_hat"].max())
    lowest_ess = float(table["ess_bulk"].min())
    failed = table[~table["converged"]]
    if failed.empty:
        return (f"converged: worst R-hat {worst_r_hat:.4f} "
                f"(threshold {settings.R_HAT_GOOD}), lowest bulk ESS "
                f"{lowest_ess:.0f} (threshold {settings.ESS_GOOD})")
    return (f"NOT converged: {len(failed)} of {len(table)} parameter(s) failed. "
            f"Worst is {failed.index[0]} at R-hat {worst_r_hat:.4f}, bulk ESS "
            f"{lowest_ess:.0f}. Increase draws and tuning, or simplify the model.")


def plot_convergence(fit_or_model, ax=None, var_names=None):
    """Plot R-hat per parameter against its threshold - the Gelman-Rubin plot.

    A horizontal bar chart with the threshold marked. Bars past the line are the
    parameters whose chains did not agree.

    Parameters
    ----------
    fit_or_model : RatingModel or Fit
        A completed fit.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    var_names : list of str, optional
        Restrict to these parameters.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    table = convergence(fit_or_model, var_names=var_names)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(2.5, 0.32 * len(table) + 1.2)))
    if table.empty:
        ax.text(0.5, 0.5, "no posterior kept", ha="center", va="center",
                transform=ax.transAxes)
        return ax

    positions = np.arange(len(table))[::-1]
    colors = np.where(table["converged"], "#4c9f70", "#c44e52")
    ax.barh(positions, table["r_hat"], color=colors, height=0.7)
    ax.axvline(settings.R_HAT_GOOD, color="black", ls="--", lw=1.2,
               label=f"threshold {settings.R_HAT_GOOD}")
    ax.axvline(1.0, color="0.6", lw=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(table.index, fontsize="small")
    lower = min(0.999, float(table["r_hat"].min()) - 0.002)
    upper = max(settings.R_HAT_GOOD + 0.01, float(table["r_hat"].max()) * 1.02)
    ax.set_xlim(lower, upper)
    ax.set_xlabel("R-hat (Gelman-Rubin): between-chain over within-chain variance")
    ax.set_title("did the chains agree?")
    ax.legend(fontsize="small", loc="lower right")
    return ax


def plot_effective_sample_size(fit_or_model, ax=None, var_names=None):
    """Plot bulk and tail effective sample size per parameter.

    Parameters
    ----------
    fit_or_model : RatingModel or Fit
        A completed fit.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    var_names : list of str, optional
        Restrict to these parameters.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    table = convergence(fit_or_model, var_names=var_names)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(2.5, 0.32 * len(table) + 1.2)))
    if table.empty:
        ax.text(0.5, 0.5, "no posterior kept", ha="center", va="center",
                transform=ax.transAxes)
        return ax

    positions = np.arange(len(table))[::-1]
    ax.barh(positions + 0.18, table["ess_bulk"], height=0.34, color="#4c72b0",
            label="bulk (the centre of the distribution)")
    ax.barh(positions - 0.18, table["ess_tail"], height=0.34, color="#dd8452",
            label="tail (the credible interval)")
    ax.axvline(settings.ESS_GOOD, color="black", ls="--", lw=1.2,
               label=f"threshold {settings.ESS_GOOD}")
    ax.set_yticks(positions)
    ax.set_yticklabels(table.index, fontsize="small")
    ax.set_xlabel("effective sample size (independent-equivalent draws)")
    ax.set_title("how many independent draws are these worth?")
    ax.legend(fontsize="small", loc="lower right")
    return ax


def plot_trace(fit_or_model, var_names=None, **kwargs):
    """ArviZ trace plot: each chain's path, alongside its marginal density.

    The visual companion to R-hat. Well-mixed chains overlap and look like noise
    around a stable level; a chain that wandered off, or got stuck, is obvious here
    in a way a single number is not.

    Parameters
    ----------
    fit_or_model : RatingModel or Fit
        A completed fit.
    var_names : list of str, optional
        Restrict to these parameters.
    **kwargs
        Forwarded to ``arviz.plot_trace``.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes or None
        None when the fit kept no posterior.
    """
    import arviz as az
    idata = _idata(fit_or_model)
    if idata is None:
        log.info("this fit kept no posterior, so there is no trace to plot")
        return None
    return az.plot_trace(idata, var_names=var_names, **kwargs)


def plot_pareto_k(model, ax=None):
    """Plot per-measurement Pareto-k against stage - an influence diagnostic.

    Each point is one measurement, placed at its stage. Points above the threshold
    are the ones PSIS-LOO could not reweight away, meaning the fit leans on them
    heavily. Reading it against stage is the useful part: high k at the top of the
    stage range says the high-flow end of the rating rests on a single measurement,
    which is exactly the situation in which extrapolation should not be trusted.

    Parameters
    ----------
    model : RatingModel
        A fitted estimator (its Bayesian scores are computed if not already).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    table = model.pareto_k()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))
    values = table["pareto_k"].to_numpy(float)
    if not np.isfinite(values).any():
        ax.text(0.5, 0.5, "no Pareto-k available - PSIS-LOO needs an MCMC fit\n"
                          "(refit with method='nuts')", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xlabel(f"stage - {model.sample.stage_label}")
        ax.set_ylabel("Pareto-k")
        return ax
    colors = np.where(values > settings.PARETO_K_GOOD, "#c44e52", "#4c9f70")
    ax.scatter(table["stage_ft"], values, c=colors, s=60,
               edgecolor="black", linewidth=0.6, zorder=5)
    ax.axhline(settings.PARETO_K_GOOD, color="black", ls="--", lw=1.2,
               label=f"threshold {settings.PARETO_K_GOOD}")
    ax.set_xlabel(f"stage - {model.sample.stage_label}")
    ax.set_ylabel("Pareto-k")
    ax.set_title(f"{model.label}: which measurements is the fit leaning on?")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    return ax


def convergence_table(rating_set) -> pd.DataFrame:
    """Convergence verdicts for every model in a comparison, in one table.

    The first thing to look at after fitting several models: a model whose sampler
    did not converge has no business being ranked against the others, however good
    its scores look.

    Parameters
    ----------
    rating_set : RatingSet
        A completed comparison.

    Returns
    -------
    pandas.DataFrame
        One row per model with ``model``, ``n_parameters``, ``worst_r_hat``,
        ``lowest_ess_bulk``, ``n_failed`` and ``verdict``.
    """
    rows = []
    for model in rating_set:
        table = convergence(model)
        if table.empty:
            rows.append({"model": model.name, "n_parameters": 0,
                         "worst_r_hat": np.nan, "lowest_ess_bulk": np.nan,
                         "n_failed": np.nan, "verdict": "no posterior kept"})
            continue
        failed = int((~table["converged"]).sum())
        rows.append({
            "model": model.name, "n_parameters": len(table),
            "worst_r_hat": float(table["r_hat"].max()),
            "lowest_ess_bulk": float(table["ess_bulk"].min()),
            "n_failed": failed,
            "verdict": "converged" if failed == 0 else f"{failed} parameter(s) failed"})
    return pd.DataFrame(rows)
