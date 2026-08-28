"""Expected information gain from a gauging, as a function of stage.

For a power-law rating fitted by ``ratingcurve``, scores candidate stages by the
mutual information between the rating's parameters and a hypothetical gauging at that
stage, under a Gaussian moment-match of the predictive:

    gain(h) = 0.5 * ln(1 + var[ln q(h)] / sigma**2)

where ``var[ln q(h)]`` is the variance of the mean curve across posterior draws,
excluding the gauging noise ``sigma``.

For a one-segment power law ``var[ln q(h)]`` is quadratic in ``ln(h - hs)`` with no
interior maximum, so the gain increases monotonically toward the ends of the stage
range and is largest at whichever end the candidate grid stops. Read it as a profile
of where the curve is least constrained, and supply `weight` to express which stages
can actually be gauged.
"""

import numpy as np
import pandas as pd

from .. import settings


def acquisition_curve(fit, stage=None, *, weight=None, measurement_sd=None,
                      max_draws: int = settings.ACQUISITION_MAX_DRAWS,
                      seed: int = None) -> pd.DataFrame:
    """Expected information gain from one gauging at each stage.

    Parameters
    ----------
    fit : BayesianRating or Fit
        A fitted power-law rating.
    stage : array-like, optional
        Candidate stages (feet). Defaults to the fitted curve's grid.
    weight : array-like or callable, optional
        Per-stage multiplier applied to the gain. A callable receives the candidate
        stages and returns one weight each. See :func:`stage_weight`.
    measurement_sd : float, optional
        Standard deviation of a new gauging in log-discharge units. Defaults to the
        rating's fitted ``sigma``.
    max_draws : int, optional
        Posterior draws to use.
    seed : int, optional
        Seed for subsampling draws. Defaults to :data:`..settings.SEED`.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage_ft``, ``discharge_cfs``, ``curve_sd_ln`` (curve uncertainty
        excluding gauging noise, as a fractional error), ``gain_nats``, and - when
        `weight` is given - ``weight`` and ``weighted_gain_nats``. One row per
        candidate stage above the stage of zero flow, ordered by stage.
    """

    seed = settings.SEED if seed is None else seed
    stage = _candidate_stages(fit, stage)

    log_curve, params = _curve_draws(_rating_of(fit), stage, max_draws, seed)
    spread, median, keep = _summarize(log_curve)

    noise = (float(np.mean(params["sigma"])) if measurement_sd is None
             else float(measurement_sd))
    if not noise > 0:
        raise ValueError("measurement_sd must be positive")

    table = pd.DataFrame({
        "stage_ft": stage,
        "discharge_cfs": median,
        "curve_sd_ln": spread,
        "gain_nats": 0.5 * np.log1p((spread / noise) ** 2),
    })
    if weight is not None:
        table["weight"] = _as_weight(weight, stage)
        table["weighted_gain_nats"] = table["gain_nats"] * table["weight"]
    return table[keep].reset_index(drop=True)


def stage_weight(observed_stage, stage, *, bandwidth: float = None):
    """Relative frequency of each candidate stage in a set of observed stages.

    A Gaussian kernel density over `observed_stage`, evaluated at `stage` and scaled
    to a maximum of one, for use as the `weight` argument to
    :func:`acquisition_curve`.

    Parameters
    ----------
    observed_stage : array-like
        Stages to build the density from (feet).
    stage : array-like
        Candidate stages to evaluate at.
    bandwidth : float, optional
        Kernel bandwidth in feet. Defaults to Scott's rule.

    Returns
    -------
    numpy.ndarray
        Weights in ``[0, 1]``, one per candidate stage.
    """
    observed_stage = np.asarray(observed_stage, float)
    observed_stage = observed_stage[np.isfinite(observed_stage)]
    if observed_stage.size < 2:
        raise ValueError("need at least two stages to build a weight from")
    stage = np.asarray(stage, float).ravel()

    if bandwidth is None:
        bandwidth = float(observed_stage.std()) * observed_stage.size ** (-0.2)
    if not bandwidth > 0:
        raise ValueError("observed_stage has no spread to build a weight from")

    density = np.exp(-0.5 * ((stage[:, None] - observed_stage[None, :])
                             / bandwidth) ** 2).mean(axis=1)
    peak = density.max()
    return density / peak if peak > 0 else density


def plot_acquisition(fit, stage=None, *, weight=None, measurement_sd=None,
                     axes=None, stage_label: str = "stage (ft)"):
    """Plot the rating curve above its acquisition function.

    Parameters
    ----------
    fit : BayesianRating or Fit
        A fitted power-law rating.
    stage : array-like, optional
        Candidate stages (feet). Defaults to the fitted curve's grid.
    weight : array-like or callable, optional
        Per-stage multiplier, as for :func:`acquisition_curve`. When given, both the
        raw and weighted gain are drawn.
    measurement_sd : float, optional
        Standard deviation of a new gauging in log-discharge units.
    axes : pair of matplotlib.axes.Axes, optional
        Axes to draw into, sharing an x-axis. Created if not given.
    stage_label : str, optional
        X-axis label.

    Returns
    -------
    pair of matplotlib.axes.Axes
        The rating axes and the acquisition axes.
    """
    import matplotlib.pyplot as plt

    rating = _rating_of(fit)
    scored = acquisition_curve(fit, stage, weight=weight,
                               measurement_sd=measurement_sd)

    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    curve_ax, gain_ax = axes

    curve = scored["discharge_cfs"]
    curve_ax.fill_between(scored["stage_ft"],
                          curve * np.exp(-1.96 * scored["curve_sd_ln"]),
                          curve * np.exp(1.96 * scored["curve_sd_ln"]),
                          color="tab:blue", alpha=0.18,
                          label="curve uncertainty (95%)")
    curve_ax.plot(scored["stage_ft"], curve, color="tab:blue", lw=2.2,
                  label="rating curve")
    curve_ax.scatter(rating.model.h_obs, rating.model.q_obs, s=45, facecolor="white",
                     edgecolor="black", zorder=5, label="measurements")
    curve_ax.set_yscale("log")
    curve_ax.set_ylabel("discharge (cfs)")
    curve_ax.grid(True, which="both", alpha=0.25)
    curve_ax.legend(fontsize="small", loc="lower right")

    gain_ax.plot(scored["stage_ft"], scored["gain_nats"], color="tab:purple", lw=2.0,
                 label="expected information gain")
    if "weighted_gain_nats" in scored:
        gain_ax.plot(scored["stage_ft"], scored["weighted_gain_nats"],
                     color="tab:orange", lw=2.0, label="weighted")
    gain_ax.set_ylim(bottom=0)
    gain_ax.set_xlabel(stage_label)
    gain_ax.set_ylabel("gain (nats)")
    gain_ax.grid(True, alpha=0.25)
    gain_ax.legend(fontsize="small", loc="upper left")

    return curve_ax, gain_ax


def _rating_of(fit):
    """Return the fitted BayesianRating from either a rating or a Fit."""
    rating = getattr(fit, "rating", fit)
    if not getattr(rating, "fitted", False):
        raise ValueError("the rating has not been fitted")
    if rating.algorithm != "power_law":
        raise ValueError(f"expected a power-law rating, got {rating.algorithm!r}")
    return rating


def _candidate_stages(fit, stage):
    """Stages to score on: `stage` if given, else the fitted curve's grid."""
    if stage is not None:
        return np.asarray(stage, float).ravel()
    curve = getattr(fit, "curve", None)
    if curve is not None and "stage_ft" in curve:
        return np.asarray(curve["stage_ft"], float)
    from ..core import padded_stage_grid
    return padded_stage_grid(_rating_of(fit).model.h_obs)


def _draws(rating, max_draws: int, seed: int) -> dict:
    """Power-law parameter draws, denormalized to log-discharge units.

    ratingcurve fits in a standardized log space, so ``a``, ``b`` and ``sigma`` on the
    posterior are z-scores. They are rescaled here by the same transform
    ``PowerLawRating.equation`` applies to the posterior means, giving parameters of
    ``ln(q) = a + sum(b[i] * ln(max(h - hs[i], 0) + ho[i]))``.

    Returns
    -------
    dict
        ``a`` (draws,), ``b`` (draws, segments), ``hs`` (draws, segments) and
        ``sigma`` (draws,), plus ``ho`` (segments,), the per-segment offset.
    """
    posterior = rating.idata.posterior.stack(sample=("chain", "draw"))
    n_draws = posterior.sizes["sample"]
    index = np.arange(n_draws)
    if max_draws and n_draws > max_draws:
        index = np.sort(np.random.default_rng(seed).choice(n_draws, max_draws,
                                                           replace=False))

    def draws_of(name):
        return np.asarray(posterior[name].values,
                          float).reshape(-1, n_draws).T[index]

    scale = float(rating.model.q_transform.std_)
    location = float(rating.model.q_transform.mean_)
    exponents = draws_of("b") * scale
    offsets = np.ones(exponents.shape[1])
    offsets[0] = 0.0
    return {
        "a": draws_of("a").ravel() * scale + location,
        "b": exponents,
        "hs": draws_of("hs"),
        "sigma": draws_of("sigma").ravel() * scale,
        "ho": offsets,
    }


def _curve_draws(rating, stage, max_draws, seed):
    """Draws of the log mean curve, and the parameters behind them.

    The mean curve carries no gauging noise, so its spread across draws is the
    parameter uncertainty alone. Stages at or below the first breakpoint give
    ``-inf``.

    Returns
    -------
    tuple
        ``log_curve`` (draws, n_stage) and the parameter dict from :func:`_draws`.
    """
    params = _draws(rating, max_draws, seed)
    above = np.clip(stage[None, :, None] - params["hs"][:, None, :], 0.0, None)
    with np.errstate(divide="ignore"):
        design = np.log(above + params["ho"][None, None, :])
    log_curve = params["a"][:, None] + np.einsum("dsk,dk->ds", design, params["b"])
    return log_curve, params


def _summarize(log_curve):
    """Per-stage spread and median of the log mean curve, and which stages to keep.

    Draws that put a stage below zero flow contribute ``-inf`` and are excluded from
    the summary rather than allowed to make it infinite.
    """
    finite = np.isfinite(log_curve)
    masked = np.where(finite, log_curve, np.nan)
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(masked, axis=0)
        median = np.exp(np.nanmedian(masked, axis=0))
    mostly_dry = finite.mean(axis=0) > settings.ACQUISITION_ZERO_FLOW_DRAW_FRACTION
    return spread, median, mostly_dry


def _as_weight(weight, stage):
    """A `weight` argument, callable or array, as one value per candidate stage."""
    values = np.asarray(weight(stage) if callable(weight) else weight, float).ravel()
    if values.size != stage.size:
        raise ValueError(f"weight must give one value per candidate stage: got "
                         f"{values.size} for {stage.size} stages")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("weight must be finite and non-negative")
    return values
