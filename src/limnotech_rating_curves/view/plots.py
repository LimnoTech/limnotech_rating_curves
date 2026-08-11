import logging
from pathlib import Path

import numpy as np

from ..models import catalog

log = logging.getLogger(__name__)

#: How a published reference rating is drawn, everywhere.
REFERENCE_STYLE = dict(color="black", linewidth=2.0, linestyle="--")

#: Color for the part of a curve that sits outside the measured stage range. Every
#: family is tabulated on a grid padded past both ends of the measurements
#: (:func:`limnotech_rating_curves.core.padded_stage_grid`), so the curves are
#: comparable end to end; this is what marks the part of them that no measurement
#: supports.
EXTRAPOLATION_COLOR = "#e08214"

#: Legend text for that part.
EXTRAPOLATION_LABEL = "extrapolated beyond the measurements"

#: Plotly dash names to matplotlib linestyles, so a curve has the same shape in a
#: figure as it does on the map. The catalog names the styles once, in Plotly's
#: vocabulary, because that is where most of them are drawn.
_MATPLOTLIB_DASH = {
    "solid": "-", "dash": "--", "dot": ":", "dashdot": "-.",
    "longdash": (0, (8, 3)), "longdashdot": (0, (8, 2, 2, 2)),
}


def _safe_name(text) -> str:
    """A string safe to use in a file name."""
    return str(text).replace(":", "__").replace("/", "-").replace("\\", "-")


def limit_discharge_axis(ax, discharge, log_discharge: bool = True):
    """Scale the discharge axis to the measurements, not to the extrapolation.

    A power law drawn below the stage of zero flow falls to nothing, and on a log
    axis that one tail would compress every measurement into the top centimetre of
    the figure. The extrapolated part is drawn, and the axis stays where the
    measurements are - one decade past them either way.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to limit.
    discharge : array-like
        The measured discharges (cfs).
    log_discharge : bool, default True
        Whether the axis is log-scaled.

    Returns
    -------
    matplotlib.axes.Axes
    """
    values = np.asarray(discharge, float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return ax
    if log_discharge:
        ax.set_ylim(10.0 ** np.floor(np.log10(values.min()) - 1),
                    10.0 ** np.ceil(np.log10(values.max()) + 1))
    else:
        margin = 0.5 * ((values.max() - values.min()) or values.max())
        ax.set_ylim(max(0.0, values.min() - margin), values.max() + margin)
    return ax


def draw_curve(ax, curve, measured_range, *, color, label=None, level=None,
               linewidth=2.2, linestyle="-", band_alpha=0.18, zorder=3,
               extrapolation_color=EXTRAPOLATION_COLOR,
               extrapolation_label=None, x="stage_ft"):
    """Draw one fitted curve, marking the part outside the measured range.

    The whole curve is drawn in the extrapolation style and the measured part is
    drawn over it, so the two segments join without a gap and the curve always runs
    to the ends of the grid it was fitted on rather than stopping wherever its
    backend's own table happened to end.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    curve : pandas.DataFrame
        A fitted curve: `x`, ``discharge_cfs``, and ``lower``/``upper`` if a band
        is wanted.
    measured_range : tuple of float
        ``(lowest, highest)`` measured value on the `x` axis. Everything outside it
        is extrapolation.
    color : str
        The model's color, used for the measured part.
    label : str, optional
        Legend entry for the measured part.
    level : float, optional
        Draw the credible band at this level. ``None`` draws the line only, which
        is what keeps a multi-model figure readable.
    linewidth, linestyle, band_alpha, zorder
        Matplotlib styling for the measured part.
    extrapolation_color : str, optional
        Color for the extrapolated part. ``None`` keeps the model's own color and
        dots the line instead, which is what an overlay of several models needs -
        one shared color there would erase which curve is whose.
    extrapolation_label : str, optional
        Legend entry for the extrapolated part. ``None`` leaves it out, so an
        overlay can label it once rather than once per model.
    x : str, default "stage_ft"
        Column drawn on the x axis. ``plot_log_log`` passes effective head.

    Returns
    -------
    matplotlib.axes.Axes
    """
    curve = curve.sort_values(x)
    abscissa = curve[x].to_numpy(float)
    discharge = curve["discharge_cfs"].to_numpy(float)
    low, high = measured_range
    inside = (abscissa >= low) & (abscissa <= high)
    outside_color = color if extrapolation_color is None else extrapolation_color
    outside_style = ":" if extrapolation_color is None else linestyle

    if level is not None and "lower" in curve:
        lower = curve["lower"].to_numpy(float)
        upper = curve["upper"].to_numpy(float)
        drawable = np.isfinite(lower) & (lower > 0) & np.isfinite(discharge)
        if (drawable & ~inside).any():
            ax.fill_between(abscissa[drawable], lower[drawable], upper[drawable],
                            color=outside_color, alpha=band_alpha * 0.55,
                            zorder=zorder - 1)
        band = drawable & inside
        ax.fill_between(abscissa[band], lower[band], upper[band], color=color,
                        alpha=band_alpha, zorder=zorder - 1,
                        label=f"{level:.0%} credible band")

    if (~inside).any():
        ax.plot(abscissa, discharge, color=outside_color, lw=linewidth,
                ls=outside_style, alpha=0.85, zorder=zorder,
                label=extrapolation_label)
    ax.plot(abscissa[inside], discharge[inside], color=color, lw=linewidth,
            ls=linestyle, zorder=zorder + 1, label=label)
    return ax


def plot_site(site, ax=None, *, level: float | None = None):
    """One figure for a site: its measurements, every fitted curve, the reference.

    Parameters
    ----------
    site : SiteRating
        The site and its fits.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    level : float, optional
        Draw each curve's credible band at this level. ``None`` draws curves only,
        which is what keeps a multi-model figure readable.

    Returns
    -------
    matplotlib.axes.Axes or None
        None when the site has no measurements.
    """
    import matplotlib.pyplot as plt
    sample = site.sample
    if len(sample) == 0:
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 6))

    ax.scatter(sample.stage_ft, sample.discharge_cfs, marker="o", s=55,
               facecolor="white", edgecolor="black", linewidth=1.3, zorder=5,
               label="measurements")

    measured_range = sample.stage_range
    extrapolated = False
    for fit in site.fits:
        if not fit.ok or fit.curve is None:
            continue
        scores = fit.metrics
        draw_curve(ax, fit.curve, measured_range, color=catalog.color(fit.key),
                   level=level, band_alpha=0.12, zorder=2,
                   linestyle=_MATPLOTLIB_DASH.get(catalog.dash(fit.key), "-"),
                   extrapolation_color=None,
                   label=(f"{fit.label} [{_config_summary(fit.config)}]  "
                          f"NSE={scores.get('nse', float('nan')):.2f}, "
                          f"R²log={scores.get('r2_log', float('nan')):.2f}"))
        extrapolated = extrapolated or bool(
            (fit.curve["stage_ft"] < measured_range[0]).any()
            or (fit.curve["stage_ft"] > measured_range[1]).any())
    if extrapolated:
        ax.plot([], [], color="0.4", ls=":", lw=2.2, label=EXTRAPOLATION_LABEL)

    _overlay_external(ax, site)

    ax.set_yscale("log")
    limit_discharge_axis(ax, sample.discharge_cfs)
    ax.set_xlabel(f"stage - {site.stage_label}")
    ax.set_ylabel("discharge (cfs)")
    ax.set_title(f"{site.label}  (n={len(sample)})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small", loc="lower right")
    return ax


def _overlay_external(ax, site):
    """Draw every curve at a site that this package did not fit.

    A USGS published rating and a field spreadsheet's typed equation both land here
    and are drawn differently, because they are different claims - see
    :class:`limnotech_rating_curves.core.ExternalCurve`. Styles come from the same
    table the interactive map uses, so a figure and the map cannot disagree.

    Clipping matters: a published rating spans the gage's full historic range, and
    drawing all of it on a log axis compresses the part the measurements actually
    cover into nothing.
    """
    from . import mapview

    low, high = site.sample.stage_range
    pad = 0.1 * ((high - low) or 1.0)
    for external in site.external_curves():
        curve = external.curve.sort_values("stage_ft")
        curve = curve[(curve["stage_ft"] >= low - pad)
                      & (curve["stage_ft"] <= high + pad)]
        if curve.empty:
            continue
        style = mapview._external_style(external)
        scores = external.metrics
        ax.plot(curve["stage_ft"], curve["discharge_cfs"], zorder=4,
                color=style["color"], linewidth=style["width"],
                linestyle=_MATPLOTLIB_DASH.get(style["dash"], "--"),
                label=(f"{external.label}  "
                       f"NSE={scores.get('nse', float('nan')):.2f}, "
                       f"R²log={scores.get('r2_log', float('nan')):.2f}"))


def _config_summary(config: dict) -> str:
    """A one-token summary of a model's configuration, for a legend entry."""
    if "segments" in config:
        return f"{config['segments']} seg"
    if "df" in config:
        return f"{config['df']} knots"
    if "variant" in config:
        return str(config["variant"])
    return ""


def save_site_figure(site, directory) -> "str | None":
    """Write a site's figure to ``<directory>/<source>/<sample id>.png``.

    Parameters
    ----------
    site : SiteRating
        The site and its fits.
    directory : path-like
        Figure root; a per-source subdirectory is created inside it.

    Returns
    -------
    str or None
        The path written, or None when the site has no measurements.
    """
    import matplotlib.pyplot as plt
    ax = plot_site(site)
    if ax is None:
        return None
    directory = Path(directory) / (site.source or "sites")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(site.sample_id)}.png"
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=140)
    plt.close(ax.figure)
    return str(path)


def plot_fold(fold, ax=None, *, stage_label: str = "stage (ft)", title: str = ""):
    """One cross-validation fold: its curve, its band, and what it did and did not see.

    The training points are drawn solid and the held-out points hollow, which is the
    whole point of the figure - you can see immediately whether the curve went
    through measurements it was shown or measurements it was not.

    Parameters
    ----------
    fold : FoldCurve
        The fold.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    stage_label : str
        Description of the stage axis.
    title : str
        Extra title text, prepended to the fold description.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))
    color = catalog.color(fold.model_key)

    stage = np.asarray(fold.stage_ft, float)
    lower = np.asarray(fold.lower_cfs, float)
    upper = np.asarray(fold.upper_cfs, float)
    drawable = np.isfinite(lower) & np.isfinite(upper) & (lower > 0)
    if drawable.any():
        ax.fill_between(stage[drawable], lower[drawable], upper[drawable], color=color,
                        alpha=0.18, zorder=2, label="95% credible band")
    ax.plot(stage, fold.discharge_median_cfs, color=color, lw=2.6, zorder=4,
            label="posterior median")
    ax.scatter(fold.test_stage, fold.test_discharge, s=45, facecolor="none",
               edgecolor="0.7", linewidth=1.2, zorder=5, label="held out")
    ax.scatter(fold.train_stage, fold.train_discharge, s=55, facecolor="white",
               edgecolor="black", linewidth=1.4, zorder=6,
               label=f"trained on (n={len(fold.train_stage)})")

    nse = fold.test_metrics.get("nse")
    nse_text = (f", held-out NSE {nse:.2f}"
                if nse is not None and np.isfinite(nse) else "")
    ax.set_yscale("log")
    ax.set_xlabel(f"stage - {stage_label}")
    ax.set_ylabel("discharge (cfs)")
    ax.set_title(f"{title}\n{catalog.label(fold.model_key)} - fold {fold.split}"
                 f" ({len(fold.train_stage)} train{nse_text})".strip())
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small", loc="lower right")
    return ax


def save_fold_figures(site, folds, directory) -> list:
    """Write one figure per fold, grouped by model.

    Parameters
    ----------
    site : SiteRating
        The site the folds belong to (for the title and stage axis).
    folds : sequence of FoldCurve
        The folds to draw.
    directory : path-like
        Figure root; a ``folds/<sample id>__<model>`` subdirectory is created.

    Returns
    -------
    list of str
        The paths written.
    """
    import matplotlib.pyplot as plt
    if not folds:
        return []
    root = Path(directory) / "folds"
    written = []
    for fold in folds:
        ax = plot_fold(fold, stage_label=site.stage_label or "stage (ft)",
                       title=site.label)
        model_directory = root / f"{_safe_name(site.sample_id)}__{fold.model_key}"
        model_directory.mkdir(parents=True, exist_ok=True)
        path = model_directory / f"fold_{fold.split:02d}.png"
        ax.figure.tight_layout()
        ax.figure.savefig(path, dpi=140)
        plt.close(ax.figure)
        written.append(str(path))
    return written


def plot_high_flow_error(cross_validation, ax=None):
    """Predicted over observed discharge at the held-out points, against stage.

    The figure to read when the question is extrapolation. A perfect model sits on
    the 1.0 line at every stage; a model that fits the middle and fails high shows as
    points drifting off the line toward the right of the plot, which no aggregate
    score makes visible.

    Parameters
    ----------
    cross_validation : CrossValidation
        A completed sweep.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 5.5))
    drawn = set()
    for fold in cross_validation.curves:
        observed = np.asarray(fold.test_discharge, float)
        predicted = fold.predict_test()
        stage = np.asarray(fold.test_stage, float)
        usable = np.isfinite(observed) & np.isfinite(predicted) & (observed > 0)
        if not usable.any():
            continue
        label = (None if fold.model_key in drawn
                 else catalog.label(fold.model_key))
        drawn.add(fold.model_key)
        ax.scatter(stage[usable], predicted[usable] / observed[usable],
                   color=catalog.color(fold.model_key), s=32, alpha=0.75,
                   edgecolor="none", label=label)
    ax.axhline(1.0, color="black", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("stage of the held-out measurement (ft)")
    ax.set_ylabel("predicted / observed discharge")
    ax.set_title("held-out error against stage - where does each model go wrong?")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small")
    return ax


def plot_log_log(rating, ax=None, *, zero_flow=None, level: float = 0.95):
    """A fitted rating on log-log axes, against effective head.

    A power law is a straight line in these coordinates, so this is the figure that
    shows whether the form fits: curvature here is the model bending to reach points
    a power law cannot.

    Parameters
    ----------
    rating : RatingModel
        A fitted model.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    zero_flow : float, optional
        Stage of zero flow to subtract, so the x axis is effective head. Defaults to
        the model's own fitted value (``rating.zero_flow``), and to 0 for models that
        do not have one.
    level : float, default 0.95
        Width of the credible band drawn.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    if zero_flow is None:
        zero_flow = getattr(rating, "zero_flow", np.nan)
    zero_flow = 0.0 if not np.isfinite(zero_flow) else float(zero_flow)

    sample = rating.sample
    curve = rating.curve(level=level).assign(
        head_ft=lambda frame: frame["stage_ft"] - zero_flow)
    curve = curve[(curve["head_ft"] > 0) & (curve["discharge_cfs"] > 0)]
    measured_head = tuple(edge - zero_flow for edge in sample.stage_range)
    draw_curve(ax, curve, measured_head, x="head_ft", color=rating.color,
               level=level, band_alpha=0.2, linewidth=2.0, label=rating.label,
               extrapolation_label=EXTRAPOLATION_LABEL)

    ax.plot(sample.stage_ft - zero_flow, sample.discharge_cfs, "o", color="black",
            ms=5, label="measurements")
    ax.set(xscale="log", yscale="log",
           xlabel=(f"effective head, stage - {zero_flow:g} ft (ft)" if zero_flow
                   else f"stage - {sample.stage_label}"),
           ylabel="discharge (cfs)")
    limit_discharge_axis(ax, sample.discharge_cfs)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small")
    return ax


def plot_residual_ratio(rating, ax=None):
    """Predicted over observed discharge at every measurement, against stage.

    The companion to a curve plot. Overlaid curves hide a bias at high flow, because
    the eye cannot read a factor of two off a log axis at the top of the range; as a
    ratio a factor of two reads the same at any flow, and 1.0 is right.

    Parameters
    ----------
    rating : RatingModel or RatingSet
        One fitted model, or several. A ``RatingSet`` draws one series per model, plus
        its reference curve when it has one.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    models = list(getattr(rating, "models", [rating]))
    sample = rating.sample
    observed = np.asarray(sample.discharge_cfs, float)
    stage = np.asarray(sample.stage_ft, float)

    ax.axhline(1.0, color="0.4", lw=1)
    marker = "o-" if len(models) > 1 else "o"
    for model in models:
        ax.plot(stage, model.predict(stage) / observed, marker, ms=5, lw=0.8,
                alpha=0.8, color=model.color, label=model.label)

    reference = getattr(rating, "reference", None)
    if reference:
        curve = reference["curve"].sort_values("stage_ft")
        predicted = np.interp(stage, curve["stage_ft"], curve["discharge_cfs"],
                              left=np.nan, right=np.nan)
        ax.plot(stage, predicted / observed, "kx", ms=7, label=reference["label"])

    ax.set(xlabel=f"stage - {sample.stage_label}", ylabel="predicted / observed")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")
    return ax


def plot_fit_check(rating, *, zero_flow=None, level: float = 0.95, figsize=(13, 4.5)):
    """The two panels a rating is judged on: its shape, and its residuals.

    Left, the curve on log-log axes, where a power law is a straight line. Right, the
    residuals as a ratio, which says whether the error is the same size at low flow as
    at high. Reading one without the other is how a bad rating gets accepted.

    Parameters
    ----------
    rating : RatingModel
        A fitted model.
    zero_flow : float, optional
        Passed to :func:`plot_log_log`.
    level : float, default 0.95
        Width of the credible band drawn.
    figsize : tuple, default (13, 4.5)
        Figure size.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The two axes, left to right.
    """
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    plot_log_log(rating, axes[0], zero_flow=zero_flow, level=level)
    axes[0].set_title("a power law is a line in log-log space")
    plot_residual_ratio(rating, axes[1])
    axes[1].set_title("residuals as a ratio, across the stage range")
    figure.tight_layout()
    return axes


def plot_ranking(rating_set, ax=None):
    """Model ranking by predictive score, with the uncertainty on each difference.

    The form a ranking decision is actually made from: anything whose error bar
    reaches into the two-standard-error band is not distinguishable from the winner by
    this sample, so take the simpler model.

    Parameters
    ----------
    rating_set : RatingSet
        Several fitted models.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.

    Returns
    -------
    matplotlib.axes.Axes or None
        None when no model produced a predictive score - an ADVI comparison, for
        instance, where there is nothing to rank.
    """
    import matplotlib.pyplot as plt
    ranking = rating_set.ranking()
    ranking = ranking[np.isfinite(ranking["elpd_diff"])].iloc[::-1]
    if ranking.empty:
        log.warning("no model has a predictive score to rank; nothing to draw")
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.7 * len(ranking) + 2))

    positions = np.arange(len(ranking))
    ax.axvspan(-2 * ranking["dse"].max(), 0, color="#4c9f70", alpha=0.12,
               label="within 2 standard errors of the best")
    ax.errorbar(ranking["elpd_diff"], positions, xerr=ranking["dse"], fmt="o",
                color="#4c72b0", ecolor="0.4", capsize=4, ms=7)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(ranking["model"])
    ax.set(xlabel="elpd_diff: predictive score minus the best model's (0 is the winner)",
           title="model ranking, with the uncertainty on the difference")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize="small", loc="lower left")
    return ax


def plot_record(sample, series, *, discharge=None, figsize=(12, 7)):
    """A continuous stage record with the measurements a rating was fitted to on it.

    The record is what a rating has to speak for, and a handful of points is what it
    is built from; drawing them together is the only way to see whether those points
    cover the record or sit in one corner of it.

    Parameters
    ----------
    sample : Sample
        The measurements. Drawn on the record's own axis, by adding back
        ``datum_reference_ft``, so points and record line up whatever datum the
        sample was converted to.
    series : pandas.Series
        The continuous record, indexed by time.
    discharge : pandas.DataFrame, optional
        All the discharge measurements, with ``time`` and ``discharge_cfs`` columns,
        for the lower panel. Defaults to the sample's own, which is the subset that
        was successfully matched to a stage reading.
    figsize : tuple, default (12, 7)
        Figure size.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The two axes, record above and discharge below.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(series.index, series.to_numpy(), lw=0.7, color="#4c72b0")

    reference = sample.datum_reference_ft
    on_record = sample.stage_ft + (reference if np.isfinite(reference) else 0.0)
    if sample.time is not None:
        axes[0].plot(sample.time, on_record, "o", color="#c44e52", ms=8, zorder=5,
                     label="matched to a discharge measurement")
        axes[0].legend(fontsize="small")
    axes[0].set(ylabel="stage as reported (ft)",
                title=f"{sample.site_id or 'station'}: the stage record")
    axes[0].grid(alpha=0.25)

    if discharge is None:
        discharge = pd.DataFrame({"time": sample.time,
                                  "discharge_cfs": sample.discharge_cfs})
    axes[1].vlines(discharge["time"], 0, discharge["discharge_cfs"], color="0.6", lw=1)
    axes[1].plot(discharge["time"], discharge["discharge_cfs"], "o", color="black", ms=6)
    axes[1].set(yscale="log", ylabel="discharge (cfs)", xlabel="",
                title=f"the {len(discharge)} field discharge measurements")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    return axes


def plot_rating_cloud(record, ax=None, *, rating=None, measurements=None,
                      color_by: str = "year", sample_size: int = 20000, seed: int = 0):
    """Every paired reading in a continuous record, as a stage-discharge cloud.

    The figure that answers "is this one rating or several?". A gage operated under
    a single rating draws one narrow curve; a rating that was rebuilt draws separate
    strands, and colouring by year says which strand belongs to when.

    Parameters
    ----------
    record : pandas.DataFrame
        Paired readings with ``stage_ft``, ``discharge_cfs`` and the colouring
        column - what
        :func:`limnotech_rating_curves.data.usgs.continuous_record` returns.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    rating : pandas.DataFrame, optional
        A published rating to overlay, with ``stage_ft`` and ``discharge_cfs``.
    measurements : pandas.DataFrame, optional
        Field measurements to overlay, same columns.
    color_by : str, default 'year'
        Column of `record` to colour the points by.
    sample_size : int, default 20000
        Draw at most this many points, taken at random. A million overplotted
        markers hide the very structure the figure is for.
    seed : int, default 0
        Seed for that draw, so the figure is reproducible.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))
    drawn = (record.sample(sample_size, random_state=seed)
             if len(record) > sample_size else record)
    scatter = ax.scatter(drawn["stage_ft"], drawn["discharge_cfs"], s=4, alpha=0.35,
                         c=drawn[color_by], cmap="viridis", linewidths=0, zorder=2)
    ax.figure.colorbar(scatter, ax=ax, label=color_by)

    if rating is not None and not rating.empty:
        ax.plot(rating["stage_ft"], rating["discharge_cfs"], **REFERENCE_STYLE,
                zorder=4, label="published rating")
    if measurements is not None and not measurements.empty:
        ax.scatter(measurements["stage_ft"], measurements["discharge_cfs"], s=45,
                   facecolor="white", edgecolor="#c44e52", linewidth=1.4, zorder=5,
                   label="field measurements")
    ax.set(yscale="log", xlabel="gage height (ft)", ylabel="discharge (cfs)",
           title=f"{len(record):,} paired readings from the continuous record")
    ax.grid(True, which="both", alpha=0.25)
    if rating is not None or measurements is not None:
        ax.legend(fontsize="small", loc="lower right")
    return ax


def plot_rating_deviation(deviation, axes=None, *, figsize=(12, 8)):
    """How far each field measurement sits from the published rating, over time.

    Three panels, all reading the same disagreement: the percent difference against
    time, which is where a rating drifting under the gage shows up as a trend; the
    same against stage, which separates a rating that is wrong everywhere from one
    that is wrong only at the low end; and the implied stage shift against time,
    which is the form the USGS keeps it in.

    Parameters
    ----------
    deviation : pandas.DataFrame
        What :func:`limnotech_rating_curves.data.usgs.rating_deviation` returns.
    axes : sequence of matplotlib.axes.Axes, optional
        Three axes to draw into.
    figsize : tuple, default (12, 8)
        Figure size.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=figsize)
    usable = deviation.dropna(subset=["percent_diff"])

    axes[0].axhline(0, color="0.4", lw=1)
    axes[0].axhspan(-5, 5, color="#4c9f70", alpha=0.15,
                    label="within 5% of the rating")
    axes[0].scatter(usable["time"], usable["percent_diff"], s=28, color="#4c72b0")
    axes[0].set(ylabel="percent difference", title="measured discharge against the "
                "published rating, over time")
    axes[0].legend(fontsize="small")

    axes[1].axhline(0, color="0.4", lw=1)
    axes[1].scatter(usable["stage_ft"], usable["percent_diff"], s=28, color="#4c72b0")
    axes[1].set(xlabel="gage height (ft)", ylabel="percent difference",
                title="the same difference against stage")

    shifted = deviation.dropna(subset=["shift_ft"])
    axes[2].axhline(0, color="0.4", lw=1)
    axes[2].scatter(shifted["time"], shifted["shift_ft"], s=28, color="#c44e52")
    axes[2].set(xlabel="", ylabel="implied shift (ft)",
                title="the stage shift that would put the rating through each "
                      "measurement")

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].figure.tight_layout()
    return axes


FIGURES_README = """\
# Rating-curve figures

The primary output of a batch run is the interactive map
(`output/rating_curves_map.html`); these static PNGs are written with `--png`.

## Layout

- `<source>/<sample id>.png` - one figure per site: the measurements as open
  markers, every fitted rating curve overlaid, and the published reference rating
  (black dashed) where the site has one. The legend reports in-sample NSE and
  R²(log) per curve.
- `folds/<sample id>__<model>/fold_NN.png` - with `--cv --png`, one figure per model
  per cross-validation fold: that fold's median rating and 95% credible band, the
  measurements it trained on drawn solid and the ones held out from it hollow.

Discharge is on a log axis in every figure; the x-axis is the sample's own stage
axis, named in the axis label. Curve colors come from `catalog.py` - the same
colors the map uses.

## Why a model may be missing from a figure

Model complexity follows sample size: the segmented power laws and the spline are
skipped on samples too small to identify them (the per-model minimum is in
`catalog.py`). A model that was attempted and failed is likewise absent, with
its reason recorded in `results.csv`.

## The published reference (black dashed)

Sites with a USGS gage carry that gage's own published rating, clipped to the
observed stage range and scored on the same measurements. It is a maintained
product, not data - the reference to compare against, never something to fit to.
"""


def write_figures_readme(directory) -> str:
    """Write the README that explains the figure layout.

    Parameters
    ----------
    directory : path-like
        Figure root.

    Returns
    -------
    str
        The path written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    path.write_text(FIGURES_README, encoding="utf-8")
    return str(path)
