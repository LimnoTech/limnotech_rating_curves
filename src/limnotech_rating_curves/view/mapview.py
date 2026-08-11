import json
import logging
from pathlib import Path

import numpy as np

from ..evaluate import metrics as metrics_module
from ..models import catalog
from .. import settings

log = logging.getLogger(__name__)

#: Esri topographic basemap with a hydrography overlay. Raster tiles, so the map
#: needs no map-provider token.
BASEMAP_LAYERS = [
    {"below": "traces", "sourcetype": "raster", "type": "raster",
     "sourceattribution": "Esri, USGS, NOAA - World Topographic Map",
     "source": ["https://services.arcgisonline.com/ArcGIS/rest/services/"
                "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"]},
    {"below": "traces", "sourcetype": "raster", "type": "raster",
     "sourceattribution": "Esri - World Hydro Reference Overlay",
     "source": ["https://services.arcgisonline.com/ArcGIS/rest/services/Reference/"
                "World_Hydro_Reference_Overlay/MapServer/tile/{z}/{y}/{x}"]},
]

#: How each sample source is drawn: (legend name, marker color, longitude jitter).
#: The jitter keeps co-located sites individually hoverable.
SOURCE_STYLE = {
    "usgs_gage": ("USGS discharge + USGS stage", "#08519c", 0.0),
    "colocated": ("USGS discharge + MAGL stage (co-located)", "#e6550d", 0.0007),
    "magl": ("MAGL discharge + MAGL stage", "#31a354", -0.0007),
    "pagaia": ("pagaia station", "#756bb1", 0.0014),
}
_FALLBACK_STYLE = ("sites", "#7f7f7f", 0.0)

#: How each kind of external curve is drawn. The two are deliberately far apart: one
#: is a federal agency's published rating and the other is an equation somebody typed
#: into a spreadsheet, and a reader must never have to guess which is which.
EXTERNAL_STYLE = {
    "published_reference": {"color": "#000000", "dash": "dash", "width": 2.5,
                            "legend": "USGS published rating"},
    "spreadsheet": {"color": "#e7298a", "dash": "longdashdot", "width": 3.2,
                    "legend": "the field spreadsheet's own equation"},
}
_FALLBACK_EXTERNAL = {"color": "#404040", "dash": "dot", "width": 2.5,
                      "legend": "external curve"}

#: Opacity of a credible / prediction band's fill.
BAND_ALPHA = 0.16

#: Figure height in pixels. Fixed, because the panel geometry below is in paper
#: coordinates and a button row's height in paper units depends on it - a taller
#: figure would leave gaps and a shorter one would make the control rows collide.
FIGURE_HEIGHT = 900

# --- right-hand panel geometry -----------------------------------------------
# One place, so the collision check and the layout read the same numbers.

#: Left edge of everything in the inspector, and of the map's right edge.
PANEL_X = [0.63, 1.0]
MAP_DOMAIN_X = [0.0, 0.575]

#: The rating plot.
PLOT_DOMAIN_Y = [0.56, 0.94]

#: The control row, present only with cross-validation: the model selector, then the
#: fold arrows under it. Both are anchored at the top and grow downward.
MODEL_ROW_Y = 0.530
FOLD_ROW_Y = 0.480

#: A button row's height, in paper units at ``FIGURE_HEIGHT``. Used only by the
#: collision check, which has to know how far a top-anchored row reaches down.
BUTTON_ROW_HEIGHT = 34.0 / FIGURE_HEIGHT

#: The score table. Two variants, because without the control row there is space for
#: more rows and leaving it empty would be waste.
TABLE_DOMAIN_Y_WITH_CV = [0.02, 0.43]
TABLE_DOMAIN_Y_NO_CV = [0.02, 0.51]

#: Column widths of the score table, as fractions of the panel. The first column is
#: the curve's name and needs the room.
_FIT_TABLE_WIDTHS = (0.30, 0.055, 0.075, 0.085, 0.08, 0.11, 0.07, 0.07, 0.075)
_FIT_TABLE_HEADERS = ("curve", "n", "NSE", "RMSE", "R²log", "ELPD_LOO", "se",
                      "p_loo", "k_max")
#: Which glossary entry each header explains.
_FIT_TABLE_GLOSSARY = ("curve", "n", "nse", "rmse", "r2_log", "elpd_loo", "se_loo",
                       "p_loo", "pareto_k_max")

_CV_TABLE_WIDTHS = (0.26, 0.10, 0.11, 0.11, 0.11, 0.11, 0.20)
_CV_TABLE_HEADERS = ("model", "train/test", "fold NSE", "fold R²log", "all folds NSE",
                     "all folds R²log", "ELPD_LOO (all data)")


def _style_for(source):
    """Legend name, color and jitter for a sample source."""
    return SOURCE_STYLE.get(source, _FALLBACK_STYLE)


def _external_style(external) -> dict:
    """Line style for an external curve: its own, falling back to its kind's."""
    style = dict(EXTERNAL_STYLE.get(external.kind, _FALLBACK_EXTERNAL))
    if external.color:
        style["color"] = external.color
    if external.dash:
        style["dash"] = external.dash
    return style


def _drawable(fit) -> bool:
    """True when a fit produced a curve worth drawing."""
    return fit is not None and fit.ok and fit.curve is not None


def _has_curves(site) -> bool:
    """True when a site can appear on the map: it has coordinates and a curve."""
    return bool(site.coords) and any(_drawable(fit) for fit in site.fits)


def _headline_fit(site):
    """The fit a map marker's tooltip quotes.

    Ranked on the **predictive** score, not on in-sample NSE. Quoting the best NSE
    told the reader which curve passed closest to the measurements, which the most
    flexible model always wins and which is not the question a map of many sites is
    asked. ``r2_log`` breaks ties and stands in where no ELPD exists (the
    least-squares forms have none at all).
    """
    drawable = [fit for fit in site.fits if _drawable(fit)]
    if not drawable:
        return None

    def rank(fit):
        bayes = fit.bayes or {}
        elpd = bayes.get("elpd_loo", np.nan)
        elpd = elpd if elpd is not None and np.isfinite(elpd) else -np.inf
        r2 = fit.metrics.get("r2_log", np.nan)
        r2 = r2 if r2 is not None and np.isfinite(r2) else -np.inf
        return (elpd, r2)

    return max(drawable, key=rank)


def _format(value, spec: str = "{:.2f}") -> str:
    """Format a number for a table cell, blanking anything not finite."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """A hex color as an rgba string with the given alpha."""
    digits = str(hex_color).lstrip("#")
    if len(digits) != 6:
        return f"rgba(120,120,120,{alpha})"
    red, green, blue = (int(digits[position:position + 2], 16)
                        for position in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _draw_window(site) -> tuple:
    """The one stage window every curve at this site is clipped to.

    Model curves are evaluated on a grid padded past the measurements; a published or
    typed curve is defined everywhere. Drawing them as they come left the lines
    ending at different places, which reads as a difference between the curves rather
    than a difference between how they were tabulated. So one window, applied to all
    of them.
    """
    low, high = site.sample.stage_range
    if not np.isfinite(low) or not np.isfinite(high):
        return (-np.inf, np.inf)
    pad = settings.GRID_PAD_FRACTION * ((high - low) or 1.0)
    return (low - pad, high + pad)


def _clip(curve, window):
    """A curve restricted to the shared draw window, sorted by stage."""
    curve = curve.sort_values("stage_ft")
    low, high = window
    return curve[(curve["stage_ft"] >= low) & (curve["stage_ft"] <= high)]


# --- fitted-curves pane -------------------------------------------------------

def _curve_trace(stage, discharge, color, name, sample_id, *, dash="solid",
                 width=2.5, hovertemplate=None, pane="fit", legendkey=None,
                 **tag_extra):
    """One rating-curve line on the shared inspector axes, hidden until selected."""
    import plotly.graph_objects as go
    tag = {"sample": sample_id, "pane": pane, "kind": "curve"}
    if legendkey:
        tag["legendkey"] = legendkey
    tag.update(tag_extra)
    return go.Scatter(
        x=stage, y=discharge, mode="lines", xaxis="x2", yaxis="y2",
        line=dict(width=width, color=color, dash=dash), name=name, meta=tag,
        visible=False, showlegend=False,
        legendgroup=legendkey or name,
        hovertemplate=hovertemplate or (f"<b>{name}</b><br>stage %{{x:.2f}} ft<br>"
                                        f"Q %{{y:.3g}} cfs<extra></extra>"))


def _band_trace(stage, lower, upper, color, sample_id, *, pane="fit", legendkey=None,
                label="95% band", **tag_extra):
    """A filled uncertainty band, tied to its line by legend group.

    Same ``legendgroup`` as the curve it belongs to, so one legend click hides both -
    a line without its band, or a band without its line, is worse than neither.
    """
    import plotly.graph_objects as go
    stage = np.asarray(stage, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    drawable = np.isfinite(lower) & np.isfinite(upper) & (lower > 0) & (upper > 0)
    if drawable.sum() < 2:
        return None
    tag = {"sample": sample_id, "pane": pane, "kind": "band"}
    if legendkey:
        tag["legendkey"] = legendkey
    tag.update(tag_extra)
    return go.Scatter(
        x=np.concatenate([stage[drawable], stage[drawable][::-1]]),
        y=np.concatenate([upper[drawable], lower[drawable][::-1]]),
        mode="lines", xaxis="x2", yaxis="y2", fill="toself",
        fillcolor=_hex_to_rgba(color, BAND_ALPHA), line=dict(width=0), name=label,
        visible=False, showlegend=False, legendgroup=legendkey or label,
        hoverinfo="skip", meta=tag)


def _measurement_trace(site):
    """The site's measurements, with each model's Pareto-k in the point tooltip.

    Pareto-k is per measurement *and* per model, so the tooltip lists one line per
    model. That is what makes the influence of an individual measurement visible at
    the point rather than only as a summary in the table.
    """
    import plotly.graph_objects as go
    sample = site.sample
    per_model = {}
    for fit in site.fits:
        if fit.ok and fit.bayes:
            per_model[catalog.short_label(fit.key)] = \
                metrics_module.pareto_k_per_point(fit)

    customdata = []
    threshold = settings.PARETO_K_GOOD
    for index in range(len(sample)):
        parts = []
        for name, values in per_model.items():
            value = values[index] if index < len(values) else np.nan
            if np.isfinite(value):
                flag = " ⚠" if value > threshold else ""
                parts.append(f"{name} k={value:.2f}{flag}")
        customdata.append(["<br>".join(parts) if parts else "no Pareto-k available"])

    return go.Scatter(
        x=sample.stage_ft, y=sample.discharge_cfs, mode="markers", xaxis="x2",
        yaxis="y2",
        marker=dict(size=9, color="black", symbol="circle-open", line=dict(width=2)),
        name="measurements", meta={"sample": site.sample_id, "pane": "both",
                                   "kind": "measurements",
                                   "legendkey": "measurements"},
        visible=False, showlegend=False, legendgroup="measurements",
        customdata=customdata,
        hovertemplate=("<b>measurement</b><br>stage %{x:.2f} ft<br>Q %{y:.3g} cfs"
                       "<br><i>Pareto-k (influence on each fit)</i><br>"
                       "%{customdata[0]}<extra></extra>"))


def _fit_pane_traces(site):
    """The fitted-curves pane for one site: every curve, and every curve's band."""
    sample_id = site.sample_id
    window = _draw_window(site)
    traces = [_measurement_trace(site)]

    for fit in site.fits:
        if not _drawable(fit):
            continue
        curve = _clip(fit.curve, window)
        if curve.empty:
            continue
        color = catalog.color(fit.key)
        scores = fit.metrics
        bayes = fit.bayes or {}
        # the band first, so the line draws over its own fill
        if {"lower", "upper"} <= set(curve.columns):
            band = _band_trace(curve["stage_ft"], curve["lower"], curve["upper"],
                               color, sample_id, legendkey=fit.key, model=fit.key,
                               label=f"{fit.label} - 95% band")
            if band is not None:
                traces.append(band)
        traces.append(_curve_trace(
            curve["stage_ft"], curve["discharge_cfs"], color, fit.label, sample_id,
            dash=catalog.dash(fit.key), legendkey=fit.key, model=fit.key,
            hovertemplate=(f"<b>{fit.label}</b><br>stage %{{x:.2f}} ft<br>"
                           f"Q %{{y:.3g}} cfs<br>"
                           f"NSE={_format(scores.get('nse'))} "
                           f"R²log={_format(scores.get('r2_log'))}<br>"
                           f"ELPD_LOO={_format(bayes.get('elpd_loo'), '{:.1f}')}"
                           f"<extra></extra>")))

    for external in site.external_curves():
        curve = _clip(external.curve, window)
        if curve.empty:
            continue
        style = _external_style(external)
        scores = external.metrics
        if {"lower", "upper"} <= set(curve.columns):
            band = _band_trace(curve["stage_ft"], curve["lower"], curve["upper"],
                               style["color"], sample_id, legendkey=external.key,
                               model=external.key,
                               label=f"{external.label} - band")
            if band is not None:
                traces.append(band)
        detail = ""
        if external.detail.get("form"):
            detail += f"<br>declared form: {external.detail['form']}"
        if external.detail.get("equation"):
            detail += f"<br>{external.detail['equation']}"
        traces.append(_curve_trace(
            curve["stage_ft"], curve["discharge_cfs"], style["color"], external.label,
            sample_id, dash=style["dash"], width=style["width"],
            legendkey=external.key, model=external.key,
            hovertemplate=(f"<b>{external.label}</b><br>"
                           f"<i>{style['legend']} - not fitted here</i>"
                           f"<br>stage %{{x:.2f}} ft<br>Q %{{y:.3g}} cfs<br>"
                           f"NSE={_format(scores.get('nse'))} "
                           f"R²log={_format(scores.get('r2_log'))}{detail}"
                           f"<extra></extra>")))
    return traces


# --- the score tables, as hoverable grids -------------------------------------
# Plotly's go.Table cannot carry a tooltip on a cell or a header, and a metrics table
# nobody can interrogate is the reason METRIC_GLOSSARY sat unused. So the tables here
# are drawn as text on their own axis pair, with a transparent marker behind every
# cell to hover. That costs a little arithmetic and buys a definition on every column.

def _column_positions(widths) -> list:
    """Left edge of each column, in axis-fraction coordinates."""
    total = float(sum(widths))
    edges, running = [], 0.0
    for width in widths:
        edges.append(running / total)
        running += width
    return edges


def _grid_traces(sample_id, pane, kind, headers, glossary_keys, widths, rows,
                 footer=(), *, tag_extra=None):
    """A text grid with a hover definition on every header and cell.

    Parameters
    ----------
    sample_id, pane, kind : str
        Trace tags, as everywhere else in this module.
    headers : sequence of str
        Column headers as displayed.
    glossary_keys : sequence of str or None
        Which :data:`METRIC_GLOSSARY` entry explains each column. ``None`` for a
        column with no entry.
    widths : sequence of float
        Relative column widths.
    rows : sequence of sequence of str
        Formatted cell text, one inner sequence per row.
    footer : sequence of str, optional
        Lines drawn under the table, spanning its whole width. For the sentence that
        says what the numbers on screen actually are.
    tag_extra : dict, optional
        Extra trace tags (``model``, ``fold``).

    Returns
    -------
    list of plotly.graph_objects.Scatter
        Three traces: the header text, the body text, and the transparent hover
        markers.
    """
    import plotly.graph_objects as go

    tag = {"sample": sample_id, "pane": pane, "kind": kind}
    tag.update(tag_extra or {})
    edges = _column_positions(widths)
    # rows are laid out top-down; leave room for the footer lines
    line_count = 1 + len(rows) + (1 if footer else 0) + len(footer)
    step = 1.0 / max(line_count + 0.5, 6.0)
    top = 1.0 - 0.4 * step

    header_x, header_y, header_text, header_hover = [], [], [], []
    for column, (name, key) in enumerate(zip(headers, glossary_keys)):
        header_x.append(edges[column])
        header_y.append(top)
        header_text.append(f"<b>{name}</b>")
        header_hover.append(metrics_module.metric_hover(key, name) if key else "")

    body_x, body_y, body_text, body_hover = [], [], [], []
    for index, cells in enumerate(rows):
        y = top - (index + 1) * step
        for column, value in enumerate(cells):
            body_x.append(edges[column])
            body_y.append(y)
            body_text.append(str(value))
            definition = (metrics_module.metric_hover(glossary_keys[column],
                                                      headers[column])
                          if column < len(glossary_keys) and glossary_keys[column]
                          else "")
            label = cells[0] if cells else ""
            body_hover.append(
                f"<b>{label}</b><br>{headers[column]} = {value}"
                + (f"<br><br>{definition}" if definition else ""))

    footer_x, footer_y, footer_text = [], [], []
    for index, line in enumerate(footer):
        footer_x.append(0.0)
        footer_y.append(top - (len(rows) + 1.6 + index) * step)
        footer_text.append(line)

    common = dict(xaxis="x3", yaxis="y3", visible=False, showlegend=False,
                  cliponaxis=False)
    text_trace = go.Scatter(
        x=header_x + body_x + footer_x, y=header_y + body_y + footer_y,
        mode="text", text=header_text + body_text + footer_text,
        textposition="middle right", textfont=dict(size=11, family="monospace"),
        hoverinfo="skip", meta={**tag, "part": "text"}, **common)
    footer_trace = None
    if footer_text:
        # the footer is prose, so it gets a proportional face and a lighter colour
        text_trace.text = header_text + body_text + ["" for _ in footer_text]
        footer_trace = go.Scatter(
            x=footer_x, y=footer_y, mode="text", text=footer_text,
            textposition="middle right",
            textfont=dict(size=10, color="#555555"), hoverinfo="skip",
            meta={**tag, "part": "footer"}, **common)
    hover_trace = go.Scatter(
        x=header_x + body_x, y=header_y + body_y, mode="markers",
        marker=dict(size=16, color="rgba(0,0,0,0)"),
        customdata=[[text] for text in header_hover + body_hover],
        hovertemplate="%{customdata[0]}<extra></extra>",
        hoverlabel=dict(align="left", bgcolor="rgba(255,255,255,0.96)",
                        font=dict(size=11)),
        meta={**tag, "part": "hover"}, **common)
    return [trace for trace in (text_trace, footer_trace, hover_trace)
            if trace is not None]


def _fit_table_traces(site, rows):
    """The per-curve score table for the fitted-curves pane."""
    threshold = settings.PARETO_K_GOOD
    body = []
    for row in rows:
        pareto_k = row.get("pareto_k_max")
        pareto_text = _format(pareto_k, "{:.2f}")
        if pareto_k is not None and np.isfinite(pareto_k) and pareto_k > threshold:
            pareto_text = "⚠" + pareto_text
        label = str(row.get("model_label", row.get("model", "")))
        body.append([label[:34],
                     _format(row.get("n"), "{:.0f}"), _format(row.get("nse")),
                     _format(row.get("rmse"), "{:.3g}"), _format(row.get("r2_log")),
                     _format(row.get("elpd_loo"), "{:.1f}"),
                     _format(row.get("se_loo"), "{:.1f}"),
                     _format(row.get("p_loo"), "{:.1f}"), pareto_text])
    footer = ("hover any header or cell for what the metric means",
              "ELPD_LOO ranks generalization; NSE ranks in-sample closeness - they "
              "disagree on purpose")
    return _grid_traces(site.sample_id, "fit", "table", _FIT_TABLE_HEADERS,
                        _FIT_TABLE_GLOSSARY, _FIT_TABLE_WIDTHS, body, footer)


def _cv_table_traces(site, cv_entry, fold):
    """The cross-validation score table for one fold.

    One table per fold rather than one per site, because the interesting comparison is
    "this fold against all of them" and that needs the selected fold's own numbers.
    The tables are text, so the extra traces cost bytes rather than time.
    """
    scheme = cv_entry.get("scheme", {})
    headline = cv_entry.get("headline", "per_fold")
    bayes_by_model = {fit.key: (fit.bayes or {}) for fit in site.fits if fit.ok}

    body = []
    for model_key, model_entry in cv_entry.get("models", {}).items():
        this_fold = next((entry for entry in model_entry["folds"]
                          if entry["fold"] == fold), None)
        fold_scores = (this_fold or {}).get("test_metrics", {}) or {}
        pooled = model_entry.get("pooled", {}) or {}
        per_fold = model_entry.get("per_fold", {}) or {}
        if headline == "pooled":
            all_nse = _format(pooled.get("nse"))
            all_r2 = _format(pooled.get("r2_log"))
        else:
            all_nse = (f"{_format(per_fold.get('nse_mean'))}"
                       f"±{_format(per_fold.get('nse_std'))}")
            all_r2 = (f"{_format(per_fold.get('r2_log_mean'))}"
                      f"±{_format(per_fold.get('r2_log_std'))}")
        bayes = bayes_by_model.get(model_key, {})
        elpd = (f"{_format(bayes.get('elpd_loo'), '{:.1f}')}"
                f"±{_format(bayes.get('se_loo'), '{:.1f}')}")
        body.append([
            str(catalog.short_label(model_key))[:26],
            f"{(this_fold or {}).get('n_train', '–')}/"
            f"{(this_fold or {}).get('n_test', '–')}",
            _format(fold_scores.get("nse")), _format(fold_scores.get("r2_log")),
            all_nse, all_r2, elpd])

    support = cv_entry.get("support", {}) or {}
    footer = [
        f"scheme: {scheme.get('description', 'cross-validation')}; folds fitted by "
        f"{str(cv_entry.get('fold_method', 'advi')).upper()}",
        ("'all folds' is pooled over every held-out prediction - a per-fold NSE is "
         "undefined when one point is held out"
         if headline == "pooled" else
         "'all folds' is mean±sd across folds; the sd is how much the model swings "
         "between training draws"),
        "ELPD_LOO comes from the all-data NUTS fit, not from these folds: PSIS-LOO "
        "reweights one posterior to approximate leaving each",
        "point out, so it uses every measurement at once. Fold scores are honest "
        "refits; ELPD is an approximation. Read both.",
    ]
    if support:
        footer.append(
            f"sample support: n={support.get('n')}, "
            f"{support.get('distinct_discharge')} distinct discharges spanning "
            f"x{_format(support.get('discharge_span_ratio'), '{:.1f}')} - "
            f"{'identifiable' if support.get('identifiable') else 'too narrow to identify a rating'}")

    glossary = ("curve", "n", "nse", "r2_log", "nse", "r2_log", "elpd_loo")
    return _grid_traces(site.sample_id, "cv", "cvtable", _CV_TABLE_HEADERS, glossary,
                        _CV_TABLE_WIDTHS, body, footer, tag_extra={"fold": fold})


# --- cross-validation pane ----------------------------------------------------

def _cv_pane_traces(site, cv_entry):
    """The cross-validation pane for one site: every (model, fold) refit."""
    import plotly.graph_objects as go
    sample_id = site.sample_id
    traces = []
    for model_key, model_entry in cv_entry.get("models", {}).items():
        color = model_entry.get("color", catalog.color(model_key))
        line_dash = model_entry.get("dash", catalog.dash(model_key))
        for fold in model_entry["folds"]:
            index = fold["fold"]
            scores = fold.get("test_metrics", {}) or {}
            band = _band_trace(fold["stage_ft"], fold["q_lower"], fold["q_upper"],
                               color, sample_id, pane="cv", legendkey=model_key,
                               model=model_key, fold=index,
                               label=f"{model_entry['label']} - 95% band")
            if band is not None:
                traces.append(band)
            traces.append(_curve_trace(
                fold["stage_ft"], fold["q_median"], color,
                f"{model_entry['label']} - fold {index}", sample_id, pane="cv",
                dash=line_dash, legendkey=model_key, model=model_key, fold=index,
                hovertemplate=(f"<b>{model_entry['label']} - fold {index}</b><br>"
                               f"stage %{{x:.2f}} ft<br>Q %{{y:.3g}} cfs<br>"
                               f"held-out NSE={_format(scores.get('nse'))} "
                               f"R²log={_format(scores.get('r2_log'))}"
                               f"<extra></extra>")))
            traces.append(go.Scatter(
                x=fold["train_stage"], y=fold["train_q"], mode="markers",
                xaxis="x2", yaxis="y2",
                marker=dict(size=9, color="black", symbol="circle-open",
                            line=dict(width=2)),
                name="trained on", visible=False, showlegend=False,
                legendgroup="measurements",
                meta={"sample": sample_id, "pane": "cv", "kind": "train",
                      "model": model_key, "fold": index,
                      "legendkey": "measurements"},
                hovertemplate=("trained on<br>stage %{x:.2f} ft<br>"
                               "Q %{y:.3g} cfs<extra></extra>")))
            traces.append(go.Scatter(
                x=fold["test_stage"], y=fold["test_q"], mode="markers",
                xaxis="x2", yaxis="y2",
                marker=dict(size=11, color="rgba(120,120,120,0.85)", symbol="x"),
                name="held out", visible=False, showlegend=False,
                legendgroup="held out",
                meta={"sample": sample_id, "pane": "cv", "kind": "test",
                      "model": model_key, "fold": index, "legendkey": "held out"},
                hovertemplate=("held out of this fold<br>stage %{x:.2f} ft<br>"
                               "Q %{y:.3g} cfs<extra></extra>")))
    for fold in _site_folds(cv_entry):
        traces.extend(_cv_table_traces(site, cv_entry, fold))
    return traces


# --- shared axes --------------------------------------------------------------

def _panel_range(site, cv_entry=None) -> list:
    """One frozen axis range per site, spanning its points and every curve drawn.

    Freezing the axes per site is what makes the panes comparable: switching model,
    switching fold or switching pane never rescales the plot, so a curve that moved
    can be seen to have moved. The bands are included, because a band that runs off
    the top of the plot is not visibly a band at all.

    Returns
    -------
    list
        ``[x0, x1, log10(y0), log10(y1), title]``.
    """
    stage = site.sample.stage_ft
    discharge = site.sample.discharge_cfs
    stages = [float(np.nanmin(stage)), float(np.nanmax(stage))]
    discharges = [float(value) for value in discharge if value > 0]
    window = _draw_window(site)

    def collect(curve, columns=("discharge_cfs",)):
        curve = _clip(curve, window)
        if curve.empty:
            return
        stages.extend([float(curve["stage_ft"].min()),
                       float(curve["stage_ft"].max())])
        for column in columns:
            if column not in curve:
                continue
            positive = curve[column][np.isfinite(curve[column])
                                     & (curve[column] > 0)]
            if len(positive):
                discharges.extend([float(positive.min()), float(positive.max())])

    for fit in site.fits:
        if _drawable(fit):
            collect(fit.curve, ("discharge_cfs", "lower", "upper"))
    for external in site.external_curves():
        collect(external.curve, ("discharge_cfs",))

    if cv_entry:
        import pandas as pd
        for model_entry in cv_entry.get("models", {}).values():
            for fold in model_entry["folds"]:
                collect(pd.DataFrame({"stage_ft": fold["stage_ft"],
                                      "discharge_cfs": fold["q_median"],
                                      "lower": fold["q_lower"],
                                      "upper": fold["q_upper"]}),
                        ("discharge_cfs", "lower", "upper"))

    x0, x1 = min(stages), max(stages)
    pad = 0.05 * ((x1 - x0) or 1.0)
    y0 = np.log10(min(discharges) * 0.6) if discharges else -1.0
    y1 = np.log10(max(discharges) * 1.8) if discharges else 1.0
    title = f"{site.label}  (n={len(site.sample)}) - stage: {site.stage_label}"
    return [x0 - pad, x1 + pad, float(y0), float(y1), title]


# --- legend and markers -------------------------------------------------------

def _legend_keys(sites) -> list:
    """``[(legend key, name, color, dash, width)]`` for everything drawn anywhere.

    The legend is built from the same keys the traces are tagged with, which is what
    lets a click on one legend entry hide the model's line *and* its band across
    every site.
    """
    keys = [("measurements", "measurements", "#000000", "solid", 0)]
    present = {fit.key for site in sites for fit in site.fits if _drawable(fit)}
    for entry in catalog.MODELS:
        if entry.key in present:
            keys.append((entry.key, entry.label, entry.color, entry.dash, 3))
    seen_external = {}
    for site in sites:
        for external in site.external_curves():
            seen_external.setdefault(external.key, external)
    for key, external in seen_external.items():
        style = _external_style(external)
        keys.append((key, style["legend"], style["color"], style["dash"], 3.2))
    if any(getattr(site, "_has_cv", False) for site in sites):
        keys.append(("held out", "held out of this fold", "#787878", "solid", 0))
    return keys


def _legend_traces(legend_keys):
    """Persistent legend entries. Clicking one is intercepted by the inspector.

    They carry no data - the real curves are per-site and hidden, so a legend built
    from them would appear and vanish as the cursor moved. Plotly's own legend toggle
    is therefore not what happens on a click: the injected script reads the click,
    records that the group is off, and repaints. Without that, a repaint (which sets
    ``visible`` on every tagged trace) would immediately undo the toggle, which is
    exactly why the legend used to do nothing.
    """
    import plotly.graph_objects as go
    traces = []
    for key, name, color, dash, width in legend_keys:
        if width:
            marker = dict(size=1)
            mode, line = "lines", dict(width=width, color=color, dash=dash)
        else:
            marker = dict(size=9, color=color, symbol="circle-open",
                          line=dict(width=2))
            mode, line = "markers", dict(width=0)
        traces.append(go.Scatter(
            x=[None], y=[None], mode=mode, xaxis="x2", yaxis="y2", line=line,
            marker=marker, name=name, showlegend=True, legendgroup=key,
            hoverinfo="skip", meta={"legendkey": key, "kind": "legend"}))
    return traces


def _map_markers(sites):
    """One map trace per sample source. ``customdata[0]`` drives the inspector."""
    import plotly.graph_objects as go
    traces = []
    sources = list(dict.fromkeys(site.source for site in sites))
    for source in sources:
        name, color, jitter = _style_for(source)
        subset = [site for site in sites if site.source == source]
        if not subset:
            continue
        payload = []
        for site in subset:
            headline = _headline_fit(site)
            bayes = (headline.bayes or {}) if headline else {}
            payload.append([site.sample_id, site.label, len(site.sample),
                            headline.label if headline else "-",
                            headline.metrics.get("r2_log", float("nan"))
                            if headline else float("nan"),
                            bayes.get("elpd_loo", float("nan"))])
        traces.append(go.Scattermap(
            lat=[site.coords[0] for site in subset],
            lon=[site.coords[1] + jitter for site in subset],
            mode="markers", marker=dict(size=13, color=color), name=name,
            legendgroup="sites", customdata=payload,
            hovertemplate=("<b>%{customdata[1]}</b><br>" + name + "<br>"
                           "n=%{customdata[2]} | best predictive: "
                           "%{customdata[3]}<br>"
                           "R²log %{customdata[4]:.2f}, "
                           "ELPD_LOO %{customdata[5]:.1f}<br>"
                           "<i>hover holds the inspector →</i><extra></extra>")))
    return traces


# --- controls and layout ------------------------------------------------------

def _controls(cv_models, has_cv: bool):
    """The inspector's buttons: pane toggle, model selector, fold arrows.

    Every button uses Plotly's ``skip`` method, so clicking it changes nothing by
    itself - it only announces a selection (``args[0]`` = what, ``args[1]`` = value)
    that the injected script reads and repaints from. That is what keeps one state
    machine in charge of visibility instead of scattering it across button
    definitions.

    The two cross-validation rows sit in their own band between the plot and the
    table (``MODEL_ROW_Y``, ``FOLD_ROW_Y``), which is reserved for them by
    ``TABLE_DOMAIN_Y_WITH_CV``. They used to be placed inside the table's band, so
    they drew on top of it.
    """
    background = "rgba(255,255,255,0.92)"
    pane_buttons = [dict(label="Fitted curves", method="skip", args=["pane", "fit"])]
    if has_cv:
        pane_buttons.append(dict(label="Cross-validation", method="skip",
                                 args=["pane", "cv"]))
    menus = [dict(type="buttons", direction="right", showactive=True, active=0,
                  x=PANEL_X[0], xanchor="left", y=1.045, yanchor="top",
                  buttons=pane_buttons, pad=dict(r=4, t=2), bgcolor=background)]
    if has_cv and cv_models:
        menus.append(dict(
            type="buttons", direction="right", showactive=True, active=0,
            visible=False, x=PANEL_X[0], xanchor="left", y=MODEL_ROW_Y,
            yanchor="top", pad=dict(t=2, r=2), bgcolor=background,
            buttons=[dict(label=catalog.short_label(key), method="skip",
                          args=["model", key]) for key, _ in cv_models]))
        menus.append(dict(
            type="buttons", direction="right", showactive=False, visible=False,
            x=PANEL_X[0], xanchor="left", y=FOLD_ROW_Y, yanchor="top",
            pad=dict(t=2, r=2), bgcolor=background,
            buttons=[dict(label="◀ prev fold", method="skip", args=["foldstep", -1]),
                     dict(label="next fold ▶", method="skip", args=["foldstep", 1])]))
    return menus


def layout_boxes(has_cv: bool) -> dict:
    """The named rectangles of the right-hand panel, in paper coordinates.

    Returned as ``{name: (x0, y0, x1, y1)}``. Exists so the collision check and the
    figure read one description of the geometry rather than two that can drift.

    Parameters
    ----------
    has_cv : bool
        Whether the cross-validation control rows are present, which is what decides
        how tall the table can be.

    Returns
    -------
    dict
    """
    table_y = TABLE_DOMAIN_Y_WITH_CV if has_cv else TABLE_DOMAIN_Y_NO_CV
    boxes = {
        "plot": (PANEL_X[0], PLOT_DOMAIN_Y[0], PANEL_X[1], PLOT_DOMAIN_Y[1]),
        "table": (PANEL_X[0], table_y[0], PANEL_X[1], table_y[1]),
        "map": (MAP_DOMAIN_X[0], 0.0, MAP_DOMAIN_X[1], 1.0),
    }
    if has_cv:
        boxes["model_buttons"] = (PANEL_X[0], MODEL_ROW_Y - BUTTON_ROW_HEIGHT,
                                  PANEL_X[1], MODEL_ROW_Y)
        boxes["fold_buttons"] = (PANEL_X[0], FOLD_ROW_Y - BUTTON_ROW_HEIGHT,
                                 PANEL_X[1], FOLD_ROW_Y)
    return boxes


def layout_overlaps(has_cv: bool) -> list:
    """Every pair of panel rectangles that intersects. Empty is the correct answer.

    A Plotly figure will happily draw a button row on top of a table, and the result
    looks like a rendering bug rather than a layout mistake. This makes the mistake
    checkable - :func:`build_map` calls it and logs, and the tests assert on it.

    Parameters
    ----------
    has_cv : bool

    Returns
    -------
    list of tuple
        ``(name_a, name_b, overlapping area)``.
    """
    boxes = layout_boxes(has_cv)
    names = sorted(boxes)
    clashes = []
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            ax0, ay0, ax1, ay1 = boxes[first]
            bx0, by0, bx1, by1 = boxes[second]
            width = min(ax1, bx1) - max(ax0, bx0)
            height = min(ay1, by1) - max(ay0, by0)
            if width > 1e-9 and height > 1e-9:
                clashes.append((first, second, float(width * height)))
    return clashes


def _cv_model_list(cv_data) -> list:
    """``[(model key, label)]`` across every site, in catalog order."""
    seen = {}
    for entry in cv_data.values():
        for key, model_entry in entry.get("models", {}).items():
            seen.setdefault(key, model_entry.get("label", key))
    order = catalog.all_keys()
    return sorted(seen.items(),
                  key=lambda item: (order.index(item[0]) if item[0] in order else 99,
                                    item[0]))


def _site_folds(cv_entry) -> list:
    """Sorted fold indices available for a site, so the arrows offer only real folds."""
    folds = set()
    for model_entry in cv_entry.get("models", {}).values():
        folds.update(fold["fold"] for fold in model_entry["folds"])
    return sorted(folds)


def build_figure(sites, cv_data=None, *, comparison_rows=None,
                 title="Rating curves - hover a site to inspect its fits",
                 default_site=None) -> tuple:
    """Assemble the figure and its inspector script, without writing anything.

    Separate from :func:`build_map` so the structure can be checked in a test - the
    trace tags, the legend keys, the styles and the panel geometry - rather than only
    eyeballed in a browser.

    Parameters
    ----------
    sites : sequence of SiteRating
        The sites to draw. Sites with no coordinates or no successful fit are skipped.
    cv_data : dict, optional
        See :func:`build_map`.
    comparison_rows : callable, optional
        See :func:`build_map`.
    title : str
        Figure title.
    default_site : str, optional
        ``sample_id`` the inspector opens on.

    Returns
    -------
    tuple
        ``(figure, script)`` - a ``plotly.graph_objects.Figure`` and the JavaScript to
        pass as ``post_script``.

    Raises
    ------
    ValueError
        If no site has both coordinates and a fitted curve.
    """
    return _assemble(sites, cv_data, comparison_rows, title, default_site)


def build_map(sites, output_html=None, cv_data=None, *, comparison_rows=None,
              title="Rating curves - hover a site to inspect its fits",
              default_site=None) -> str:
    """Assemble and write the interactive map.

    Parameters
    ----------
    sites : sequence of SiteRating
        The sites to draw. Sites with no coordinates or no successful fit are
        skipped.
    output_html : path-like, optional
        Where to write. Defaults to ``settings.MAP_HTML``.
    cv_data : dict, optional
        Cross-validation payload keyed by ``sample_id``, each value being
        :meth:`CrossValidation.to_map_payload` output - the fold curves *and* the
        pooled / per-fold summary tables and the scheme. When omitted the
        cross-validation pane and its controls are left out entirely and the score
        table takes the extra room.
    comparison_rows : callable, optional
        ``f(site) -> list of dict`` producing the score-table rows. Defaults to
        :func:`limnotech_rating_curves.evaluate.metrics.comparison_rows`.
    title : str
        Figure title.
    default_site : str, optional
        ``sample_id`` the inspector opens on. Defaults to the first site.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If no site has both coordinates and a fitted curve, since there would be
        nothing to draw.
    """
    figure, script = _assemble(sites, cv_data, comparison_rows, title, default_site)
    path = Path(settings.MAP_HTML if output_html is None else output_html)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), post_script=script, config={"scrollZoom": True},
                      include_plotlyjs=True)
    log.info("map -> %s (%d traces)", path, len(figure.data))
    return str(path)


def _assemble(sites, cv_data, comparison_rows, title, default_site) -> tuple:
    """Build the figure and the inspector script. See :func:`build_figure`."""
    import plotly.graph_objects as go

    if comparison_rows is None:
        comparison_rows = metrics_module.comparison_rows
    sites = [site for site in sites if _has_curves(site)]
    if not sites:
        raise ValueError("no site has both coordinates and a fitted curve, so there "
                         "is nothing to map")
    cv_data = cv_data or {}
    has_cv = bool(cv_data)

    traces, ranges, folds_by_site = [], {}, {}
    for site in sites:
        cv_entry = cv_data.get(site.sample_id)
        site._has_cv = bool(cv_entry)
        traces.extend(_fit_pane_traces(site))
        traces.extend(_fit_table_traces(site, comparison_rows(site)))
        if cv_entry:
            traces.extend(_cv_pane_traces(site, cv_entry))
            folds_by_site[site.sample_id] = _site_folds(cv_entry)
        ranges[site.sample_id] = _panel_range(site, cv_entry)

    legend_keys = _legend_keys(sites)
    traces.extend(_legend_traces(legend_keys))
    traces.extend(_map_markers(sites))

    cv_models = _cv_model_list(cv_data)
    center = dict(lat=float(np.mean([site.coords[0] for site in sites])),
                  lon=float(np.mean([site.coords[1] for site in sites])))
    table_y = TABLE_DOMAIN_Y_WITH_CV if has_cv else TABLE_DOMAIN_Y_NO_CV

    clashes = layout_overlaps(has_cv)
    if clashes:
        log.warning("the inspector's panel geometry overlaps: %s - fix the domains "
                    "in view.mapview rather than shipping a map with a control row "
                    "drawn over the table",
                    ", ".join(f"{a}/{b}" for a, b, _ in clashes))

    figure = go.Figure(data=traces, layout=go.Layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=14)),
        height=FIGURE_HEIGHT,
        map=dict(style="white-bg", layers=BASEMAP_LAYERS, center=center, zoom=8,
                 domain=dict(x=MAP_DOMAIN_X, y=[0.0, 1.0])),
        xaxis2=dict(domain=PANEL_X, anchor="y2", title="stage (ft)",
                    showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis2=dict(domain=PLOT_DOMAIN_Y, anchor="x2", type="log",
                    title="discharge (cfs, log)", showgrid=True,
                    gridcolor="rgba(0,0,0,0.08)"),
        # the score table's own axes: fixed 0-1, invisible, no interaction
        xaxis3=dict(domain=PANEL_X, anchor="y3", range=[0, 1], visible=False,
                    fixedrange=True),
        yaxis3=dict(domain=table_y, anchor="x3", range=[0, 1], visible=False,
                    fixedrange=True),
        # over the basemap's bottom-left corner: it grows upward over the map as
        # models are added, and can therefore never reach the plot or the table
        legend=dict(x=0.004, y=0.01, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.82)", bordercolor="#bbbbbb",
                    borderwidth=1, font=dict(size=10), groupclick="togglegroup",
                    itemsizing="constant"),
        updatemenus=_controls(cv_models, has_cv=has_cv),
        annotations=[
            dict(x=(PANEL_X[0] + PANEL_X[1]) / 2, y=1.005, xref="paper",
                 yref="paper", xanchor="center", yanchor="bottom", showarrow=False,
                 font=dict(size=12, color="#333"),
                 text="hover a site"),                       # [0] the site title
            dict(x=PANEL_X[1], y=MODEL_ROW_Y, xref="paper", yref="paper",
                 xanchor="right", yanchor="top", showarrow=False,
                 font=dict(size=11, color="#333"),
                 text="")],                                  # [1] the fold indicator
        margin=dict(l=0, r=10, t=80, b=20), dragmode="pan", hovermode="closest",
        paper_bgcolor="white", plot_bgcolor="white"))

    opening = default_site if default_site in ranges else sites[0].sample_id
    script = (_INSPECTOR_JS
              .replace("__RANGES__", json.dumps(ranges))
              .replace("__FOLDS__", json.dumps(folds_by_site))
              .replace("__DEFAULT__", json.dumps(opening))
              .replace("__HASCV__", json.dumps(has_cv)))
    log.info("assembled %d site(s), %d trace(s), %d legend key(s)", len(sites),
             len(traces), len(legend_keys))
    return figure, script


_INSPECTOR_JS = r"""
var gd = document.getElementById('{plot_id}');
var RANGES = __RANGES__;      // sample_id -> [x0, x1, log10 y0, log10 y1, title]
var FOLDS = __FOLDS__;        // sample_id -> [fold indices]
var DEFAULT_SITE = __DEFAULT__;
var HAS_CV = __HASCV__;

// the inspector's whole state; every control mutates this and repaints. `hidden`
// holds the legend groups the reader has switched off, which is why a legend click
// survives the next repaint.
var state = {site: DEFAULT_SITE, pane: 'fit', model: null, fold: null, hidden: {}};

function modelButtons() {
    var menus = gd.layout.updatemenus || [];
    return (menus.length > 1) ? menus[1].buttons.map(function (b) { return b.args[1]; })
                              : [];
}

function firstModel() {
    var keys = modelButtons();
    return keys.length ? keys[0] : null;
}

function isVisible(tag) {
    if (!tag || tag.sample !== state.site) return false;
    if (tag.legendkey && state.hidden[tag.legendkey]) return false;
    if (tag.pane === 'both') return true;
    if (tag.pane !== state.pane) return false;
    if (state.pane === 'cv') {
        if (['band', 'curve', 'train', 'test'].indexOf(tag.kind) >= 0) {
            return tag.model === state.model && tag.fold === state.fold;
        }
        if (tag.kind === 'cvtable') return tag.fold === state.fold;
    }
    return true;
}

function siteFolds() { return FOLDS[state.site] || []; }

function clampFold() {
    var folds = siteFolds();
    if (state.fold === null || folds.indexOf(state.fold) < 0) {
        state.fold = folds.length ? folds[0] : null;
    }
}

function stepFold(delta) {
    var folds = siteFolds();
    if (!folds.length) { state.fold = null; return; }
    var at = folds.indexOf(state.fold);
    if (at < 0) at = 0;
    state.fold = folds[(at + delta + folds.length) % folds.length];
}

function updateControls() {
    var inCv = state.pane === 'cv';
    var relayout = {};
    var menus = gd.layout.updatemenus || [];
    if (HAS_CV) {
        relayout['updatemenus[0].active'] = inCv ? 1 : 0;
        if (menus.length > 1) {
            var at = modelButtons().indexOf(state.model);
            relayout['updatemenus[1].visible'] = inCv;
            relayout['updatemenus[1].active'] = at < 0 ? 0 : at;
        }
        if (menus.length > 2) relayout['updatemenus[2].visible'] = inCv;
    }
    var folds = siteFolds();
    relayout['annotations[1].text'] = (inCv && state.fold !== null)
        ? ('fold ' + (folds.indexOf(state.fold) + 1) + ' of ' + folds.length
           + '  (index ' + state.fold + ')')
        : '';
    Plotly.relayout(gd, relayout);
}

function repaint() {
    if (!RANGES[state.site]) return;
    if (state.pane === 'cv' && state.model === null) state.model = firstModel();
    var indices = [], visible = [];
    gd.data.forEach(function (trace, index) {
        var tag = trace.meta;
        if (!tag) return;
        if (tag.sample !== undefined) {
            indices.push(index);
            visible.push(isVisible(tag));
        } else if (tag.kind === 'legend') {
            // keep the key on screen either way; 'legendonly' is how Plotly shows a
            // switched-off series, so the reader can switch it back on
            indices.push(index);
            visible.push(state.hidden[tag.legendkey] ? 'legendonly' : true);
        }
    });
    Plotly.restyle(gd, {visible: visible}, indices);
    var range = RANGES[state.site];
    Plotly.relayout(gd, {
        'xaxis2.range': [range[0], range[1]],
        'yaxis2.range': [range[2], range[3]],
        'annotations[0].text': range[4]
    });
    updateControls();
}

function selectSite(sampleId) {
    if (!RANGES[sampleId]) return;
    state.site = sampleId;
    if (HAS_CV) clampFold();
    repaint();
}

gd.on('plotly_hover', function (event) {
    if (!event.points || !event.points.length) return;
    var point = event.points[0];
    if (point.customdata && typeof point.customdata[0] === 'string'
        && RANGES[point.customdata[0]]) {
        selectSite(point.customdata[0]);
    }
});

// Plotly's own legend toggle sets `visible` on the clicked trace, and the next
// repaint would overwrite it. So the click is intercepted: record the group, repaint,
// and return false to stop Plotly acting on it.
gd.on('plotly_legendclick', function (event) {
    var trace = gd.data[event.curveNumber];
    var key = trace && trace.meta && trace.meta.legendkey;
    if (!key) return true;                    // the map's own site traces: let Plotly
    state.hidden[key] = !state.hidden[key];
    repaint();
    return false;
});

gd.on('plotly_legenddoubleclick', function (event) {
    var trace = gd.data[event.curveNumber];
    var key = trace && trace.meta && trace.meta.legendkey;
    if (!key) return true;
    // double click: show only this group, or everything if it is already alone
    var others = Object.keys(state.hidden).filter(function (k) {
        return state.hidden[k];
    });
    var alone = true;
    gd.data.forEach(function (t) {
        var m = t.meta;
        if (m && m.kind === 'legend' && m.legendkey !== key && !state.hidden[m.legendkey]) {
            alone = false;
        }
    });
    state.hidden = {};
    if (!alone) {
        gd.data.forEach(function (t) {
            var m = t.meta;
            if (m && m.kind === 'legend' && m.legendkey !== key) {
                state.hidden[m.legendkey] = true;
            }
        });
    }
    repaint();
    return false;
});

// buttons use method 'skip', so their selection arrives here as (args[0], args[1])
gd.on('plotly_buttonclicked', function (event) {
    var args = event.button && event.button.args;
    if (!args || args.length < 2) return;
    if (args[0] === 'pane') {
        state.pane = args[1];
        if (HAS_CV) clampFold();
    } else if (args[0] === 'model') {
        state.model = args[1];
    } else if (args[0] === 'foldstep') {
        stepFold(args[1]);
    }
    repaint();
});

if (HAS_CV) clampFold();
if (DEFAULT_SITE) selectSite(DEFAULT_SITE);
"""
