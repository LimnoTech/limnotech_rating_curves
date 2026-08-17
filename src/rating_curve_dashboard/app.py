"""
Streamlit dashboard: pull USGS field measurements (or upload your own), get a
fitted rating curve.

Deployment note: this app only needs pandas/numpy/scipy/matplotlib/streamlit
plus the lightweight `dataretrieval` USGS client (see requirements.txt). It
deliberately does NOT depend on the rest of this package (limnotech_rating_curves)
or its PyMC/PyTensor-based Bayesian models - the segmented power-law approach in
engine.py was validated once, offline, against those models, but importing them
here would drag PyMC/PyTensor into this process, which conflicts with
matplotlib's Agg backend in the same process on Windows. Keeping the dashboard
on this lightweight stack is what makes it fast enough to refit on every
fetch/upload and easy to deploy on a free host.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import engine  # noqa: E402  (sets MKL env vars before numpy loads - keep this import first)
import usgs_fetch as uf  # noqa: E402

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

COLOR_DATA = "#2a78d6"
COLOR_CURVE = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#52514e"

STAGE_ALIASES = (
    "stage_ft", "stage", "gage_height_ft", "gh_ft", "gage height", "stage (ft)",
    "gage_height_va", "gage height, ft", "gage height, feet", "gage_ht", "gh_va",
    "gage_height", "stage_ht_ft",
)
DISCHARGE_ALIASES = (
    "discharge_cfs", "discharge", "q_cfs", "q", "discharge (cfs)", "flow_cfs",
    "discharge_va", "discharge, cfs", "discharge, ft3/s", "chan_discharge",
    "q_va", "streamflow", "flow",
)
DATETIME_ALIASES = (
    "datetime", "date", "time", "date_time", "measurement_dt", "timestamp",
    "measurement date", "date/time",
)

st.set_page_config(page_title="USGS Rating Curve Builder", layout="wide")


def guess_column(columns, aliases):
    lower = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


@st.cache_data(show_spinner=False)
def run_pipeline(stage: np.ndarray, discharge: np.ndarray, max_segments: int, parsimony_margin: float):
    return engine.select_best_model(stage, discharge, max_segments=max_segments,
                                     parsimony_margin=parsimony_margin)


def make_plot(stage, discharge, model, site_label):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    curve_stage = np.linspace(stage.min(), stage.max() * 1.03, 600)
    curve_q = model.predict(curve_stage)
    ax.plot(curve_q, curve_stage, color=COLOR_CURVE, linewidth=2, zorder=2,
             label=f"Fitted rating curve ({model.n_segments}-segment power law)")

    for bp in model.breakpoints:
        ax.axhline(bp, color=COLOR_AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=1)

    ax.scatter(discharge, stage, s=24, color=COLOR_DATA, edgecolor="white",
               linewidth=0.6, zorder=3, label=f"Measured field data (n={len(stage)})")

    ax.set_xscale("log")
    ax.set_xlabel("Discharge (cfs)", color=COLOR_INK)
    ax.set_ylabel("Stage / gage height (ft)", color=COLOR_INK)
    ax.set_title(f"Stage-Discharge Rating Curve\n{site_label}", color=COLOR_INK,
                 fontsize=12.5, fontweight="bold", linespacing=1.6)
    ax.grid(True, which="major", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=COLOR_GRID, linewidth=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED)
    legend = ax.legend(loc="lower right", frameon=True, fontsize=9)
    legend.get_frame().set_edgecolor(COLOR_AXIS)
    fig.tight_layout()
    return fig


st.title("USGS Stage-Discharge Rating Curve Builder")
st.caption(
    "Pull real USGS field measurements straight from NWIS, or upload your own (discrete "
    "gaugings — a measured stage paired with a measured discharge). The app fits 1-, 2-, "
    "and 3-segment power-law rating curves, picks the best one by cross-validation, and "
    "gives you an equation, a stage ↔ discharge lookup table, and a plot with R²."
)

with st.sidebar:
    st.header("1. Data source")
    source = st.radio("Where should the data come from?", ["Fetch from USGS", "Upload a file"], index=0)
    site_label = st.text_input("Site label (for the plot/report title)", value="")
    max_segments = st.radio(
        "Candidate models to compare",
        options=[2, 3],
        format_func=lambda n: "1- and 2-segment power law" if n == 2 else "1-, 2-, and 3-segment power law",
        index=1,
    )
    parsimony_margin_pct = st.slider(
        "Parsimony margin (%)", min_value=0, max_value=10, value=2, step=1,
        help="Auto-selection picks the simplest candidate model whose cross-validated "
             "log-RMSE is within this margin of the best-fitting candidate. Lower it to "
             "0 to always auto-select whichever model cross-validates most accurately, "
             "regardless of how many breakpoints it has.",
    )
    parsimony_margin = parsimony_margin_pct / 100
    table_step = st.number_input("Lookup table stage increment (ft)", value=0.01, min_value=0.001,
                                  max_value=1.0, step=0.01, format="%.3f")

if source == "Fetch from USGS":
    st.subheader("2. Choose a site, data type, and date range")
    site_input = st.text_input(
        "USGS site number", value="06893578",
        help="e.g. 06893578 for Blue River at Stadium Drive, Kansas City, MO. "
             "Find a site number at waterdata.usgs.gov/nwis.",
    )
    if not site_input.strip():
        st.info("Enter a USGS site number to get started.")
        st.stop()

    with st.spinner("Looking up site..."):
        try:
            info = uf.site_lookup(site_input)
        except Exception as exc:
            st.error(f"Couldn't reach USGS to look up that site: {exc}")
            st.stop()
    if info is None:
        st.error(f"No USGS site found for '{site_input}'. Double-check the site number.")
        st.stop()
    drainage = info.get("drainage_area_sqmi")
    st.caption(
        f"**{info['name']}** — {info['state']}"
        + (f", drainage area {drainage:,.0f} sq mi" if pd.notna(drainage) else "")
    )

    with st.spinner("Checking what data this site publishes..."):
        try:
            avail = uf.available_series(site_input)
        except Exception as exc:
            st.error(f"Couldn't reach USGS to check available data: {exc}")
            st.stop()
    if avail.empty:
        st.error("This site doesn't publish any stage/discharge data this dashboard understands.")
        st.stop()

    data_type_label = st.selectbox("Data type", avail["label"])
    row = avail.loc[avail["label"] == data_type_label].iloc[0]
    kind = row["kind"]
    record_start, record_end = row["begin"].date(), row["end"].date()

    if kind != uf.FIELD_MEASUREMENTS:
        st.warning(
            "This series is the gage's own continuous record: its discharge was computed "
            "from stage through the rating that's already in place, so a curve fit to it "
            "mostly reproduces that existing rating rather than deriving an independent one. "
            "Prefer field measurements to actually validate or rebuild the rating."
        )

    default_start = record_start
    if kind == uf.INSTANTANEOUS:
        # 15-minute data over decades is a lot to fetch and pivot - default to a recent
        # window and require the user to deliberately widen it, rather than fetching a
        # 20+ year, ~2M-row range the first time this option is picked.
        default_start = max(record_start, record_end - pd.Timedelta(days=730))

    date_range = st.date_input(
        "Date range", value=(default_start, record_end),
        min_value=record_start, max_value=record_end,
    )
    if len(date_range) != 2:
        st.info("Pick both a start and end date.")
        st.stop()
    start_date, end_date = date_range
    span_days = (end_date - start_date).days

    if kind == uf.INSTANTANEOUS and span_days > 730:
        st.warning(
            "Instantaneous data over more than ~2 years can be slow to fetch and fit "
            "(15-minute readings add up fast)."
        )
        if not st.checkbox("I understand this may take a while - fetch it anyway"):
            st.stop()

    with st.spinner(f"Fetching data from USGS site {uf.normalize_site_id(site_input)}..."):
        try:
            fetched = uf.fetch(site_input, str(start_date), str(end_date), kind)
        except Exception as exc:
            st.error(f"Couldn't fetch data from USGS: {exc}")
            st.stop()

    data = fetched.dropna(subset=["stage_ft", "discharge_cfs"])
    data = data[data["discharge_cfs"] > 0]
    if len(data) < 8:
        st.error(
            f"Only {len(data)} valid measurement(s) for this data type and date range - "
            "need at least 8. Try widening the date range or a different data type."
        )
        st.stop()

    stage = data["stage_ft"].to_numpy(dtype=float)
    discharge = data["discharge_cfs"].to_numpy(dtype=float)
    data_time = data["time"]
    label = site_label or info["name"] or f"USGS {uf.normalize_site_id(site_input)}"
    st.success(f"Fetched {len(data)} measurement(s), {start_date} to {end_date}.")

else:
    st.subheader("2. Upload and select columns")
    uploaded = st.file_uploader("USGS field measurements (CSV or Excel)", type=["csv", "xlsx", "xls"])
    st.caption(
        "Needs a stage column (ft) and a discharge column (cfs) — column names like "
        "`stage_ft`/`discharge_cfs` are auto-detected, or pick them manually below."
    )
    if uploaded is None:
        st.info("Upload a file to get started.")
        st.stop()

    try:
        raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")
        st.stop()

    guessed_stage = guess_column(raw.columns, STAGE_ALIASES)
    guessed_discharge = guess_column(raw.columns, DISCHARGE_ALIASES)
    if guessed_stage is None or guessed_discharge is None:
        st.info(
            "Couldn't auto-detect the stage/discharge columns from this file's headers — "
            "double-check the selections below before continuing."
        )
    default_stage_idx = list(raw.columns).index(guessed_stage) if guessed_stage else 0
    default_discharge_idx = list(raw.columns).index(guessed_discharge) if guessed_discharge else (
        1 if len(raw.columns) > 1 and default_stage_idx == 0 else 0
    )
    col1, col2 = st.columns(2)
    with col1:
        stage_col = st.selectbox("Stage column (ft)", raw.columns, index=default_stage_idx)
    with col2:
        discharge_col = st.selectbox("Discharge column (cfs)", raw.columns, index=default_discharge_idx)

    if stage_col == discharge_col:
        st.error("Stage and discharge must be different columns — please select two distinct columns above.")
        st.stop()

    data = raw[[stage_col, discharge_col]].rename(columns={stage_col: "stage_ft", discharge_col: "discharge_cfs"})
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    n_dropped = len(raw) - len(data)
    data = data[data["discharge_cfs"] > 0]

    if n_dropped:
        st.warning(f"Dropped {n_dropped} row(s) with missing/non-numeric stage or discharge.")
    if len(data) < 8:
        st.error("Need at least 8 valid measurements to fit and cross-validate a rating curve.")
        st.stop()

    stage = data["stage_ft"].to_numpy(dtype=float)
    discharge = data["discharge_cfs"].to_numpy(dtype=float)
    label = site_label or uploaded.name

    guessed_time = guess_column(raw.columns, DATETIME_ALIASES)
    data_time = None
    if guessed_time:
        parsed_time = pd.to_datetime(raw.loc[data.index, guessed_time], errors="coerce")
        if parsed_time.notna().all():
            data_time = parsed_time

with st.expander("Preview selected data", expanded=False):
    fig_preview, ax_preview = plt.subplots(figsize=(6, 4), dpi=120)
    ax_preview.scatter(discharge, stage, s=16, color=COLOR_DATA)
    ax_preview.set_xlabel("Discharge (cfs)")
    ax_preview.set_ylabel("Stage (ft)")
    ax_preview.set_title("Raw selected columns (before fitting)")
    st.pyplot(fig_preview, use_container_width=True)
    st.caption(
        "Sanity-check this before fitting: stage should span a modest range in feet, "
        "discharge should generally increase with stage. If this looks wrong, re-check "
        "the column selections above."
    )

st.subheader("3. Fitted rating curve")
try:
    with st.spinner("Fitting candidate models and cross-validating on your measured data..."):
        best_model, comparison, fitted_models = run_pipeline(stage, discharge, max_segments, parsimony_margin)
except RuntimeError as exc:
    st.error(
        f"Couldn't fit a rating curve to this data: {exc}\n\n"
        "This usually means the stage/discharge columns are mismatched (check the preview "
        "plot above) or the data doesn't follow a single- to triple-segment power-law shape "
        "(e.g. too few distinct stage values, or stage and discharge not consistently "
        "increasing together)."
    )
    st.stop()

auto_n_seg = best_model.n_segments
model_options = sorted(fitted_models)


def _model_option_label(n_seg):
    piece = "" if n_seg == 1 else f" (piecewise, {n_seg - 1} breakpoint{'s' if n_seg > 2 else ''})"
    auto = " — auto-selected" if n_seg == auto_n_seg else ""
    return f"{n_seg}-segment power law{piece}{auto}"


chosen_n_seg = st.radio(
    "Model to display", options=model_options, format_func=_model_option_label,
    index=model_options.index(auto_n_seg), horizontal=True,
)
display_model = fitted_models[chosen_n_seg]
if chosen_n_seg != auto_n_seg:
    st.caption(
        f"Showing the {chosen_n_seg}-segment model instead of the auto-selected "
        f"{auto_n_seg}-segment one. Auto-selection picks the *simplest* candidate within "
        f"the parsimony margin (currently {parsimony_margin_pct}%, set in the sidebar) of "
        "the best cross-validated log-RMSE — lower that margin to 0 to auto-select purely "
        "on cross-validated accuracy instead."
    )

dm = display_model.metrics
cv = dm["cv"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Model shown", f"{display_model.n_segments}-segment")
col2.metric("R² (cross-validated)", f"{cv['r2']:.3f}")
col3.metric("RMSE (cross-validated)", f"{cv['rmse_cfs']:,.0f} cfs")
col4.metric("Bias (cross-validated)", f"{cv['pbias_pct']:+.1f}%")
col5.metric("Measurements used", f"{len(stage)}")

st.code(display_model.equation(), language=None)

fig = make_plot(stage, discharge, display_model, label)
st.pyplot(fig, use_container_width=True)

with st.expander("Model comparison (why this model was selected)", expanded=True):
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption(
        "cv_rmse_log is cross-validated RMSE in log space — the primary metric for comparing "
        "models on a curve spanning orders of magnitude, since a linear-space RMSE would be "
        "dominated by the highest flows. cv_pbias_pct is percent bias (systematic over/under-"
        "prediction): two models can have similar RMSE but very different bias, which matters "
        "if you care about total flow volume rather than just point accuracy. The simplest "
        "model within the parsimony margin (sidebar) of the best cv_rmse_log is auto-selected — "
        "an extra breakpoint has to earn its keep on held-out predictions, not just fit the "
        "training data more closely. Use the model selector above to override the auto pick."
    )

if data_time is not None:
    st.subheader("Is the rating stable over time?")
    trend = engine.residual_trend(display_model, data_time, stage, discharge)
    rolling = trend["percent_diff"].rolling(
        window=max(5, len(trend) // 15), center=True, min_periods=3
    ).median()

    fig_trend, ax_trend = plt.subplots(figsize=(9, 3.5), dpi=140)
    fig_trend.patch.set_facecolor("#fcfcfb")
    ax_trend.set_facecolor("#fcfcfb")
    ax_trend.axhline(0, color=COLOR_AXIS, linewidth=1)
    ax_trend.scatter(trend["time"], trend["percent_diff"], s=14, color=COLOR_DATA,
                      alpha=0.6, label="Percent residual (observed vs. predicted)")
    ax_trend.plot(trend["time"], rolling, color=COLOR_CURVE, linewidth=2, label="Rolling median")
    ax_trend.set_ylabel("Percent difference")
    ax_trend.set_title("Residual vs. time — a trend or step here means the rating shifted",
                        fontsize=11, color=COLOR_INK)
    ax_trend.grid(True, color=COLOR_GRID, linewidth=0.6)
    for spine in ("top", "right"):
        ax_trend.spines[spine].set_visible(False)
    ax_trend.legend(loc="best", fontsize=8, frameon=True)
    fig_trend.tight_layout()
    st.pyplot(fig_trend, use_container_width=True)
    st.caption(
        "If this drifts or steps away from zero over time rather than scattering randomly "
        "around it, the physical rating at this gage likely changed partway through the "
        "fitted record (channel/sediment change, vegetation, a USGS re-rating) — no curve "
        "shape fits two different underlying relationships well at once. Narrow the date "
        "range in step 2 to the recent, stable era for the most accurate predictive model."
    )

st.subheader("4. Stage ↔ discharge lookup")
tab1, tab2, tab3 = st.tabs(["Predict from stage", "Predict from discharge", "Full table"])

with tab1:
    stage_input = st.number_input("Stage (ft)", value=float(np.median(stage)), format="%.3f")
    q = display_model.predict(stage_input)
    extrapolated = display_model.is_extrapolation(stage_input)
    st.metric("Predicted discharge", f"{q:,.1f} cfs")
    if extrapolated:
        st.warning("This stage is outside the measured data range — treat the prediction as extrapolated.")

with tab2:
    discharge_input = st.number_input("Discharge (cfs)", value=float(np.median(discharge)), format="%.1f")
    try:
        h = display_model.predict_stage(discharge_input)
        st.metric("Predicted stage", f"{h:,.3f} ft")
    except ValueError as exc:
        st.error(str(exc))

with tab3:
    table = display_model.curve_table(step=table_step)
    st.dataframe(table, use_container_width=True, height=400)
    st.download_button(
        "Download full lookup table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="rating_curve_table.csv",
        mime="text/csv",
    )

st.subheader("5. Download results")
d1, d2, d3 = st.columns(3)
with d1:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    st.download_button("Plot (PNG)", data=buf.getvalue(), file_name="rating_curve_plot.png", mime="image/png")
with d2:
    st.download_button("Fitted model (JSON)", data=display_model.to_json(), file_name="rating_curve_model.json",
                        mime="application/json")
with d3:
    st.download_button("Model comparison (CSV)", data=comparison.to_csv(index=False).encode("utf-8"),
                        file_name="model_comparison.csv", mime="text/csv")
