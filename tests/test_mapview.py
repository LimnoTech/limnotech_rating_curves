import numpy as np
import pandas as pd
import pytest

import limnotech_rating_curves as lrc
from limnotech_rating_curves.core import ExternalCurve, SiteRating
from limnotech_rating_curves.models import catalog
from limnotech_rating_curves.view import mapview


def _site(sample_id="magl:SBR-99", coords=(42.3, -83.4), models=("linear",
                                                                 "quadratic")):
    stage = np.linspace(2.0, 9.0, 12)
    discharge = 12.0 * (stage - 1.5) ** 1.9
    frame = pd.DataFrame({"stage_ft": stage, "discharge_cfs": discharge})
    sample = lrc.Sample.of(frame, site_id=sample_id.split(":")[-1], source="magl")
    return SiteRating(
        sample_id=sample_id, source="magl", label=sample_id, sample=sample,
        coords=coords,
        fits=[catalog.get(key).fit(sample) for key in models])


def _spreadsheet_curve(site, factor=1.15):
    curve = site.fits[0].curve[["stage_ft", "discharge_cfs"]].copy()
    curve["discharge_cfs"] *= factor
    return ExternalCurve(
        key="spreadsheet", label="the workbook's own quadratic",
        kind="spreadsheet", curve=curve,
        metrics={"n": 12, "nse": 0.9, "rmse": 3.0, "r2_log": 0.95},
        detail={"form": "quadratic", "axis": "distance_mm",
                "equation": "Q = 6.29 h^2 - 10024 h + 3990000"})


def _by_kind(figure, kind, pane=None):
    return [trace for trace in figure.data
            if trace.meta and trace.meta.get("kind") == kind
            and (pane is None or trace.meta.get("pane") == pane)]


def _figure(sites, cv_data=None):
    """The figure the map would write, built through the same code path."""
    figure, _ = mapview.build_figure(sites if isinstance(sites, list) else [sites],
                                     cv_data=cv_data, title="test")
    return figure


# --- geometry -----------------------------------------------------------------

@pytest.mark.parametrize("has_cv", [False, True])
def test_no_panel_element_overlaps_another(has_cv):
    """The defect that made the map look broken: buttons drawn over the table."""
    assert mapview.layout_overlaps(has_cv) == []


@pytest.mark.parametrize("has_cv", [False, True])
def test_every_panel_box_is_inside_the_figure(has_cv):
    for name, (x0, y0, x1, y1) in mapview.layout_boxes(has_cv).items():
        assert 0.0 <= x0 < x1 <= 1.0, name
        assert 0.0 <= y0 < y1 <= 1.0, name


def test_the_control_rows_sit_between_the_plot_and_the_table():
    boxes = mapview.layout_boxes(has_cv=True)
    plot_bottom = boxes["plot"][1]
    table_top = boxes["table"][3]
    for name in ("model_buttons", "fold_buttons"):
        _, low, _, high = boxes[name]
        assert high <= plot_bottom, f"{name} reaches into the plot"
        assert low >= table_top, f"{name} reaches into the table"


def test_the_legend_is_over_the_map_not_the_panel(tmp_path):
    site = _site()
    figure = _figure([site])
    legend = figure.layout.legend
    assert legend.x < mapview.MAP_DOMAIN_X[1], \
        "the legend must sit over the basemap, where it cannot reach the plot"
    assert legend.yanchor == "bottom", \
        "anchored at the bottom so more models grow it upward over the map"
    assert legend.groupclick == "togglegroup"


def _written(tmp_path, sites, cv_data=None):
    path = tmp_path / "map.html"
    mapview.build_map(sites, path, cv_data=cv_data, title="test")
    assert path.exists() and path.stat().st_size > 100_000     # plotly.js is inlined
    return path


# --- the fitted-curves pane ---------------------------------------------------

def test_every_fitted_curve_has_a_band_in_the_fit_pane(tmp_path):
    """The fitted-curves pane used to have no credible interval at all."""
    site = _site(models=("linear", "quadratic", "exponential"))
    figure = _figure([site])

    curves = {trace.meta["model"] for trace in _by_kind(figure, "curve", "fit")}
    bands = {trace.meta["model"] for trace in _by_kind(figure, "band", "fit")}
    assert curves == {"linear", "quadratic", "exponential"}
    assert bands == curves, "every curve with an interval must have its band drawn"


def test_a_band_and_its_curve_share_one_legend_group(tmp_path):
    site = _site()
    figure = _figure([site])
    for kind in ("curve", "band"):
        for trace in _by_kind(figure, kind, "fit"):
            assert trace.legendgroup == trace.meta["legendkey"] == trace.meta["model"]


def test_every_legend_key_matches_a_real_trace_group(tmp_path):
    site = _site()
    site.extra_curves = [_spreadsheet_curve(site)]
    figure = _figure([site])

    legend = [trace for trace in figure.data
              if trace.meta and trace.meta.get("kind") == "legend"]
    keys = {trace.meta["legendkey"] for trace in legend}
    assert keys >= {"measurements", "linear", "quadratic", "spreadsheet"}
    # a legend key nothing is tagged with would toggle nothing at all
    tagged = {trace.meta.get("legendkey") for trace in figure.data
              if trace.meta and trace.meta.get("legendkey")}
    assert keys <= tagged
    assert all(trace.showlegend for trace in legend)
    # and the data traces must not add their own legend entries
    assert not any(trace.showlegend for trace in figure.data
                   if trace.meta and trace.meta.get("sample"))


def test_curves_use_the_catalog_colour_and_dash(tmp_path):
    site = _site(models=("linear", "quadratic", "exponential"))
    figure = _figure([site])
    for trace in _by_kind(figure, "curve", "fit"):
        key = trace.meta["model"]
        assert trace.line.color == catalog.color(key)
        assert trace.line.dash == catalog.dash(key)
    # least-squares forms must not be solid: shape is what carries the category
    assert catalog.dash("quadratic") != catalog.dash("power_law")
    assert catalog.dash("exponential") != catalog.dash("quadratic")


def test_the_spreadsheet_curve_is_not_styled_as_a_published_rating(tmp_path):
    site = _site()
    site.extra_curves = [_spreadsheet_curve(site)]
    figure = _figure([site])
    spreadsheet = next(trace for trace in _by_kind(figure, "curve", "fit")
                       if trace.meta["model"] == "spreadsheet")
    published = mapview.EXTERNAL_STYLE["published_reference"]
    assert spreadsheet.line.color != published["color"]
    assert spreadsheet.line.dash != published["dash"]
    assert "not fitted here" in spreadsheet.hovertemplate


def test_all_curves_are_clipped_to_the_same_stage_window(tmp_path):
    """Lines that stop at different x read as a difference between the curves."""
    site = _site()
    wide = site.fits[0].curve[["stage_ft", "discharge_cfs"]].copy()
    wide["stage_ft"] = np.linspace(-50, 200, len(wide))       # a published-rating span
    site.extra_curves = [ExternalCurve(key="spreadsheet", label="wide",
                                       kind="spreadsheet", curve=wide,
                                       metrics={"n": 12, "nse": 0.5})]
    figure = _figure([site])
    low, high = mapview._draw_window(site)
    for trace in _by_kind(figure, "curve", "fit"):
        x = np.asarray(trace.x, float)
        assert x.min() >= low - 1e-9 and x.max() <= high + 1e-9, trace.meta


# --- the score table ----------------------------------------------------------

def test_the_score_table_carries_a_definition_on_every_metric(tmp_path):
    from limnotech_rating_curves.evaluate import metrics as metrics_module

    site = _site()
    figure = _figure([site])
    hover = next(trace for trace in _by_kind(figure, "table", "fit")
                 if trace.meta.get("part") == "hover")
    texts = [row[0] for row in hover.customdata]
    for column, header in zip(mapview._FIT_TABLE_GLOSSARY,
                              mapview._FIT_TABLE_HEADERS):
        definition = metrics_module.METRIC_GLOSSARY[column]
        assert any(header in text for text in texts), header
        # the glossary text itself must reach the tooltip, wrapped
        assert any(definition.split()[0] in text for text in texts), column
    assert any("<i>" in text for text in texts), \
        "each tooltip carries an intuition as well as a definition"


def test_the_table_reports_the_spreadsheet_curve_as_its_own_category(tmp_path):
    site = _site()
    site.extra_curves = [_spreadsheet_curve(site)]
    rows = lrc.evaluate.metrics.comparison_rows(site)
    external = [row for row in rows if row["role"] == "external"]
    assert len(external) == 1
    assert external[0]["model"] == "spreadsheet"
    assert external[0]["family"] == "spreadsheet"
    assert external[0]["family"] != "reference", \
        "a typed workbook equation must never be typed as a published USGS rating"
    assert external[0]["form"] == "quadratic"
    refits = [row for row in rows if row["role"] == "spreadsheet_form"]
    assert {row["model"] for row in refits} == {"linear", "quadratic"}
    assert all("refit" in row["model_label"] for row in refits)


# --- the cross-validation pane ------------------------------------------------

def test_cross_validation_pane_carries_fold_stats_and_a_table_per_fold(tmp_path):
    site = _site(models=("linear", "quadratic"))
    result = lrc.cross_validate(site.sample, models=["linear", "quadratic"],
                                scheme="loo")
    assert result.curves
    payload = result.to_map_payload()
    assert payload["scheme"]["name"] == "loo"
    assert payload["headline"] == "pooled"
    first = next(iter(payload["models"].values()))
    assert first["pooled"], "the pooled summary must cross into the map"
    assert first["folds"][0]["test_metrics"] is not None
    assert "n_train" in first["folds"][0]

    figure = _figure([site], cv_data={site.sample_id: payload})
    folds = mapview._site_folds(payload)
    tables = _by_kind(figure, "cvtable", "cv")
    assert {trace.meta["fold"] for trace in tables} == set(folds)
    # every fold's curve must have its band
    curves = {(t.meta["model"], t.meta["fold"])
              for t in _by_kind(figure, "curve", "cv")}
    bands = {(t.meta["model"], t.meta["fold"])
             for t in _by_kind(figure, "band", "cv")}
    assert curves and bands == curves

    text = next(t for t in tables if t.meta.get("part") == "footer"
                and t.meta["fold"] == folds[0])
    joined = " ".join(text.text)
    assert "PSIS-LOO" in joined, "the UI must explain that PSIS-LOO uses all the data"
    assert "pooled" in joined


# --- the injected script ------------------------------------------------------

def test_the_inspector_script_handles_legend_clicks_and_hidden_groups(tmp_path):
    site = _site()
    html = _written(tmp_path, [site]).read_text(encoding="utf-8")
    for needle in ("plotly_legendclick", "plotly_legenddoubleclick",
                   "state.hidden", "legendonly", "plotly_buttonclicked",
                   "plotly_hover"):
        assert needle in html, needle
    # the placeholders must all have been substituted
    for placeholder in ("__RANGES__", "__FOLDS__", "__DEFAULT__", "__HASCV__"):
        assert placeholder not in html, placeholder


def test_the_map_marker_tooltip_quotes_the_predictive_score(tmp_path):
    site = _site()
    figure = _figure([site])
    marker = next(trace for trace in figure.data
                  if trace.type == "scattermap")
    assert "ELPD_LOO" in marker.hovertemplate
    assert "best predictive" in marker.hovertemplate


def test_several_sources_appear_as_separate_map_traces(tmp_path):
    """The USGS gages have to be on the map, not only the workbook sites."""
    magl = _site("magl:SBR-99", coords=(42.3, -83.4))
    gage = _site("usgs_gage:04176356", coords=(41.9, -83.6))
    gage.source = "usgs_gage"
    figure = _figure([magl, gage])
    sources = {trace.name for trace in figure.data if trace.type == "scattermap"}
    assert sources == {mapview.SOURCE_STYLE["magl"][0],
                       mapview.SOURCE_STYLE["usgs_gage"][0]}
