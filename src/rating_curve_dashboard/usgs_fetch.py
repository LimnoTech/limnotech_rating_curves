"""
Live USGS NWIS data access for the rating-curve dashboard.

Calls the `dataretrieval` package's Water Data API client directly rather than
going through `limnotech_rating_curves`, whose package import pulls in
PyMC/PyTensor (see app.py's module docstring for why that dependency is kept
out of this process). `dataretrieval` is a lightweight, official USGS client
with no such baggage.

Three kinds of stage-discharge data can come back from a site:

- ``FIELD_MEASUREMENTS`` - discrete gaugings (a person/sensor reading stage and
  discharge on one visit). These are independent observations of the rating
  and are what a rating curve should be fit to.
- ``DAILY`` / ``INSTANTANEOUS`` - the gage's own continuous record, paired by
  timestamp. Useful to look at, but the discharge in these series was already
  computed *from* the stage through the gage's existing rating, so fitting to
  them mostly reproduces that rating rather than deriving an independent one.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

FIELD_MEASUREMENTS = "field_measurements"
DAILY = "daily"
INSTANTANEOUS = "instantaneous"

PARAM_DISCHARGE = "00060"
PARAM_STAGE = "00065"


def normalize_site_id(site: str) -> str:
    """A USGS site id as a bare, zero-padded number (accepts a pasted ``USGS-...`` id too)."""
    text = str(site).strip()
    if text.upper().startswith("USGS-"):
        text = text[5:]
    return text.zfill(8) if len(text) < 8 else text


def _location_id(site: str) -> str:
    return f"USGS-{normalize_site_id(site)}"


def _interval(start, end) -> str:
    return f"{start}T00:00:00Z/{end}T23:59:59Z"


@st.cache_data(show_spinner=False, ttl=3600)
def site_lookup(site: str) -> dict | None:
    """Station name/state/drainage area for a site id, or None if the id isn't found."""
    from dataretrieval import waterdata
    locs, _ = waterdata.get_monitoring_locations(monitoring_location_id=_location_id(site))
    if locs.empty:
        return None
    row = locs.iloc[0]
    return {
        "name": row.get("monitoring_location_name"),
        "state": row.get("state_name"),
        "drainage_area_sqmi": row.get("drainage_area"),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def available_series(site: str) -> pd.DataFrame:
    """The stage/discharge data types this site actually publishes, and their period of record.

    One row per option to offer in the dashboard's data-type dropdown. A kind is
    only included when the site has real data for it, so the dropdown always
    reflects what's actually available on USGS's end rather than a fixed list.

    Returns
    -------
    pandas.DataFrame
        Columns ``kind``, ``label``, ``begin``, ``end``. Empty if the site
        publishes none of the three kinds this dashboard understands.
    """
    from dataretrieval import waterdata

    loc = _location_id(site)
    rows = []

    fm_meta, _ = waterdata.get_field_measurements_metadata(monitoring_location_id=loc)
    fm_meta = fm_meta[fm_meta["reading_type"].isin(["Discharge", "MeanGageHeight"])]
    if not fm_meta.empty:
        rows.append({
            "kind": FIELD_MEASUREMENTS,
            "label": "Field measurements (discrete gaugings) - recommended for fitting",
            "begin": fm_meta["begin"].min(),
            "end": fm_meta["end"].max(),
        })

    ts_meta, _ = waterdata.get_time_series_metadata(monitoring_location_id=loc)
    if not ts_meta.empty:
        for kind, computation_ids, kind_label in (
            (DAILY, {"Mean"}, "Daily values (mean stage & discharge)"),
            (INSTANTANEOUS, {"Instantaneous"}, "Instantaneous values (continuous stage & discharge)"),
        ):
            subset = ts_meta[ts_meta["computation_identifier"].isin(computation_ids)
                              & ts_meta["parameter_code"].isin([PARAM_DISCHARGE, PARAM_STAGE])]
            has_both = {PARAM_DISCHARGE, PARAM_STAGE} <= set(subset["parameter_code"])
            if has_both:
                rows.append({
                    "kind": kind,
                    "label": f"{kind_label} - the gage's own continuous record, "
                              "not independent of its existing rating",
                    "begin": subset["begin"].min(),
                    "end": subset["end"].max(),
                })

    columns = ["kind", "label", "begin", "end"]
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_field_measurements(site: str, start: str, end: str) -> pd.DataFrame:
    """Discrete stage-discharge gaugings for one site over [start, end] (YYYY-MM-DD)."""
    from dataretrieval import waterdata

    raw, _ = waterdata.get_field_measurements(
        monitoring_location_id=_location_id(site), time=_interval(start, end))
    readings = raw[raw["reading_type"].isin(["Discharge", "MeanGageHeight"])]
    wide = readings.pivot_table(
        index=["field_visit_id", "time"], columns="reading_type",
        values="value", aggfunc="first").reset_index()
    wide = wide.rename(columns={"Discharge": "discharge_cfs", "MeanGageHeight": "stage_ft"})
    wide = wide.dropna(subset=["discharge_cfs", "stage_ft"])
    return (wide[["time", "stage_ft", "discharge_cfs"]]
            .sort_values("time").reset_index(drop=True))


def _paired_continuous(get_fn, site: str, start: str, end: str) -> pd.DataFrame:
    """Stage and discharge from the same continuous-style service, joined on time."""
    loc = _location_id(site)
    interval = _interval(start, end)
    discharge, _ = get_fn(monitoring_location_id=loc, parameter_code=PARAM_DISCHARGE, time=interval)
    stage, _ = get_fn(monitoring_location_id=loc, parameter_code=PARAM_STAGE, time=interval)
    if discharge.empty or stage.empty:
        return pd.DataFrame(columns=["time", "stage_ft", "discharge_cfs"])
    merged = pd.merge(
        stage[["time", "value"]].rename(columns={"value": "stage_ft"}),
        discharge[["time", "value"]].rename(columns={"value": "discharge_cfs"}),
        on="time", how="inner")
    return (merged.dropna(subset=["stage_ft", "discharge_cfs"])
            .sort_values("time").reset_index(drop=True))


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_daily(site: str, start: str, end: str) -> pd.DataFrame:
    """Daily mean stage and discharge for one site over [start, end], paired by date."""
    from dataretrieval import waterdata
    return _paired_continuous(waterdata.get_daily, site, start, end)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_continuous(site: str, start: str, end: str) -> pd.DataFrame:
    """Instantaneous (~15-min) stage and discharge for one site over [start, end], paired by time."""
    from dataretrieval import waterdata
    return _paired_continuous(waterdata.get_continuous, site, start, end)


def fetch(site: str, start: str, end: str, kind: str) -> pd.DataFrame:
    """Stage/discharge pairs for a site, dispatching on the data-type ``kind``."""
    if kind == FIELD_MEASUREMENTS:
        return fetch_field_measurements(site, start, end)
    if kind == DAILY:
        return fetch_daily(site, start, end)
    if kind == INSTANTANEOUS:
        return fetch_continuous(site, start, end)
    raise ValueError(f"unknown data type {kind!r}")
