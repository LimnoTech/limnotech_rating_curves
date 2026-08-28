"""The MAGL spreadsheet timeseries record.

Before the stations were reporting to the pagaia database, the MAGL distance-to-surface
record lived in a tree of Excel workbooks, which were scraped once into
``magl_spreadsheet_timeseries.pkl`` - a list of ``("WL", station, frame)`` tuples,
one frame per station, in **feet**. That pickle is still the only source for the
earliest part of every station's record, and the only source at all for a station
pagaia does not carry, so it is read here rather than retired.

Two things come out of it:

* the distance record itself (:func:`distance_ft`), which
  :func:`~limnotech_rating_curves.data.magl.distance_record_ft` splices pagaia's
  tail onto;
* the control-point offset (:func:`control_point_offsets`), which is measurable only
  here, because these frames carry both the raw distance reading and the
  spreadsheet's own derived water-surface elevation - and the difference between them
  is the gap between the surveyed benchmark and the radar's own datum.

The survey itself (the control-point elevations and the sensor moves) is CSV and lives
in :mod:`~limnotech_rating_curves.data.magl`; this module reads it from there.

Not to be confused with :mod:`~limnotech_rating_curves.data.magl_workbook`, which
parses the *field flow* workbooks (``flow@<sensor>.xlsx``) for their typed rating
coefficients. This module is only the historical stage record.
"""

from pathlib import Path

import pandas as pd

from .. import settings
from . import datum as datum_module

#: Column in the spreadsheet frames holding its own derived water-surface elevation,
#: used only to measure the control-point offset - never as the stage a rating is
#: fitted to.
_ELEVATION_COLUMN = "Water Elevation (NAVD88)"

_frame_cache: dict = {}
_offset_cache: dict = {}


def frames(path=None) -> dict:
    """Per-station frames from the spreadsheet pickle. Memoized; the pickle is large.

    Parameters
    ----------
    path : path-like, optional
        The pickle. Defaults to ``settings.MAGL_SPREADSHEET_PICKLE``.

    Returns
    -------
    dict
        ``{station: DataFrame}`` for the ``"WL"`` entries, keyed by station name with
        the ``_WL`` suffix.
    """
    path = Path(settings.MAGL_SPREADSHEET_PICKLE if path is None else path)
    key = str(path)
    if key not in _frame_cache:
        payload = pd.read_pickle(path)
        _frame_cache[key] = {
            element[1]: element[2] for element in payload
            if isinstance(element, tuple) and len(element) == 3 and element[0] == "WL"}
    return _frame_cache[key]


def clean_series(series: pd.Series) -> pd.Series:
    """tz-naive, sorted, de-duplicated float series at one datetime resolution."""
    series = series.dropna().astype(float).copy()
    series.index = datum_module._as_naive_index(series.index)
    return series[~series.index.duplicated(keep="first")].sort_index()


def distance_ft(station: str, path=None,
                quality_controlled: bool = True) -> pd.Series:
    """Distance-to-surface from the spreadsheet record, in feet.

    Parameters
    ----------
    station : str
        Station name with ``_WL``.
    path : path-like, optional
        The pickle. Defaults to ``settings.MAGL_SPREADSHEET_PICKLE``.
    quality_controlled : bool, default True
        Keep only rows the spreadsheet flagged as passing QA/QC.

    Returns
    -------
    pandas.Series
        Native (sub-daily) resolution, feet. Empty if the station is not in the
        pickle.
    """
    station_frames = frames(path)
    if station not in station_frames:
        return pd.Series(dtype=float, name=station)
    frame = station_frames[station]
    distance = frame["distance"]
    if quality_controlled and "QAQC" in frame.columns:
        distance = distance[frame["QAQC"] == 0]
    return clean_series(distance).rename(station)


# ---------------------------------------------------------------------------
# the control-point offset (control-point datum vs sensor datum)
# ---------------------------------------------------------------------------


def control_point_offsets(path=None, quality_controlled: bool = True) -> pd.DataFrame:
    """Per-station gap between the surveyed control point and the sensor's own datum.

    The surveyed control point is a benchmark; the radar measures from its own
    face. The constant gap between them is what separates the two datums, and it is
    recoverable because the spreadsheet carries both the distance reading and its
    own derived water-surface elevation: their sum is the sensor's reference
    elevation, and the offset is the control point minus that.

    The sum is taken over the window *before the first sensor move* (the whole
    record for a station that never moved), because a move changes the reference and
    would otherwise smear the estimate.

    Parameters
    ----------
    path : path-like, optional
        The spreadsheet pickle.
    quality_controlled : bool, default True
        Apply the spreadsheet's QA/QC flag.

    Returns
    -------
    pandas.DataFrame
        Indexed by station, with ``control_point_ft``, ``sensor_reference_ft``,
        ``offset_ft``, ``n_baseline`` and ``suspect`` - the last flagging an offset
        larger than ``settings.MAGL_OFFSET_SUSPECT_FT``.
    """
    key = f"offsets:{path}:{quality_controlled}"
    if key in _offset_cache:
        return _offset_cache[key]

    # Imported here, not at module scope: magl imports this module, and the survey
    # CSVs it owns are the only thing needed back from it.
    from .magl import control_point_elevations, sensor_moves

    elevations = control_point_elevations()
    moves = sensor_moves()
    rows = {}
    for station, frame in frames(path).items():
        if (station not in elevations
                or _ELEVATION_COLUMN not in frame.columns):
            continue
        sensor_reference = frame["distance"] + frame[_ELEVATION_COLUMN]
        if quality_controlled and "QAQC" in frame.columns:
            sensor_reference = sensor_reference[frame["QAQC"] == 0]
        sensor_reference = clean_series(sensor_reference)
        if sensor_reference.empty:
            continue
        events = moves.get(station, [])
        baseline = (sensor_reference[sensor_reference.index < events[0][0]]
                    if events else sensor_reference)
        if baseline.empty:
            baseline = sensor_reference
        reference = float(baseline.median())
        control_point = elevations[station]
        rows[station] = {
            "control_point_ft": control_point,
            "sensor_reference_ft": reference,
            "offset_ft": control_point - reference,
            "n_baseline": int(len(baseline)),
            "suspect": abs(control_point - reference) > settings.MAGL_OFFSET_SUSPECT_FT,
        }
    table = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    table.index.name = "station"
    _offset_cache[key] = table
    return table


def control_point_offset(station: str, **kwargs) -> float:
    """One station's control-point offset, in feet.

    Parameters
    ----------
    station : str
        Station name with ``_WL``.
    **kwargs
        Forwarded to :func:`control_point_offsets`.

    Returns
    -------
    float
        Feet to subtract from a control-point-datum elevation to reach the sensor
        datum.

    Raises
    ------
    KeyError
        If the station is not in the spreadsheet record, so the offset cannot be
        measured.
    """
    offsets = control_point_offsets(**kwargs)
    if station not in offsets.index:
        raise KeyError(f"no control-point offset for {station!r} - it has no "
                       f"spreadsheet record to measure one from")
    return float(offsets.loc[station, "offset_ft"])
