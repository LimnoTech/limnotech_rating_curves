import logging

import numpy as np
import pandas as pd

from ..support import cache
from . import datum as datum_module
from .. import settings
from ..core import Sample

log = logging.getLogger(__name__)

#: How to install the optional client, quoted in the error when it is missing.
INSTALL_HINT = ("pip install "
                "'fb-pagaia @ git+https://github.com/LimnoTech/fb-pagaia.git'")


def installed() -> bool:
    """True when the optional ``fb_pagaia`` client is importable.

    The client is an extra rather than a dependency: it reaches a LimnoTech database
    over the VPN, which is of no use to anyone fitting their own measurements or USGS
    field measurements. Everything in this module that takes a station object works on
    what the object does, not on its type, so only the loaders that *build* a station
    need it.
    """
    try:
        require()
    except ImportError:
        return False
    return True


def require():
    """Import ``fb_pagaia``, or say how to install it.

    Returns
    -------
    module
        The ``fb_pagaia`` package.

    Raises
    ------
    ImportError
        When it is not installed, naming the extra and the install command rather
        than leaving a bare ``No module named 'fb_pagaia'``.
    """
    try:
        import fb_pagaia
    except ImportError as exc:
        raise ImportError(
            "this needs the optional fb_pagaia client, which reads the LimnoTech "
            "ODM2 database over the VPN. Install it with the 'pagaia' extra "
            "(pip install 'limnotech-rating-curves[pagaia]') or directly:\n"
            f"    {INSTALL_HINT}") from exc
    return fb_pagaia


#: Substrings identifying a station variable, by what the reading means.
VARIABLE_TERMS = {
    "distance": ("distance",),
    "stage": ("stage", "gage height", "gageheight", "water level", "waterlevel"),
    "elevation": ("elevation", "water elevation"),
    "discharge": ("discharge", "flow"),
}

#: Largest acceptable gap between a discharge measurement and the station reading
#: used for it.
DEFAULT_MATCH_TOLERANCE = "3h"

#: Units assumed for a length variable whose metadata does not say. The stations
#: record lengths in millimeters, so a reading that arrives unlabelled is read that
#: way rather than guessed at from its magnitude.
FALLBACK_SOURCE_UNITS = "mm"


def station_coordinates(stations) -> dict:
    """Coordinates of pagaia stations, for the map.

    Parameters
    ----------
    stations : Station or Network or sequence of Station
        Whatever you have.

    Returns
    -------
    dict
        ``{station name: (latitude, longitude)}``, omitting stations with no
        coordinates recorded.
    """
    if hasattr(stations, "stations"):
        candidates = list(getattr(stations, "stations").values())
    elif hasattr(stations, "latlon"):
        candidates = [stations]
    else:
        candidates = list(stations)
    coordinates = {}
    for candidate in candidates:
        latlon = getattr(candidate, "latlon", None)
        name = _station_name(candidate)
        if latlon and all(value is not None for value in latlon):
            coordinates[name] = (float(latlon[0]), float(latlon[1]))
    return coordinates


def _station_name(obj) -> str:
    """A station's display name, whichever attribute it carries it in."""
    for attribute in ("samplingfeaturename", "name", "samplingfeaturecode",
                      "station_id"):
        value = getattr(obj, attribute, None)
        if value:
            return str(value)
    return str(obj)


def station_series(pagaia_station, variable: str = "stage", start=None, end=None,
                   units: str = "ft", source_units: str = None,
                   refresh: bool = False) -> pd.Series:
    """One variable's timeseries from a pagaia station.

    Parameters
    ----------
    pagaia_station : fb_pagaia.core.Station
        The station.
    variable : str, default 'stage'
        Which variable to pull: one of the keys of ``VARIABLE_TERMS``
        (``"stage"``, ``"distance"``, ``"elevation"``, ``"discharge"``), or a
        substring to match against the station's own variable names.
    start, end : str or datetime, optional
        Window to fetch. Both are required by the API, so leaving them out fetches
        nothing and returns an empty series.
    units : {'ft', 'm', 'cm', 'mm'}, default 'ft'
        Units to return. The reading's own units come from the station's variable
        metadata and are converted to these.
    source_units : str, optional
        Override the units the station reports the variable in. Use this only when
        the metadata is known to be wrong; by default it is believed.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.Series
        Indexed by tz-naive timestamp, ascending, duplicates dropped. Empty when
        the station does not record that variable, or when the server is
        unreachable - which is logged rather than raised, so a caller with another
        source to fall back on can use it.
    """
    name = _station_name(pagaia_station)
    if start is None or end is None:
        log.info("station_series(%s) needs both start and end; returning empty", name)
        return pd.Series(dtype=float, name=name)

    def build():
        """Fetch the window, pull out the variable, and record its units."""
        try:
            data, metadata = pagaia_station.get_timeseries_data(
                start=pd.Timestamp(start).to_pydatetime(),
                end=pd.Timestamp(end).to_pydatetime(),
                return_as="dataframe_multi_index")
        except Exception as exc:  # noqa: BLE001 - server down / off VPN
            log.warning("pagaia unavailable for %s (%s: %s)", name,
                        type(exc).__name__, exc)
            return pd.Series(dtype=float, name=name), None
        series = _select_variable(data, variable)
        if series is None or series.empty:
            log.info("station %s records no %r variable in this window", name, variable)
            return pd.Series(dtype=float, name=name), None
        return series, _reported_units(metadata, series.name)

    series, reported = cache.cached("pagaia_series_units",
                                    (name, variable, str(start), str(end)),
                                    build, refresh)
    if series.empty:
        return series
    from_units = source_units or reported or FALLBACK_SOURCE_UNITS
    if source_units is None and reported is None:
        log.info("%s reports no units for %r; reading it as %s",
                 name, variable, FALLBACK_SOURCE_UNITS)
    factor = datum_module._length_factor(from_units, units)
    return (series.astype(float) * factor).rename(name)


def _reported_units(metadata, column) -> "str | None":
    """The units a station's variable metadata gives for one column.

    ODM2 carries the unit name in the variable's extension properties rather than
    in a column of its own, so it is read from there and handed to
    :func:`~limnotech_rating_curves.data.datum._length_factor` by name.
    """
    if metadata is None or column is None or not hasattr(metadata, "columns"):
        return None
    try:
        properties = metadata[column].get("extensionproperties")
    except Exception:  # noqa: BLE001 - column absent, or a shape we do not know
        return None
    if isinstance(properties, pd.Series):
        properties = properties.iloc[0]
    if not isinstance(properties, dict):
        return None
    reported = properties.get("variableunitsUbidots")
    if not reported:
        return None
    reported = str(reported).lower()
    return reported if reported in datum_module._LENGTH_IN_MM else None


def _select_variable(data: pd.DataFrame, variable: str) -> "pd.Series | None":
    """Pick the column of a pagaia multi-index frame matching `variable`."""
    if data is None or data.empty:
        return None
    terms = VARIABLE_TERMS.get(str(variable).lower(), (str(variable).lower(),))
    if "variableterm" in (data.columns.names or []):
        names = data.columns.get_level_values("variableterm")
    else:
        names = data.columns.get_level_values(0)
    for position, name in enumerate(names):
        if any(term in str(name).lower() for term in terms):
            column = data.iloc[:, position]
            column.index = datum_module._as_naive_index(column.index)
            column = column.dropna().astype(float)
            return column[~column.index.duplicated(keep="first")].sort_index()
    return None


def station_sample(pagaia_station, discharge, *, stage_datum=None,
                   reference_elevation=None, variable: str = "stage",
                   tolerance: str = DEFAULT_MATCH_TOLERANCE, start=None, end=None,
                   units: str = "ft", source_units: str = None,
                   refresh: bool = False) -> Sample:
    """Build a fittable :class:`Sample` from a pagaia station and field discharge.

    Parameters
    ----------
    pagaia_station : fb_pagaia.core.Station
        The station whose readings supply the stage.
    discharge : pandas.DataFrame or pandas.Series
        The field discharge measurements: a DataFrame with a time column and a
        discharge column, or a Series of discharge indexed by time.
    stage_datum : optional
        How to get from the station's readings to gage height - the elevation of
        gage height zero, ``"lowest"``, a timeseries of reference elevations, or a
        :class:`~limnotech_rating_curves.data.datum.StageDatum`. Leave it out if the
        station already reports gage height. See
        :mod:`limnotech_rating_curves.data.datum`.
    reference_elevation : float or pandas.Series, optional
        Only for stations that report **distance down to the water**: the elevation
        of the point that distance is measured from. Given this, the readings are
        converted to water-surface elevations first, and `stage_datum` then takes
        those to gage height. Pass a Series (or the output of
        :func:`~limnotech_rating_curves.data.datum.stepwise_reference_elevation`) when
        the sensor has been moved.
    variable : str, default 'stage'
        Which station variable holds the reading. Use ``"distance"`` together with
        `reference_elevation`.
    tolerance : str, default '3h'
        Largest acceptable gap between a discharge measurement and the station
        reading matched to it.
    start, end : str or datetime, optional
        Window of station data to fetch. Defaults to two days either side of the
        discharge measurements.
    units : {'ft', 'm', 'cm', 'mm'}, default 'ft'
        Units the station readings are returned in before conversion.
    source_units : str, optional
        Override the units the station reports its readings in. By default the
        variable's own metadata is believed.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    Sample
        With ``source="pagaia"`` and a ``datum_note`` recording every conversion
        applied. Measurements with no station reading in tolerance are dropped and
        counted in the log.

    Raises
    ------
    KeyError
        If `discharge` has no recognizable time or discharge column.

    Examples
    --------
    >>> sample = station_sample(station, field_discharge,
    ...                         stage_datum="lowest")          # doctest: +SKIP
    >>> sample.compare(models="all").metrics                    # doctest: +SKIP
    """
    times, flows = _discharge_measurements(discharge)
    if len(times) == 0:
        return Sample(stage_ft=[], discharge_cfs=[],
                      site_id=_station_name(pagaia_station), source="pagaia",
                      skipped="no discharge measurements given")

    if start is None:
        start = (times.min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    if end is None:
        end = (times.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    readings = station_series(pagaia_station, variable=variable, start=start, end=end,
                             units=units, source_units=source_units, refresh=refresh)
    name = _station_name(pagaia_station)
    if readings.empty:
        return Sample(stage_ft=[], discharge_cfs=[], site_id=name, source="pagaia",
                      skipped=f"no {variable} readings for {name} in {start}..{end}")

    notes = []
    if reference_elevation is not None:
        readings = datum_module.distance_to_stage(
            readings, reference_elevation, distance_units=units, stage_units=units)
        notes.append("distance-to-surface converted to water-surface elevation")

    matched = datum_module.stage_at_times(readings, times, tolerance=tolerance)
    frame = pd.DataFrame({"time": times, "discharge_cfs": flows}).set_index("time")
    frame = frame.join(matched[["stage_ft", "lag_min"]])
    usable = frame.dropna(subset=["stage_ft"]).reset_index()
    if len(usable) < len(frame):
        log.info("%s: %d of %d discharge measurement(s) had no reading within %s",
                 name, len(frame) - len(usable), len(frame), tolerance)
    if usable.empty:
        return Sample(stage_ft=[], discharge_cfs=[], site_id=name, source="pagaia",
                      skipped=(f"no discharge measurement had a {variable} reading "
                               f"within {tolerance}"))

    notes.append(f"stage matched to the nearest reading within {tolerance} "
                 f"(median lag {usable['lag_min'].median():.0f} min)")
    sample = Sample.of(usable, site_id=name, source="pagaia",
                       stage_datum=stage_datum, datum_note="; ".join(notes))
    log.info("%s: %d fittable measurement(s), stage %.3f-%.3f ft", name, len(sample),
             *sample.stage_range)
    return sample


def _discharge_measurements(discharge):
    """Times and discharge values from a DataFrame or a time-indexed Series."""
    from ..core import DISCHARGE_COLUMNS, TIME_COLUMNS, _pick_column

    if isinstance(discharge, pd.Series):
        series = discharge.dropna().astype(float)
        return datum_module._as_naive_index(series.index), series.to_numpy(float)

    frame = pd.DataFrame(discharge)
    discharge_column = _pick_column(frame, DISCHARGE_COLUMNS, "discharge")
    time_column = _pick_column(frame, TIME_COLUMNS, "time")
    usable = frame[[time_column, discharge_column]].dropna()
    times = datum_module._as_naive_index(usable[time_column])
    values = pd.to_numeric(usable[discharge_column], errors="coerce").to_numpy(float)
    keep = np.isfinite(values) & ~times.isna()
    return times[keep], values[keep]
