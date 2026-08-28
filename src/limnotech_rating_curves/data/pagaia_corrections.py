"""Two known-wrong things about the pagaia units, corrected explicitly by the caller.

The database reports every water-level variable as ``'millimeter'``. It is wrong: the
values are meters. The exception is the five Geolux model 2300 stations, whose
readings really were millimeters until 2026-08-19 14:30 UTC and are meters after it.

Both corrections live here rather than inside the fetch, so that a caller turning a
pagaia reading into feet does it in three visible steps::

    raw    = pagaia.raw_station_series(station, "distance", start, end)
    meters = pagaia_corrections.to_meters(raw, name)
    feet   = datum.in_units(meters, "ft")

This module is temporary. When the database itself is corrected, delete it and convert
the fetched values directly with :func:`~limnotech_rating_curves.data.datum.in_units`.
It cannot outlive the problem quietly: :func:`to_meters` raises as soon as pagaia stops
claiming millimeters. Note that a cached reading carries the units claimed when it was
fetched, so after the database is fixed that check fires only once the cache is
refreshed - ``refresh=True``, or ``cache.clear("pagaia_series_units")``.
"""

import logging

import pandas as pd

from .. import settings

log = logging.getLogger(__name__)

#: The Geolux millimeter cutover, as a Timestamp. See
#: ``settings.PAGAIA_GEOLUX_MILLIMETER_END``.
_GEOLUX_MILLIMETER_END = pd.Timestamp(settings.PAGAIA_GEOLUX_MILLIMETER_END)

#: The unit pagaia claims for every water-level variable. The values are meters.
#: Spelled several ways because the database has not been consistent about it.
WRONG_CLAIMED_UNITS = ("millimeter", "millimeters", "mm")

#: What the readings are actually in, once this module has been applied.
TRUE_UNITS = "m"

#: Variables this module knows the units of. Discharge is deliberately absent: it is
#: not a length, and its metadata has never been checked against the values.
LENGTH_VARIABLES = ("distance", "stage", "elevation")

_sensor_cache: dict = {}


def geolux_stations(path=None) -> frozenset:
    """Names of the stations whose sensor is a Geolux.

    Read from ``settings.MAGL_GEOLUX_SENSORS_CSV``.

    Parameters
    ----------
    path : path-like, optional
        The CSV. Defaults to :data:`settings.MAGL_GEOLUX_SENSORS_CSV`.

    Returns
    -------
    frozenset
        Station names, carrying the ``_WL`` suffix exactly as the file writes them
        and as pagaia names its stations.
    """
    key = str(settings.MAGL_GEOLUX_SENSORS_CSV if path is None else path)
    if key not in _sensor_cache:
        frame = pd.read_csv(key)
        geolux = frame[frame["Sensor Type"].astype(str).str.strip()
                       .str.lower().str.startswith("geolux")]
        _sensor_cache[key] = frozenset(geolux["Station"].astype(str).str.strip())
    return _sensor_cache[key]


def is_geolux(station: str, path=None) -> bool:
    """Whether `station` carries a Geolux sensor, and so read millimeters."""
    return str(station).strip() in geolux_stations(path)


def to_meters(raw, station: str, *, variable: str = "distance", path=None) -> tuple:
    """Both database corrections, applied to one raw reading.

    Parameters
    ----------
    raw : tuple
        ``(values, claimed_units)`` from
        :func:`~limnotech_rating_curves.data.pagaia.raw_station_series` - the values
        exactly as the database gave them, and the unit it claims for them.
    station : str
        The station's name, with the ``_WL`` suffix, used to look up its sensor.
    variable : str, default 'distance'
        Which variable these readings are. Must be a length; see
        :data:`LENGTH_VARIABLES`.
    path : path-like, optional
        The sensor CSV. Defaults to :data:`settings.MAGL_GEOLUX_SENSORS_CSV`.

    Returns
    -------
    tuple
        ``(values, "m")``, ready for
        :func:`~limnotech_rating_curves.data.datum.in_units`.

    Raises
    ------
    ValueError
        If `variable` is not a length, or if pagaia no longer claims millimeters -
        which means the database has been fixed and this module should be deleted.
    """
    values, claimed = raw

    # Before the units check: an empty reading carries no claimed units, and there is
    # nothing to correct either way.
    if values.empty:
        return values, TRUE_UNITS

    if str(variable).lower() not in LENGTH_VARIABLES:
        raise ValueError(
            f"{station}: {variable!r} is not a length, so its units are not the ones "
            f"this module knows about. It handles {LENGTH_VARIABLES}.")

    if claimed is None or str(claimed).lower() not in WRONG_CLAIMED_UNITS:
        raise ValueError(
            f"{station}: pagaia reports {claimed!r} for {variable!r}, not "
            f"millimeters. The database units look fixed, so these corrections no "
            f"longer apply and would make the values 1000x wrong. Delete "
            f"limnotech_rating_curves.data.pagaia_corrections and convert with "
            f"datum.in_units((values, {claimed!r}), units) instead.")

    # The first correction is to the metadata, not the numbers: the claim of
    # millimeters is discarded and the values are taken as the meters they are. No
    # arithmetic at all.
    if is_geolux(station, path):
        # The second correction is to the numbers, and only to the readings this
        # station took before the cutover. A whole-series test would be wrong: a
        # window can straddle the cutover, and several already do.
        values = values.astype(float).copy()
        before = values.index < _GEOLUX_MILLIMETER_END
        if before.any():
            values.loc[before] = values.loc[before] / 1000.0
            log.info("%s is a Geolux station: divided %d of %d reading(s) before %s "
                     "by 1000, they were recorded in millimeters", station,
                     int(before.sum()), len(values), _GEOLUX_MILLIMETER_END)
    elif station not in geolux_stations(path):
        log.info("%s is not listed in %s; reading it as meters like the Vega stations",
                 station, settings.MAGL_GEOLUX_SENSORS_CSV.name)

    return values.rename(station), TRUE_UNITS
