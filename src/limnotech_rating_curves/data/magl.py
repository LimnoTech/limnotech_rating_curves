import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..helpers import cache
from ..model_selection import crossval
from . import datum as datum_module, magl_spreadsheets, usgs
from .. import settings
from ..core import Sample
from ..site import SiteRating
from .usgs import gage_sample  # noqa: F401 - re-exported as the reference case

log = logging.getLogger(__name__)


_registry_cache: dict = {}
_survey_cache: dict = {}


def _discharge_csv(name: str) -> Path:
    """Locate one of the MAGL discharge exports, under the data dir or beside it."""
    candidates = [settings.MAGL_DATA_DIR / name, settings.DATA_DIR / name,
                  Path(name)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def discharge_csv_path() -> Path:
    """Path to the detailed manual-discharge export.

    One row per measurement with its exact timestamp, the instrument used and the
    source file, produced by scraping the Flow-Manual spreadsheet tree.

    Returns
    -------
    pathlib.Path
    """
    return _discharge_csv("magl_discharge.csv")


def discharge_report_csv_path() -> Path:
    """Path to the curated discharge report.

    One row per measurement with the field-recorded water-surface elevation
    alongside the discharge - the reviewed version of the same measurements, and
    the one to prefer, because its stage was read in the field rather than matched
    from the sensor record.

    Returns
    -------
    pathlib.Path
    """
    return _discharge_csv("magl_discharge_report.csv")


# ---------------------------------------------------------------------------
# the site registry
# ---------------------------------------------------------------------------

#: The registry's files, and the columns each one must carry.
REGISTRY_FILES = {
    "clusters.csv": ("cluster", "name"),
    "colocated_pairs.csv": ("station", "gage"),
    "standalone_gages.csv": ("gage", "cluster"),
    "special_stations.csv": ("station", "pickle", "entry_key"),
}


def _registry_table(directory: Path, name: str) -> pd.DataFrame:
    """One registry CSV, read as text so gage ids keep their leading zeros.

    A file that is not there is an empty table rather than an error: a deployment
    with no co-located pairs is a legitimate deployment, and the sections are
    independent.
    """
    path = directory / name
    if not path.exists():
        log.debug("site registry: no %s in %s, treating it as empty", name, directory)
        return pd.DataFrame(columns=list(REGISTRY_FILES[name]))
    frame = pd.read_csv(path, dtype=str, comment="#").fillna("")
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in REGISTRY_FILES[name]
               if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {', '.join(missing)}; "
                         f"expected {', '.join(REGISTRY_FILES[name])}")
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    return frame[frame[REGISTRY_FILES[name][0]] != ""]


def _gage_id(value: str, source: Path) -> str:
    """A gage id, warning when a spreadsheet has eaten its leading zero.

    Every USGS site id in this network begins with a zero, so one that does not has
    almost certainly been through a spreadsheet that read the column as a number.
    Warned about rather than repaired, because guessing how many zeros were lost
    would be worse than saying so.
    """
    identifier = str(value).strip()
    if identifier and identifier.isdigit() and not identifier.startswith("0"):
        log.warning("gage id %r in %s has no leading zero - a spreadsheet editor "
                    "strips them when it reads the column as a number. Restore it "
                    "and keep the column formatted as text", identifier, source.name)
    return identifier


def site_registry(path=None, refresh: bool = False) -> dict:
    """Load the MAGL site registry from its CSV files.

    A *cluster* is a stream-network group labeled by the station-name prefix, so
    ``SR-08_WL`` belongs to cluster ``SR``. USGS gages inherit a cluster either
    from a co-located station or from an explicit mapping in
    ``standalone_gages.csv``.

    Parameters
    ----------
    path : path-like, optional
        The directory holding the registry CSVs - see ``REGISTRY_FILES`` for the
        files and their columns. Defaults to ``settings.MAGL_SITES_DIR``, which the
        ``LRC_MAGL_SITES_DIR`` environment variable can point elsewhere.
    refresh : bool, default False
        Re-read the files instead of using the in-process copy.

    Returns
    -------
    dict
        ``clusters`` (metadata per label), ``colocated`` (``{station: gage}``),
        ``standalone_gages`` (``{gage: cluster}``) and ``special_stations``
        (``{station: (pickle, entry_key)}``).

    Raises
    ------
    FileNotFoundError
        When the directory itself is absent - which usually means the data has not
        been placed where ``settings.DATA_DIR`` points.
    """
    directory = Path(settings.MAGL_SITES_DIR if path is None else path)
    key = str(directory)
    if key in _registry_cache and not refresh:
        return _registry_cache[key]
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no MAGL site registry at {directory}. It holds "
            f"{', '.join(REGISTRY_FILES)}; point LRC_MAGL_SITES_DIR or LRC_DATA_DIR "
            f"at where the data actually is")

    clusters = _registry_table(directory, "clusters.csv")
    pairs = _registry_table(directory, "colocated_pairs.csv")
    standalone = _registry_table(directory, "standalone_gages.csv")
    special = _registry_table(directory, "special_stations.csv")

    pairs_path = directory / "colocated_pairs.csv"
    standalone_path = directory / "standalone_gages.csv"
    registry = {
        "clusters": {row["cluster"]: {"name": row["name"]}
                     for _, row in clusters.iterrows()},
        "colocated": {row["station"]: _gage_id(row["gage"], pairs_path)
                      for _, row in pairs.iterrows()},
        "standalone_gages": {_gage_id(row["gage"], standalone_path): row["cluster"]
                             for _, row in standalone.iterrows()},
        "special_stations": {row["station"]: (row["pickle"], int(row["entry_key"]))
                             for _, row in special.iterrows()},
    }
    _registry_cache[key] = registry
    return registry


def clusters() -> list:
    """Every cluster label in the registry, sorted.

    Returns
    -------
    list of str
    """
    registry = site_registry()
    labels = {station_cluster(station) for station in registry["colocated"]}
    labels |= set(registry["standalone_gages"].values())
    labels |= {station_cluster(station) for station in registry["special_stations"]}
    labels |= set(registry["clusters"])
    return sorted(labels)


def station_cluster(station: str) -> str:
    """The cluster a station name belongs to.

    Parameters
    ----------
    station : str
        Station name, with or without the ``_WL`` suffix.

    Returns
    -------
    str
        The prefix before the first hyphen, e.g. ``"SR"`` for ``"SR-08_WL"``.
    """
    return str(station).split("-")[0]


def gage_cluster(gage) -> "str | None":
    """The cluster a USGS gage belongs to.

    Parameters
    ----------
    gage : str
        USGS gage number.

    Returns
    -------
    str or None
        Inherited from a co-located station if there is one, else the explicit
        mapping, else None.
    """
    registry = site_registry()
    gage = str(gage)
    for station, paired in registry["colocated"].items():
        if paired == gage:
            return station_cluster(station)
    return registry["standalone_gages"].get(gage)


def colocated_gage(station: str) -> "str | None":
    """The USGS gage co-located with a MAGL station, if any.

    Parameters
    ----------
    station : str
        Station name, with or without ``_WL``.

    Returns
    -------
    str or None
    """
    base = str(station).split("_")[0]
    return site_registry()["colocated"].get(base)


def colocated_station(gage) -> "str | None":
    """The MAGL station co-located with a USGS gage, if any.

    Parameters
    ----------
    gage : str
        USGS gage number.

    Returns
    -------
    str or None
    """
    for station, paired in site_registry()["colocated"].items():
        if paired == str(gage):
            return station
    return None


def colocated_pairs() -> list:
    """Every co-located ``(station, gage)`` pair.

    Returns
    -------
    list of tuple
    """
    return [(station, gage) for station, gage
            in site_registry()["colocated"].items()]


def all_gages() -> list:
    """Every USGS gage in the registry, co-located and standalone.

    Returns
    -------
    list of str
        Sorted, deduplicated.
    """
    registry = site_registry()
    gages = set(registry["colocated"].values()) | set(registry["standalone_gages"])
    return sorted(gages)


def gages_in_cluster(cluster: str) -> list:
    """Every USGS gage assigned to one cluster.

    Parameters
    ----------
    cluster : str
        Cluster label.

    Returns
    -------
    list of str
    """
    return [gage for gage in all_gages() if gage_cluster(gage) == cluster]


# ---------------------------------------------------------------------------
# the survey: control points and sensor moves
# ---------------------------------------------------------------------------

def control_point_elevations(path=None) -> dict:
    """Surveyed control-point elevation per station.

    Parameters
    ----------
    path : path-like, optional
        The CSV. Defaults to ``settings.MAGL_CONTROL_POINTS_CSV``.

    Returns
    -------
    dict
        ``{station name with _WL: elevation in NAVD88 feet}``. The CSV stores bare
        station names; ``_WL`` is appended so the keys match sensor names.
    """
    path = Path(settings.MAGL_CONTROL_POINTS_CSV if path is None else path)
    key = f"control_points:{path}"
    if key not in _survey_cache:
        table = pd.read_csv(path)
        elevation_column = next(column for column in table.columns
                                if "elevation" in column.lower())
        _survey_cache[key] = {f"{str(station).strip()}_WL": float(elevation)
                              for station, elevation
                              in zip(table["station"], table[elevation_column])}
    return _survey_cache[key]


def sensor_moves(path=None) -> dict:
    """Recorded changes to each station's control-point elevation.

    A sensor that is raised or lowered changes what its distance readings mean, so
    each move has to be applied from its own timestamp onward. Several moves at one
    station accumulate.

    Parameters
    ----------
    path : path-like, optional
        The CSV. Defaults to ``settings.MAGL_SENSOR_MOVES_CSV``.

    Returns
    -------
    dict
        ``{station name with _WL: [(timestamp, change in feet), ...]}``, each list
        in time order.
    """
    path = Path(settings.MAGL_SENSOR_MOVES_CSV if path is None else path)
    key = f"moves:{path}"
    if key not in _survey_cache:
        table = pd.read_csv(path)
        moves: dict = {}
        for _, row in table.iterrows():
            station = f"{str(row['station']).strip()}_WL"
            when = pd.to_datetime(str(row["timestamp"]).strip(),
                                  format="%m/%d/%y %H:%M")
            moves.setdefault(station, []).append(
                (when, float(str(row["elevation_change_ft"]).strip())))
        for events in moves.values():
            events.sort(key=lambda event: event[0])
        _survey_cache[key] = moves
    return _survey_cache[key]


def wl_stations() -> list:
    """Every water-level station the control-point survey knows about.

    Returns
    -------
    list of str
        Sorted station names, with the ``_WL`` suffix.
    """
    return sorted(control_point_elevations())


def reference_elevation(station: str, index) -> pd.Series:
    """A station's reference elevation over time, stepping at each sensor move.

    Parameters
    ----------
    station : str
        Station name with ``_WL``.
    index : array-like
        Timestamps to produce a reference elevation for.

    Returns
    -------
    pandas.Series
        Reference elevation (NAVD88 feet), indexed by `index`. Constant at the
        control-point elevation before the first move, so it extends past the end
        of the survey record.

    Raises
    ------
    KeyError
        If the station has no surveyed control point.
    """
    elevations = control_point_elevations()
    if station not in elevations:
        raise KeyError(f"no surveyed control-point elevation for {station!r} in "
                       f"{settings.MAGL_CONTROL_POINTS_CSV.name}")
    return datum_module.stepwise_reference_elevation(
        index, elevations[station], sensor_moves().get(station, []))


def sensor_elevation(station: str, index, *, units: str = "ft",
                     sensor_datum: bool = True) -> pd.Series:
    """The elevation a station's distance readings are measured from.

    Subtract a distance reading from this to get water-surface elevation.

    Parameters
    ----------
    station : str
        Station name with ``_WL``.
    index : array-like
        Timestamps to produce an elevation for.
    units : {'ft', 'm', 'cm', 'mm'}, default 'ft'
        Units of the returned elevation. The datum stays NAVD88.
    sensor_datum : bool, default True
        Apply the control-point offset, putting the elevation at the radar's own
        face rather than the surveyed benchmark.

    Returns
    -------
    pandas.Series
        Elevation above NAVD88 in `units`, stepping at each sensor move.

    Raises
    ------
    KeyError
        If the station has no surveyed control point.
    """
    elevation = reference_elevation(station, index)
    if sensor_datum:
        try:
            elevation = elevation - magl_spreadsheets.control_point_offset(station)
        except KeyError:
            log.info("no control-point offset for %s; leaving the sensor elevation "
                     "on the control-point datum", station)
    elevation = elevation * datum_module._length_factor("ft", units)
    return elevation.rename(f"sensor_elevation_{units}")


# ---------------------------------------------------------------------------
# the pagaia database
# ---------------------------------------------------------------------------

def pagaia_session(environment: str = settings.PAGAIA_ENVIRONMENT):
    """Open a pagaia session on the environment the MAGL network lives in.

    A convenience for the MAGL loaders below, which each need a session and would
    otherwise open their own. When you are driving the package yourself, build the
    session with fb_pagaia directly - it is two calls, and writing them out keeps
    what you connected to visible::

        from fb_pagaia import defaults, utils
        api = utils.start_session(config=defaults.ENVIRONMENTS["production"],
                                  headers=defaults.HEADERS["production"])

    Parameters
    ----------
    environment : str, default 'production'
        Which pagaia environment (see ``fb_pagaia.defaults.ENVIRONMENTS``). Leave it
        alone: production is the database of record for the MAGL network.

    Returns
    -------
    fb_pagaia api session

    Raises
    ------
    ImportError
        When the optional ``fb_pagaia`` client is not installed. See
        :func:`limnotech_rating_curves.data.pagaia.require`.
    """
    from . import pagaia
    pagaia.require()
    from fb_pagaia import defaults, utils
    return utils.start_session(config=defaults.ENVIRONMENTS[environment],
                               headers=defaults.HEADERS[environment])


def pagaia_stations(stations, api=None):
    """Load MAGL water-level stations from pagaia as a network.

    For the MAGL listings and the map, which need many stations at once and must
    not lose all of them to one bad name.

    Parameters
    ----------
    stations : sequence of str
        Station names with ``_WL``, e.g. ``["SBR-09_WL", "SR-08_WL"]``.
    api : optional
        An existing pagaia session. One is opened with :func:`pagaia_session` if
        not given.

    Returns
    -------
    fb_pagaia.core.Network
        Stations pagaia does not know are logged and skipped.

    Raises
    ------
    ImportError
        When the optional ``fb_pagaia`` client is not installed.
    """
    from . import pagaia
    core = pagaia.require().core
    api = pagaia_session() if api is None else api
    network = core.Network(api, stations_key="name")
    for station in stations:
        try:
            network.add_station(core.Station.from_name(name=str(station), api=api))
        except Exception as exc:  # noqa: BLE001
            log.warning("pagaia station %s not found: %s", station, exc)
    return network


def pagaia_distance_ft(station: str, start=None, end=None,
                       refresh: bool = False) -> pd.Series:
    """Distance-to-surface from the pagaia database, converted to feet.

    Parameters
    ----------
    station : str
        Station name with ``_WL``.
    start, end : str or datetime, optional
        Window to fetch. Both are needed; without them nothing is fetched.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.Series
        Native resolution, feet. Empty when the server is unreachable, or when the
        optional ``fb_pagaia`` client is not installed - both are logged, not raised,
        so :func:`distance_record_ft` can fall back to the spreadsheet.
    """
    if start is None or end is None:
        return pd.Series(dtype=float, name=station)

    from . import pagaia, pagaia_corrections
    if not pagaia.installed():
        log.info("fb_pagaia is not installed, so %s uses the spreadsheet record only "
                 "(%s)", station, pagaia.INSTALL_HINT)
        return pd.Series(dtype=float, name=station)
    core = pagaia.require().core
    try:
        loaded = core.Station.from_name(name=str(station), api=pagaia_session())
    except Exception as exc:  # noqa: BLE001 - server down / off VPN
        log.warning("pagaia unavailable for %s (%s); using the spreadsheet record",
                    station, type(exc).__name__)
        return pd.Series(dtype=float, name=station)

    # Three steps, deliberately spelled out: fetch what the database holds, correct
    # the units it misreports, then convert. See
    # limnotech_rating_curves.data.pagaia_corrections.
    raw = pagaia.raw_station_series(loaded, variable="distance", start=start,
                                    end=end, refresh=refresh)
    meters = pagaia_corrections.to_meters(raw, pagaia.station_name(loaded),
                                          variable="distance")
    return datum_module.in_units(meters, "ft").rename(station)


def distance_record_ft(station: str, start=None, end=None,
                       quality_controlled: bool = True,
                       refresh: bool = False) -> pd.Series:
    """The merged distance-to-surface record for one station, in feet.

    The spreadsheet where it has data, pagaia for the tail beyond it.

    Parameters
    ----------
    station : str
        Station name with ``_WL``.
    start, end : str or datetime, optional
        Window for the pagaia part. Defaults to the tail the spreadsheet does not
        cover: from its last reading (or ``settings.MAGL_RECORD_START`` when it
        has none) to
        today. pagaia's API needs both ends, so leaving them out used to skip the
        database entirely and silently end the record with the spreadsheet.
    quality_controlled : bool, default True
        Apply the spreadsheet's QA/QC flag.
    refresh : bool, default False
        Refetch the pagaia part.

    Returns
    -------
    pandas.Series
        Native resolution, feet.
    """
    spreadsheet = magl_spreadsheets.distance_ft(station,
                                         quality_controlled=quality_controlled)
    if start is None:
        start = (spreadsheet.index.max() if not spreadsheet.empty
                 else pd.Timestamp(settings.MAGL_RECORD_START))
    if end is None:
        end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    database = pagaia_distance_ft(station, start, end, refresh=refresh)
    if spreadsheet.empty:
        return database
    if database.empty:
        return spreadsheet
    # pagaia supplies the tail, never the interior. Without this, asking for a window
    # inside the spreadsheet's span lets pagaia fill gaps the QA/QC flag opened, so
    # the same instant reads differently depending on the window requested.
    database = database[database.index > spreadsheet.index.max()]
    return magl_spreadsheets.clean_series(
        spreadsheet.combine_first(database)).rename(station)


def water_surface_elevation_ft(station: str, start=None, end=None, *,
                               sensor_datum: bool = True,
                               quality_controlled: bool = True,
                               refresh: bool = False) -> pd.Series:
    """Water-surface elevation for one station, in NAVD88 feet.

    This is the MAGL stage record: the merged distance readings subtracted from the
    station's time-varying reference elevation.

    Parameters
    ----------
    station : str
        Station name with ``_WL``.
    start, end : str or datetime, optional
        Window for the pagaia part of the record.
    sensor_datum : bool, default True
        Also subtract the station's control-point offset, putting the result on the
        sensor datum - the same tie the spreadsheet's own elevation column uses,
        and the one to use for anything that has to agree with it. Set False to
        stay on the raw control-point datum. See ``magl_spreadsheets.control_point_offset``.
    quality_controlled : bool, default True
        Apply the spreadsheet's QA/QC flag.
    refresh : bool, default False
        Refetch the pagaia part.

    Returns
    -------
    pandas.Series
        Elevation (NAVD88 feet) at native resolution.
    """
    distance = distance_record_ft(station, start, end,
                                  quality_controlled=quality_controlled,
                                  refresh=refresh)
    if distance.empty:
        return pd.Series(dtype=float, name=station)
    reference = reference_elevation(station, distance.index)
    elevation = datum_module.distance_to_stage(distance, reference)
    if sensor_datum:
        try:
            elevation = elevation - magl_spreadsheets.control_point_offset(station)
        except KeyError:
            log.info("no control-point offset for %s; leaving stage on the "
                     "control-point datum", station)
    return elevation.rename(station)


def stage_at_times(station: str, times,
                   tolerance: str = settings.MAGL_STAGE_MATCH_TOLERANCE,
                   refresh: bool = False, **kwargs) -> pd.DataFrame:
    """Match a station's stage record to the instants discharge was measured.

    Parameters
    ----------
    station : str
        Station name with ``_WL``.
    times : array-like
        The discharge-measurement timestamps.
    tolerance : str, default '3h'
        Largest acceptable gap between a measurement and the reading used for it.
    refresh : bool, default False
        Refetch the pagaia part of the record.
    **kwargs
        Forwarded to :func:`water_surface_elevation_ft`.

    Returns
    -------
    pandas.DataFrame
        Indexed by the requested timestamps, with ``stage_ft`` (NAVD88 elevation),
        ``stage_time`` and ``lag_min``. Cached on the request.
    """
    stamps = datum_module._as_naive_index(times)
    start = (stamps.min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end = (stamps.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    def build():
        """Fetch the stage record and match it to the requested times."""
        record = water_surface_elevation_ft(station, start=start, end=end,
                                           refresh=refresh, **kwargs)
        return datum_module.stage_at_times(record, stamps, tolerance=tolerance)

    return cache.cached("magl_stage_at_times",
                        (station, tuple(stamp.isoformat() for stamp in stamps),
                         tolerance, tuple(sorted(kwargs.items()))),
                        build, refresh)


# ---------------------------------------------------------------------------
# the manual discharge measurements
# ---------------------------------------------------------------------------

def discharge_measurements(sensor=None, path=None) -> pd.DataFrame:
    """The detailed manual discharge export, cleaned.

    One row per measurement with its exact timestamp - which is what makes it the
    source to use when stage has to be matched from the sensor record. Rows with no
    timestamp, no discharge, or a non-positive discharge are dropped, since a
    rating is fitted in log-discharge space.

    Parameters
    ----------
    sensor : str, optional
        Restrict to one sensor, e.g. ``"SBR-09"``.
    path : path-like, optional
        The CSV. Defaults to :func:`discharge_csv_path`.

    Returns
    -------
    pandas.DataFrame
        Columns ``sensor_id``, ``station``, ``time``, ``discharge_cfs``,
        ``source_type`` (``xlsx`` wading, ``qrev_adcp``, ``tsv_flowtracker``),
        ``folder_date``, sorted by sensor and time.
    """
    path = Path(discharge_csv_path() if path is None else path)
    frame = pd.read_csv(path, dtype={"sensor_id": str})
    frame["discharge_cfs"] = pd.to_numeric(frame["discharge_cfs"], errors="coerce")
    frame["time"] = pd.to_datetime(frame["timestamp"], format="mixed", errors="coerce")
    frame = frame.dropna(subset=["discharge_cfs", "time"])
    frame = frame[frame["discharge_cfs"] > 0]
    frame["sensor_id"] = frame["sensor_id"].astype(str).str.strip()
    frame["station"] = frame["sensor_id"] + "_WL"
    if sensor is not None:
        frame = frame[frame["sensor_id"] == str(sensor)]
    columns = ["sensor_id", "station", "time", "discharge_cfs", "source_type",
               "folder_date"]
    columns = [column for column in columns if column in frame.columns]
    return (frame[columns].sort_values(["sensor_id", "time"])
            .reset_index(drop=True))


def discharge_report(sensor=None, path=None) -> pd.DataFrame:
    """The curated discharge report: measurements with field-recorded stage.

    The reviewed version of the same measurements, carrying the water-surface
    elevation as read in the field. Prefer it where it has a stage, because a stage
    read at the measurement beats one matched from the sensor record afterwards.

    Parameters
    ----------
    sensor : str, optional
        Restrict to one sensor.
    path : path-like, optional
        The CSV. Defaults to :func:`discharge_report_csv_path`.

    Returns
    -------
    pandas.DataFrame
        Columns ``sensor_id``, ``station``, ``date``, ``stage_ft`` (NAVD88 water
        elevation, NaN where the report has none), ``discharge_cfs``, ``meter``.
    """
    path = Path(discharge_report_csv_path() if path is None else path)
    frame = pd.read_csv(path)
    frame = frame.rename(columns={
        "Station": "sensor_id", "Date": "date",
        "Water Elevation (NAVD88 ft)": "stage_ft",
        "Discharge (cfs)": "discharge_cfs", "Meter Used": "meter"})
    frame["sensor_id"] = frame["sensor_id"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stage_ft"] = pd.to_numeric(frame["stage_ft"], errors="coerce")
    frame["discharge_cfs"] = pd.to_numeric(frame["discharge_cfs"], errors="coerce")
    frame = frame[frame["sensor_id"].ne("") & frame["date"].notna()]
    frame = frame.dropna(subset=["discharge_cfs"])
    frame = frame[frame["discharge_cfs"] > 0]
    frame["station"] = frame["sensor_id"] + "_WL"
    if sensor is not None:
        frame = frame[frame["sensor_id"] == str(sensor)]
    return (frame[["sensor_id", "station", "date", "stage_ft", "discharge_cfs", "meter"]]
            .sort_values(["sensor_id", "date"]).reset_index(drop=True))


def list_sensors(min_points: int = 3, source: str = "report") -> pd.DataFrame:
    """Which sensors have enough discharge measurements to fit a rating.

    Parameters
    ----------
    min_points : int, default 3
        Fewest measurements to list a sensor. Three is the floor for a
        single-segment power law.
    source : {'report', 'detailed'}, default 'report'
        Which export to count.

    Returns
    -------
    pandas.DataFrame
        Indexed by sensor, with ``n`` measurements and, for the report, how many
        already carry a field-recorded stage. Most measurements first.
    """
    if source == "detailed":
        frame = discharge_measurements()
        counts = frame.groupby("sensor_id").agg(
            n=("discharge_cfs", "size"),
            n_wading=("source_type", lambda kinds: int((kinds == "xlsx").sum())),
            n_adcp=("source_type", lambda kinds: int((kinds == "qrev_adcp").sum())),
            n_flowtracker=("source_type",
                           lambda kinds: int((kinds == "tsv_flowtracker").sum())))
    else:
        frame = discharge_report()
        counts = frame.groupby("sensor_id").agg(
            n=("discharge_cfs", "size"),
            n_with_field_stage=("stage_ft", lambda values: int(values.notna().sum())))
    counts = counts[counts["n"] >= min_points]
    return counts.sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# the samples
# ---------------------------------------------------------------------------

def sensor_sample(sensor: str, *, stage_datum="lowest", source: str = "report",
                  tolerance: str = settings.MAGL_STAGE_MATCH_TOLERANCE,
                  refresh: bool = False) -> Sample:
    """MAGL discharge against MAGL stage for one sensor - the rating you would use.

    Parameters
    ----------
    sensor : str
        Sensor id, e.g. ``"SBR-09"`` (with or without ``_WL``).
    stage_datum : optional, default 'lowest'
        How to turn the NAVD88 water-surface elevation into gage height.
        ``"lowest"`` references the sensor's own lowest reading, which is the only
        option available at a site with no surveyed low-water reference and makes
        the resulting stage a *local* gage height. Pass an elevation instead if you
        have a surveyed one. ``None`` leaves stage as a raw NAVD88 elevation, which
        every model can still fit but which puts the stage of zero flow hundreds of
        feet from zero.
    source : {'report', 'detailed'}, default 'report'
        Which discharge export to use. ``"report"`` prefers the field-recorded
        stage and matches from the sensor record only where the report has none;
        ``"detailed"`` always matches from the sensor record, using each
        measurement's exact timestamp.
    tolerance : str, default '3h'
        Largest acceptable gap when stage has to be matched.
    refresh : bool, default False
        Refetch the pagaia part of the stage record.

    Returns
    -------
    Sample

    Examples
    --------
    >>> sample = sensor_sample("SBR-09")            # doctest: +SKIP
    >>> sample.compare(models="all").metrics         # doctest: +SKIP
    """
    sensor = str(sensor).replace("_WL", "")
    station = f"{sensor}_WL"

    if source == "detailed":
        measured = discharge_measurements(sensor)
        if measured.empty:
            return Sample(stage_ft=[], discharge_cfs=[], site_id=station,
                          source="magl", skipped=f"no discharge measurements for {sensor}")
        matched = stage_at_times(station, measured["time"], tolerance=tolerance,
                                 refresh=refresh)
        merged = measured.set_index("time").join(matched[["stage_ft", "lag_min"]])
        merged = merged.dropna(subset=["stage_ft"]).reset_index()
        note = (f"stage matched from the sensor record within {tolerance}")
    else:
        report = discharge_report(sensor)
        if report.empty:
            return Sample(stage_ft=[], discharge_cfs=[], site_id=station,
                          source="magl", skipped=f"no report rows for {sensor}")
        merged, note = _fill_report_stage(station, sensor, report, tolerance, refresh)
        merged = merged.dropna(subset=["stage_ft"]).reset_index(drop=True)

    if merged.empty:
        return Sample(stage_ft=[], discharge_cfs=[], site_id=station, source="magl",
                      skipped=(f"no measurement for {sensor} had a stage within "
                               f"{tolerance}"))

    sample = Sample.of(merged, site_id=station, source="magl",
                       stage_datum=stage_datum,
                       datum_note=f"MAGL water-surface elevation (NAVD88 ft); {note}")
    log.info("MAGL %s: %d fittable measurement(s), stage %.3f-%.3f ft, "
             "discharge %.3g-%.3g cfs", sensor, len(sample),
             *sample.stage_range, *sample.discharge_range)
    return sample


def _fill_report_stage(station, sensor, report, tolerance, refresh):
    """Report rows with stage filled from the sensor record where the report has none."""
    missing = report["stage_ft"].isna()
    note = "stage as recorded in the field"
    if not missing.any():
        return report.copy(), note

    detailed = discharge_measurements(sensor)
    timestamps = (detailed.assign(date=detailed["time"].dt.normalize())
                  .drop_duplicates(["date"], keep="first")
                  .set_index("date")["time"])
    filled = report.copy()
    wanted = filled.loc[missing, "date"].map(timestamps)
    resolvable = wanted.dropna()
    if len(resolvable):
        matched = stage_at_times(station, resolvable.to_numpy(), tolerance=tolerance,
                                 refresh=refresh)
        lookup = dict(zip(resolvable.index, matched["stage_ft"].to_numpy()))
        filled.loc[resolvable.index, "stage_ft"] = [lookup[i] for i in resolvable.index]
        note = (f"stage as recorded in the field, with {len(resolvable)} row(s) "
                f"filled from the sensor record within {tolerance}")
    return filled, note


def colocated_sample(station: str, gage=None, *,
                     tolerance: str = settings.MAGL_STAGE_MATCH_TOLERANCE,
                     refresh: bool = False) -> Sample:
    """USGS field discharge against the co-located MAGL sensor's stage.

    The sharpest available check on the MAGL stage derivation: the discharge is
    USGS-measured and the gage's own rating is known, so any disagreement is
    attributable to the stage. MAGL's NAVD88 elevation is tied to the gage's
    gage-height axis by subtracting ``alt_va``, the elevation of gage height zero -
    which is only valid if the gage's datum is NAVD88 too. When it is not, an empty
    sample is returned with the reason, rather than a plausible-looking wrong
    answer.

    Parameters
    ----------
    station : str
        MAGL station name, with or without ``_WL``.
    gage : str, optional
        The USGS gage. Looked up from the registry if not given.
    tolerance : str, default '3h'
        Largest acceptable gap between a USGS measurement and the MAGL reading
        matched to it.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    Sample
        Stage on the USGS gage-height axis, so a fit is directly comparable with
        the gage's published rating.

    Raises
    ------
    ValueError
        If the station has no co-located gage.
    """
    base = str(station).replace("_WL", "")
    station_wl = f"{base}_WL"
    gage = colocated_gage(base) if gage is None else str(gage)
    if gage is None:
        raise ValueError(f"{station!r} has no co-located USGS gage in "
                         "colocated_pairs.csv")
    gage = usgs.normalize_site_id(gage)

    measured = usgs.measurements(gage, refresh=refresh)
    measured = measured.copy()
    measured["time"] = datum_module._as_naive_index(
        pd.to_datetime(measured["time"], errors="coerce", utc=True)
        .dt.tz_localize(None))
    measured = measured.dropna(subset=["time"])
    # MAGL stage only exists from about MAGL_RECORD_START; older USGS visits can
    # never be matched, so drop them rather than fetch years of stage for nothing.
    measured = measured[measured["time"] >= pd.Timestamp(settings.MAGL_RECORD_START)]

    info = usgs.site_info(gage, refresh=refresh)
    if gage not in info.index:
        return Sample(stage_ft=[], discharge_cfs=[], site_id=station_wl,
                      source="colocated", skipped=f"no USGS site file for {gage}")
    alt_va = float(info.loc[gage, "alt_va"])
    alt_datum = str(info.loc[gage, "alt_datum_cd"]).strip()
    if alt_datum != "NAVD88":
        return Sample(
            stage_ft=[], discharge_cfs=[], site_id=station_wl, source="colocated",
            stage_label="USGS gage height (ft)",
            skipped=(f"gage {gage} datum is {alt_datum}, not NAVD88, so subtracting "
                     f"alt_va from a NAVD88 elevation is not a valid tie"))
    if measured.empty:
        return Sample(stage_ft=[], discharge_cfs=[], site_id=station_wl,
                      source="colocated",
                      skipped=(f"no USGS measurements at {gage} since "
                               f"{settings.MAGL_RECORD_START}"))

    matched = stage_at_times(station_wl, measured["time"], tolerance=tolerance,
                             refresh=refresh)
    # the USGS gage height is dropped deliberately: this sample's whole point is to
    # put MAGL's stage on the gage's axis and see whether it reproduces the rating
    merged = (measured.set_index("time")[["discharge_cfs"]]
              .join(matched[["stage_ft", "lag_min"]]))
    merged = merged.dropna(subset=["stage_ft"]).reset_index()
    if merged.empty:
        return Sample(stage_ft=[], discharge_cfs=[], site_id=station_wl,
                      source="colocated",
                      skipped=(f"no USGS measurement at {gage} had a MAGL reading "
                               f"within {tolerance}"))

    sample = Sample.of(
        merged, site_id=station_wl, source="colocated",
        stage_datum=datum_module.StageDatum.usgs_gage_datum(alt_va, alt_datum),
        datum_note=(f"MAGL NAVD88 elevation tied to USGS {gage} gage height by "
                    f"subtracting alt_va = {alt_va:.2f} ft {alt_datum}; stage "
                    f"matched within {tolerance}"))
    sample.stage_label = "USGS gage height via surveyed tie (ft)"
    log.info("colocated %s / USGS %s: %d fittable measurement(s)", station_wl, gage,
             len(sample))
    return sample


def site_directory(min_points: int = 3) -> dict:
    """Everything you can ask for a rating at, for validation and listings.

    Parameters
    ----------
    min_points : int, default 3
        Fewest discharge measurements before a MAGL sensor is listed.

    Returns
    -------
    dict
        ``clusters``, ``gages``, ``colocated_stations`` and ``sensors``.
    """
    return {
        "clusters": clusters(),
        "gages": all_gages(),
        "colocated_stations": sorted(station for station, _ in colocated_pairs()),
        "sensors": list(list_sensors(min_points).index),
    }


def list_sites(min_points: int = 3) -> pd.DataFrame:
    """Every MAGL-network site with the loader to use for it.

    Parameters
    ----------
    min_points : int, default 3
        Fewest discharge measurements before a sensor is listed.

    Returns
    -------
    pandas.DataFrame
        Columns ``site``, ``kind``, ``cluster``, ``loader``.
    """
    directory = site_directory(min_points)
    rows = [{"site": gage, "kind": "USGS gage", "cluster": gage_cluster(gage),
             "loader": "magl_rating_curves.gage_sample"}
            for gage in directory["gages"]]
    rows += [{"site": station, "kind": "co-located MAGL station",
              "cluster": station_cluster(station),
              "loader": "magl_rating_curves.colocated_sample"}
             for station in directory["colocated_stations"]]
    rows += [{"site": sensor, "kind": "MAGL sensor",
              "cluster": station_cluster(sensor),
              "loader": "magl_rating_curves.sensor_sample"}
             for sensor in directory["sensors"]]
    return pd.DataFrame(rows)


def cross_validate(sample, **kwargs):
    """Cross-validate a MAGL sample - leave-one-out, because the samples are tiny.

    A MAGL sensor has three to seven discharge measurements. Holding out 90% of
    those leaves three to train on, at which point the segmented power laws and the
    spline fail every fold and the exercise measures nothing. Leave-one-out trains
    on ``n - 1`` and gives every model something to work with, and the score to
    quote is the *pooled* one: each fold predicts its single held-out measurement,
    and all n predictions are scored together. Per-fold NSE is undefined for a
    single point.

    Parameters
    ----------
    sample : Sample or pandas.DataFrame
        The measurements.
    **kwargs
        Forwarded to :func:`limnotech_rating_curves.cross_validate`. The scheme
        defaults to leave-one-out here; pass ``scheme="holdout"`` to override.

    Returns
    -------
    CrossValidation
    """
    kwargs.setdefault("scheme", "loo")
    return crossval.cross_validate(sample, **kwargs)


# ==============================================================================
# the field flow workbooks (flow@<sensor>.xlsx)
# ==============================================================================

#: Sheet in a ``flow@<sensor>.xlsx`` workbook holding the paired measurements the
#: spreadsheet's own chart trendline is fitted to.
settings.MAGL_RATING_CURVE_SHEET = "Rating Curve"


def rating_curve_sheet(path,
                       sheet: str = settings.MAGL_RATING_CURVE_SHEET
                       ) -> pd.DataFrame:
    """The paired measurements behind a flow workbook's rating curve.

    A ``flow@<sensor>.xlsx`` workbook carries one sheet of the measurements its chart
    is built on: a title row, then ``Date`` / ``Stage`` / ``Discharge``. This reads
    exactly that and nothing else - the workbook's other sheets are per-visit
    gauging worksheets and the QA'd hydrograph, none of which the rating needs.

    Parameters
    ----------
    path : path-like
        The workbook.
    sheet : str, optional
        Sheet name, if it is not the usual ``"Rating Curve"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``date``, ``stage_ft``, ``discharge_cfs``, ascending by stage, with
        blank and incomplete rows dropped.

        **Stage here is a water-surface elevation** (NAVD88, around 798 ft at
        SBR-09), not a gage height, because that is what the spreadsheet plots and
        fits. Leave it that way to reproduce the spreadsheet's coefficients; pass
        ``stage_datum="lowest"`` when building a Sample if you want gage height
        instead.

    Raises
    ------
    ValueError
        If the sheet does not have the three expected columns.
    """
    frame = pd.read_excel(path, sheet_name=sheet, header=1)
    frame.columns = [str(name).strip().lower() for name in frame.columns]
    wanted = {"date": "date", "stage": "stage_ft", "discharge": "discharge_cfs"}
    missing = [name for name in wanted if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{Path(path).name} sheet {sheet!r} has columns {list(frame.columns)}; "
            f"expected Date, Stage and Discharge (missing {missing})")
    tidy = frame[list(wanted)].rename(columns=wanted)
    tidy["stage_ft"] = pd.to_numeric(tidy["stage_ft"], errors="coerce")
    tidy["discharge_cfs"] = pd.to_numeric(tidy["discharge_cfs"], errors="coerce")
    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
    tidy = tidy.dropna(subset=["stage_ft", "discharge_cfs"])
    return tidy.sort_values("stage_ft").reset_index(drop=True)


def flow_sheet_sample(path, sheet: str = settings.MAGL_RATING_CURVE_SHEET, *,
                      stage_datum=None,
                      sensor=None) -> Sample:
    """A :class:`~limnotech_rating_curves.core.Sample` from a flow workbook's sheet.

    Parameters
    ----------
    path : path-like
        The ``flow@<sensor>.xlsx`` workbook.
    sheet : str, optional
        Sheet name.
    stage_datum : optional
        Convert stage onto a gage-height reference. ``None`` (the default) keeps the
        elevations as the spreadsheet has them, which is what reproducing the
        spreadsheet's own curve requires.
    sensor : str, optional
        Site id to record. Defaults to the part of the file name after ``@``.

    Returns
    -------
    Sample
    """
    path = Path(path)
    frame = rating_curve_sheet(path, sheet=sheet)
    site = sensor or path.stem.split("@")[-1]
    return Sample.of(frame, site_id=str(site), source="magl_flow_workbook",
                     stage_datum=stage_datum,
                     stage_label=("water-surface elevation (ft NAVD88)"
                                  if stage_datum is None else None),
                     datum_note=f"{path.name} sheet {sheet!r}")


def _site_matches(tokens: Sequence[str] | None, *, group: str | None = None,
                  gage: str | None = None,
                  station: str | None = None) -> bool:
    """Does a candidate site match any selection token? No tokens matches everything."""
    if not tokens:
        return True
    for raw in tokens:
        token = str(raw).strip()
        if group and token.lower() == str(group).lower():
            return True
        if gage and (token == gage or usgs.normalize_site_id(token) == gage):
            return True
        if station and token.lower().replace("_wl", "") == str(station).lower():
            return True
    return False


def assemble_magl_sites(sites: Sequence[str] | None = None,
                        sources: Sequence[str] | None = None,
                        min_points: int = 3,
                        years: int = 5,
                        refresh: bool = False) -> list[SiteRating]:
    """Build a :class:`SiteRating` for each selected MAGL-network site, unfitted.

    Only the selected sites are fetched, so a targeted run stays fast.

    Parameters
    ----------
    sites : sequence of str, optional
        Selection tokens: a MAGL sensor (``"SBR-09"``), a co-located station
        (``"SR-08"``), a cluster (``"SR"``), or a USGS gage id (``"04176356"``).
        ``None`` takes every site.
    sources : sequence of str, optional
        Restrict to these sample sources (see ``settings.MAGL_SOURCES``).
    min_points : int, default 3
        Fewest discharge measurements before a MAGL sensor is included.
    years : int, default 5
        How far back to take a USGS gage's measurements. A channel changes, so an old
        gaging describes a rating that no longer applies. A gage with nothing this
        recent is left out.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    list of SiteRating
        With samples and coordinates filled in and ``fits`` still empty.
    """
    from . import pagaia

    wanted = lambda source: (not sources) or (source in sources)

    gages = all_gages()
    selected_gages = [gage for gage in gages if wanted("usgs_gage")
                      and _site_matches(sites, group=gage_cluster(gage),
                                        gage=gage)]
    selected_pairs = [(station, gage) for station, gage in colocated_pairs()
                      if wanted("colocated")
                      and _site_matches(sites, group=station_cluster(station),
                                        gage=gage, station=station)]
    selected_sensors = [sensor for sensor in list_sensors(min_points).index
                        if wanted("magl")
                        and _site_matches(sites, group=station_cluster(sensor),
                                          station=sensor)]

    needed_gages = set(selected_gages) | {gage for _, gage in selected_pairs}
    gage_coordinates = usgs.coordinates(sorted(needed_gages), refresh=refresh) \
        if needed_gages else {}
    gage_names = {}
    if needed_gages:
        info = usgs.site_info(sorted(needed_gages), refresh=refresh)
        gage_names = {gage: str(row.get("station_name") or "")
                      for gage, row in info.iterrows()}

    station_names = ([f"{sensor}_WL" for sensor in selected_sensors]
                     + [f"{station}_WL" for station, _ in selected_pairs])
    station_coordinates = {}
    if station_names:
        try:
            station_coordinates = pagaia.station_coordinates(
                pagaia_stations(sorted(set(station_names))))
        except Exception as exc:  # noqa: BLE001 - pagaia down, off VPN, or not installed
            log.warning("MAGL station coordinates unavailable (%s: %s); those sites "
                        "will fall back to their gage's position or be skipped",
                        type(exc).__name__, exc)

    assembled = []

    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)

    for gage in selected_gages:
        sample = usgs.gage_sample(gage, start=start.date().isoformat(),
                                  end=end.date().isoformat(), refresh=refresh)
        if not len(sample):
            log.info("usgs_gage:%s: no measurement in the last %d years, left out",
                     gage, years)
            continue
        assembled.append(SiteRating(
            sample_id=f"usgs_gage:{gage}", source="usgs_gage",
            label=f"USGS {gage} {gage_names.get(gage, '')}".strip(), sample=sample,
            gage=gage, group=gage_cluster(gage),
            coords=gage_coordinates.get(gage)))

    for station, gage in selected_pairs:
        try:
            sample = colocated_sample(station, gage, refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.warning("co-located %s could not be assembled: %s", station, exc)
            continue
        assembled.append(SiteRating(
            sample_id=f"colocated:{station}", source="colocated",
            label=f"{station} MAGL stage vs USGS {gage} discharge", sample=sample,
            station=f"{station}_WL", gage=gage,
            group=station_cluster(station),
            coords=(station_coordinates.get(f"{station}_WL")
                    or gage_coordinates.get(gage))))

    for sensor in selected_sensors:
        try:
            sample = sensor_sample(sensor, refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.warning("MAGL %s could not be assembled: %s", sensor, exc)
            continue
        assembled.append(SiteRating(
            sample_id=f"magl:{sensor}", source="magl",
            label=f"{sensor} (MAGL discharge)", sample=sample,
            station=f"{sensor}_WL", group=station_cluster(sensor),
            coords=station_coordinates.get(f"{sensor}_WL")))

    return assembled

