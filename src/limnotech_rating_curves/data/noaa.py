"""NOAA / National Weather Service published rating curves, from the NWPS API.

The NWS maintains its own stage-discharge rating at each forecast point, separate
from the USGS rating for the same gage. Where both exist they are worth putting side
by side: they are two agencies' answers to the same question, and a disagreement
between them is a finding rather than a bug.

What the API gives, and what it does not
----------------------------------------
An NWPS gauge is keyed by its NWS location id (an "LID" - five characters, e.g.
``KWPM7``), not by a USGS site number. ``/gauges/{lid}`` returns ``usgsId`` directly,
so going from an LID to a USGS site is easy. Going the other way is not: the API has
no gauge search - ``?srch=``, ``?usgsId=`` and ``?bbox.*`` are all ignored and return
the unfiltered list - and the listing carries no ``usgsId`` to filter on locally. So
the reverse lookup comes from NOAA's HADS crosswalk instead, a flat file pairing every
NWSLI with its USGS number; see :func:`crosswalk`.

The rating itself arrives as a lookup table of stage and flow, not as an equation.
Interpolate it (:func:`discharge_at`); do not fit anything to it.

Datums
------
NWPS stage and USGS gage height are usually the same axis, but "usually" is not
"always" - an NWS gage zero can differ from the USGS one, and then the two curves are
simply not on comparable axes. :func:`datum_agreement` checks this against live
readings from both agencies so a caller can warn. Nothing here silently shifts a
curve to make it line up: a mismatch is the finding.
"""

import logging

import numpy as np
import pandas as pd

from ..support import cache

log = logging.getLogger(__name__)

#: Root of the National Water Prediction Service API.
NWPS_ROOT = "https://api.water.noaa.gov/nwps/v1"

#: NOAA's NWSLI-to-USGS crosswalk. A fixed-width, pipe-delimited listing with three
#: header lines, then one row per gage: ``NWSLI|USGS number|GOES id|HSA|lat|lon|name``.
HADS_CROSSWALK_URL = "https://hads.ncep.noaa.gov/USGS/ALL_USGS-HADS_SITES.txt"

#: Largest stage difference (ft) between the two agencies' live readings that still
#: counts as "the same axis". Wider than gauge noise, far narrower than a real datum
#: offset, which is typically a whole foot or more.
DATUM_TOLERANCE_FT = 0.05

#: How long to wait on either web service, in seconds.
TIMEOUT = 30


def normalize_lid(lid) -> str:
    """An NWS location id in the form the API expects: upper case, no padding."""
    return str(lid).strip().upper()


def _get(url: str, params=None) -> dict:
    """One JSON GET against the NWPS API."""
    import requests
    response = requests.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def gauge_info(lid, refresh: bool = False):
    """Metadata for one NWPS gauge, including the USGS site it corresponds to.

    Parameters
    ----------
    lid : str
        NWS location id, e.g. ``"KWPM7"``.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    dict or None
        ``lid``, ``name``, ``usgs_site`` (empty when the gauge has no USGS
        counterpart), ``latitude``, ``longitude``, ``state``, ``datum_ft``,
        ``vertical_datum``, and the live ``observed_stage_ft`` / ``observed_time``
        used by :func:`datum_agreement`. None when the gauge is not found.
    """
    lid = normalize_lid(lid)

    def build():
        """Fetch and flatten the gauge record, or None if there is no such gauge."""
        try:
            payload = _get(f"{NWPS_ROOT}/gauges/{lid}")
        except Exception as exc:  # noqa: BLE001
            log.info("no NWPS gauge %s (%s)", lid, type(exc).__name__)
            return None
        observed = (payload.get("status") or {}).get("observed") or {}
        stage = observed.get("primary")
        # the API reports a missing reading as -999 rather than as null
        if stage is not None and float(stage) <= -999:
            stage = None
        return {
            "lid": payload.get("lid", lid),
            "name": payload.get("name", ""),
            "usgs_site": str(payload.get("usgsId") or ""),
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "state": ((payload.get("state") or {}).get("abbreviation")
                      if isinstance(payload.get("state"), dict) else payload.get("state")),
            "datum_ft": payload.get("datum"),
            "vertical_datum": ((payload.get("verticalDatum") or {}).get("name")
                               if isinstance(payload.get("verticalDatum"), dict)
                               else payload.get("verticalDatum")),
            "observed_stage_ft": None if stage is None else float(stage),
            "observed_time": observed.get("validTime"),
        }

    return cache.cached("noaa_gauge_info", (lid,), build, refresh)


def published_rating(lid, only_tenths: bool = True,
                     refresh: bool = False) -> pd.DataFrame:
    """The NWS published rating for a gauge, as a tidy curve.

    Parameters
    ----------
    lid : str
        NWS location id.
    only_tenths : bool, default True
        Ask the API for the tenth-of-a-foot table.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage_ft`` and ``discharge_cfs``, ascending in stage. An empty
        frame when the gauge publishes no rating, which is common and not an error.
    """
    lid = normalize_lid(lid)

    def build():
        """Tidy the rating table, or an empty frame if there is none."""
        empty = pd.DataFrame(columns=["stage_ft", "discharge_cfs"])
        try:
            payload = _get(f"{NWPS_ROOT}/gauges/{lid}/ratings",
                           params={"onlyTenths": str(bool(only_tenths)).lower()})
        except Exception as exc:  # noqa: BLE001
            log.info("no NWPS rating for %s (%s)", lid, type(exc).__name__)
            return empty
        rows = payload.get("data") or []
        if not rows:
            log.info("NWPS gauge %s publishes no rating", lid)
            return empty
        stage_units = (payload.get("stageUnits") or "ft").lower()
        flow_units = (payload.get("flowUnits") or "cfs").lower()
        if stage_units != "ft" or flow_units != "cfs":
            # every gauge seen so far reports ft/cfs; refuse rather than silently
            # mixing units into a curve that is compared against cfs elsewhere
            raise ValueError(
                f"NWPS gauge {lid} reports its rating in {stage_units}/{flow_units}, "
                "not ft/cfs; this package's curves are all ft/cfs")
        curve = pd.DataFrame({
            "stage_ft": [row.get("stage") for row in rows],
            "discharge_cfs": [row.get("flow") for row in rows],
        }).apply(pd.to_numeric, errors="coerce").dropna()
        return curve.sort_values("stage_ft").reset_index(drop=True)

    return cache.cached("noaa_published_rating", (lid, bool(only_tenths)),
                        build, refresh)


def crosswalk(refresh: bool = False) -> pd.DataFrame:
    """NOAA's NWSLI-to-USGS site crosswalk.

    The NWPS API cannot be searched by USGS site number, so this flat file is how a
    USGS site finds its NWS gauge.

    Parameters
    ----------
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``nws_lid``, ``usgs_site`` and ``name``. Empty if the file cannot be
        fetched, so a lookup degrades to "not found" rather than raising.
    """
    def build():
        """Parse the pipe-delimited listing past its three header lines."""
        import requests
        columns = ["nws_lid", "usgs_site", "name"]
        try:
            response = requests.get(HADS_CROSSWALK_URL, timeout=TIMEOUT)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            log.info("could not fetch the HADS crosswalk (%s)", type(exc).__name__)
            return pd.DataFrame(columns=columns)

        rows = []
        for line in response.text.splitlines()[3:]:
            fields = [field.strip() for field in line.split("|")]
            if len(fields) < 7 or not fields[0] or not fields[1]:
                continue
            # the header block ends with a rule of dashes, which splits into fields
            # like any other row - a USGS number is digits, so require that
            if not fields[1].isdigit():
                continue
            rows.append({"nws_lid": fields[0].upper(),
                         "usgs_site": fields[1], "name": fields[6]})
        table = pd.DataFrame(rows, columns=columns)
        log.info("HADS crosswalk: %d gage pairings", len(table))
        return table

    return cache.cached("noaa_hads_crosswalk", ("v1",), build, refresh)


def lid_for_usgs_site(site, refresh: bool = False):
    """The NWS location id for a USGS site, or None if it has no NWS gauge.

    Parameters
    ----------
    site : str
        USGS site number, with or without leading zeros.
    refresh : bool, default False
        Refetch the crosswalk instead of using the cache.

    Returns
    -------
    str or None
        The LID. None when the site is absent from the crosswalk - which simply
        means the NWS does not forecast at that gage, and is not an error.
    """
    from .usgs import normalize_site_id
    site = normalize_site_id(site)
    table = crosswalk(refresh=refresh)
    if table.empty:
        return None
    matches = table.loc[table["usgs_site"].map(normalize_site_id) == site, "nws_lid"]
    if matches.empty:
        return None
    if matches.nunique() > 1:
        log.info("USGS %s maps to several NWS ids %s; using the first",
                 site, sorted(matches.unique()))
    return str(matches.iloc[0])


def discharge_at(curve: pd.DataFrame, stage) -> np.ndarray:
    """Read discharge off a published rating table at given stages.

    Interpolated in log-discharge against stage, because a rating is close to a power
    law and a straight linear interpolation between tenth-of-a-foot points understates
    the curvature at the low-flow end where the table is coarsest relative to the flow.

    Stages outside the table's range come back NaN rather than extrapolated: the table
    is a published product, and continuing it past its last point would be inventing
    someone else's curve.

    Parameters
    ----------
    curve : pandas.DataFrame
        A published rating, as from :func:`published_rating`.
    stage : array-like
        Stages (ft) on the same axis as the table.

    Returns
    -------
    numpy.ndarray
        Discharge (cfs), NaN outside the table's stage range.
    """
    stage = np.asarray(stage, float)
    if curve is None or curve.empty:
        return np.full(stage.shape, np.nan)

    table = curve.dropna(subset=["stage_ft", "discharge_cfs"]).sort_values("stage_ft")
    table_stage = table["stage_ft"].to_numpy(float)
    table_flow = table["discharge_cfs"].to_numpy(float)

    positive = table_flow > 0
    predicted = np.full(stage.shape, np.nan)
    if positive.sum() >= 2:
        predicted = np.exp(np.interp(stage, table_stage[positive],
                                     np.log(table_flow[positive]),
                                     left=np.nan, right=np.nan))
    # zero-flow rows cannot be logged; fill them back in linearly where they apply
    if (~positive).any():
        linear = np.interp(stage, table_stage, table_flow, left=np.nan, right=np.nan)
        predicted = np.where(np.isnan(predicted), linear, predicted)
    outside = (stage < table_stage.min()) | (stage > table_stage.max())
    predicted[outside] = np.nan
    return predicted


def published_reference(lid, sample, refresh: bool = False):
    """The NWS rating packaged as a comparison reference for a sample.

    Mirrors :func:`limnotech_rating_curves.data.usgs.published_reference`, so an NWS
    curve appears in a comparison table as one more row alongside the USGS one.

    Parameters
    ----------
    lid : str
        NWS location id.
    sample : Sample
        The measurements to score against. Its stage must be on the gage-height axis
        the NWS table is indexed by - see :func:`datum_agreement`.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    dict or None
        ``{"label", "curve", "metrics"}``, or None when the gauge publishes no
        rating. Measurements outside the table's stage range are ignored rather than
        extrapolated, so ``metrics["n"]`` may be below ``len(sample)``.
    """
    from ..core import fit_metrics
    lid = normalize_lid(lid)
    curve = published_rating(lid, refresh=refresh)
    if curve is None or curve.empty or len(sample) == 0:
        return None
    predicted = discharge_at(curve, sample.stage_ft)
    return {"label": f"NWS published rating ({lid})",
            "curve": curve[["stage_ft", "discharge_cfs"]],
            "metrics": fit_metrics(sample.discharge_cfs, predicted)}


def datum_agreement(lid, refresh: bool = False) -> dict:
    """Do NWPS stage and USGS gage height read the same at the same moment?

    A rating table is only comparable to another agency's if both are indexed by the
    same stage axis. This checks that empirically rather than assuming it, by reading
    the NWPS gauge's latest observed stage and the USGS gage height at that same
    timestamp.

    Parameters
    ----------
    lid : str
        NWS location id.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    dict
        ``agrees`` (bool or None when it could not be checked), ``difference_ft``,
        ``nwps_stage_ft``, ``usgs_stage_ft``, ``usgs_site``, ``time`` and ``note``.
        ``agrees`` False means the two curves are on different axes and must not be
        overlaid - not that either is wrong.
    """
    lid = normalize_lid(lid)

    def build():
        """Compare the two agencies' live readings, or say why we could not."""
        unknown = {"agrees": None, "difference_ft": np.nan, "nwps_stage_ft": np.nan,
                   "usgs_stage_ft": np.nan, "usgs_site": "", "time": "", "note": ""}
        info = gauge_info(lid)
        if info is None:
            return {**unknown, "note": f"no NWPS gauge {lid}"}
        site, stamp = info["usgs_site"], info["observed_time"]
        nwps_stage = info["observed_stage_ft"]
        unknown = {**unknown, "usgs_site": site, "time": stamp or "",
                   "nwps_stage_ft": np.nan if nwps_stage is None else nwps_stage}
        if not site:
            return {**unknown, "note": f"NWPS gauge {lid} has no USGS counterpart"}
        if nwps_stage is None or not stamp:
            return {**unknown,
                    "note": f"NWPS gauge {lid} is not reporting a stage right now"}

        usgs_stage = _usgs_stage_at(site, stamp)
        if usgs_stage is None:
            return {**unknown,
                    "note": f"USGS {site} published no gage height at {stamp}"}
        difference = float(nwps_stage) - float(usgs_stage)
        agrees = abs(difference) <= DATUM_TOLERANCE_FT
        return {"agrees": bool(agrees), "difference_ft": difference,
                "nwps_stage_ft": float(nwps_stage), "usgs_stage_ft": float(usgs_stage),
                "usgs_site": site, "time": stamp,
                "note": "" if agrees else (
                    f"NWPS reads {nwps_stage:.2f} ft where USGS {site} reads "
                    f"{usgs_stage:.2f} ft at {stamp} - a {difference:+.2f} ft "
                    "difference, so the two are on different stage axes and their "
                    "curves are not directly comparable")}

    return cache.cached("noaa_datum_agreement", (lid,), build, refresh)


def _usgs_stage_at(site, timestamp):
    """USGS gage height (ft) at a timestamp, from the instantaneous-values service.

    Returns the reading closest to `timestamp` within a short window, or None.
    """
    import requests
    from .usgs import normalize_site_id, PARAM_GAGE_HEIGHT_FT

    site = normalize_site_id(site)
    wanted = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(wanted):
        return None
    window = pd.Timedelta("2h")
    try:
        response = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params={"sites": site, "parameterCd": PARAM_GAGE_HEIGHT_FT,
                    "format": "json",
                    "startDT": (wanted - window).strftime("%Y-%m-%dT%H:%MZ"),
                    "endDT": (wanted + window).strftime("%Y-%m-%dT%H:%MZ")},
            timeout=TIMEOUT)
        response.raise_for_status()
        series = response.json()["value"]["timeSeries"]
    except Exception as exc:  # noqa: BLE001
        log.info("no USGS gage height for %s at %s (%s)", site, timestamp,
                 type(exc).__name__)
        return None
    if not series:
        return None

    readings = []
    for value in series[0]["values"][0]["value"]:
        stamp = pd.to_datetime(value["dateTime"], utc=True, errors="coerce")
        try:
            reading = float(value["value"])
        except (TypeError, ValueError):
            continue
        if not pd.isna(stamp) and reading > -999:
            readings.append((abs(stamp - wanted), reading))
    if not readings:
        return None
    return min(readings)[1]
