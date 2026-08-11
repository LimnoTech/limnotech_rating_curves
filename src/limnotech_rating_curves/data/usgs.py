import io
import logging
import re

import numpy as np
import pandas as pd

from ..support import cache
from ..core import Sample

log = logging.getLogger(__name__)

#: USGS parameter code for discharge, cubic feet per second.
PARAM_DISCHARGE_CFS = "00060"

#: USGS parameter code for gage height, feet.
PARAM_GAGE_HEIGHT_FT = "00065"

#: Root of the Water Data STAC API, whose ``ratings`` collection holds the
#: published rating files.
STAC_ROOT = "https://api.waterdata.usgs.gov/stac/v0"
_RATING_ITEM_URL = STAC_ROOT + "/collections/ratings/items/USGS-{site}.{file_type}.rdb"

#: The three rating files a site may publish.
#:
#: ``exsa``  expanded, shift-adjusted rating - the interpolated ~0.01-ft table you
#:           look an observed gage height up in. The shift is already applied.
#: ``base``  the base rating points the expanded table was built from.
#: ``corr``  stage corrections.
RATING_FILE_TYPES = ("exsa", "base", "corr")

_SITE_FILE_URL = ("https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={site}"
                  "&siteOutput=expanded&siteStatus=all")

# RDB header lines look like:  # //RATING OFFSET1=2.000000E+00
_HEADER_FIELDS = re.compile(r'(\w+)="?([^"]*?)"?(?=\s+\w+=|\s*$)')
_RATING_OFFSET = re.compile(r"RATING OFFSET(\d+)=(.+)$")


def normalize_site_id(site) -> str:
    """A USGS site id as a zero-padded string.

    Parameters
    ----------
    site : str or int
        The site number. An integer is accepted but is a warning sign - leading
        zeros are significant, so an integer has already lost information if the
        id had one.

    Returns
    -------
    str
        The id, left-padded with zeros to eight characters. Ids longer than eight
        characters (some sites have nine) are returned unchanged.
    """
    text = str(site).strip()
    return text.zfill(8) if len(text) < 8 else text


def _monitoring_location_ids(sites) -> list:
    """Site ids in the ``USGS-<id>`` form the Water Data API expects."""
    if isinstance(sites, str):
        sites = [sites]
    return [site if str(site).startswith("USGS-") else f"USGS-{site}"
            for site in sites]


def _time_interval(start=None, end=None):
    """An RFC-3339 interval string for the Water Data API, or None for all time."""
    if start is None and end is None:
        return None

    def stamp(value, end_of_day=False):
        """One end of the interval as an RFC-3339 instant, or '..' for open."""
        if value is None:
            return ".."
        text = str(value)
        if "T" in text:
            return text
        return f"{text}T23:59:59Z" if end_of_day else f"{text}T00:00:00Z"

    return f"{stamp(start)}/{stamp(end, end_of_day=True)}"


# ---------------------------------------------------------------------------
# field measurements - the sample a rating is fitted to
# ---------------------------------------------------------------------------

def measurements(site, start=None, end=None, refresh: bool = False) -> pd.DataFrame:
    """Field-measured stage-discharge pairs for one gage.

    One row per discharge measurement, with the mean gage height read at the same
    visit. These are independent observations of the rating, not points off the
    published curve.

    Parameters
    ----------
    site : str
        USGS site id, e.g. ``"04176356"``.
    start, end : str, optional
        ``YYYY-MM-DD`` bounds. ``None`` for the whole record.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``stage_ft``, ``discharge_cfs``, sorted by time.
        Measurements missing either value are dropped.
    """
    site = normalize_site_id(site)

    def build():
        """Fetch the field measurements and pivot them to (stage, discharge) pairs."""
        from dataretrieval import waterdata
        raw, _ = waterdata.get_field_measurements(
            monitoring_location_id=_monitoring_location_ids(site),
            time=_time_interval(start, end))
        readings = raw[raw["reading_type"].isin(["Discharge", "MeanGageHeight"])]
        wide = readings.pivot_table(
            index=["monitoring_location_id", "field_visit_id", "time"],
            columns="reading_type", values="value", aggfunc="first").reset_index()
        wide = wide.rename(columns={"Discharge": "discharge_cfs",
                                    "MeanGageHeight": "stage_ft"})
        wide = wide.dropna(subset=["discharge_cfs", "stage_ft"])
        return (wide[["time", "stage_ft", "discharge_cfs"]]
                .sort_values("time").reset_index(drop=True))

    return cache.cached("usgs_measurements", (site, start, end), build, refresh)


def gage_sample(site, start=None, end=None, refresh: bool = False) -> Sample:
    """A gage's field measurements as a :class:`Sample`, ready to fit.

    Parameters
    ----------
    site : str
        USGS site id.
    start, end : str, optional
        ``YYYY-MM-DD`` bounds.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    Sample
        Stage on the gage's own gage-height axis, so no datum conversion is needed
        and a fit is directly comparable with the published rating.

    Examples
    --------
    >>> sample = gage_sample("04176356")            # doctest: +SKIP
    >>> sample.compare(models="all").metrics         # doctest: +SKIP
    """
    site = normalize_site_id(site)
    frame = measurements(site, start=start, end=end, refresh=refresh)
    sample = Sample.of(frame, site_id=site, source="usgs_gage",
                       stage_label="USGS gage height (ft)",
                       datum_note="stage as read at the gage, above its own datum")
    log.info("USGS %s: %d field measurement(s), stage %.2f-%.2f ft, "
             "discharge %.3g-%.3g cfs", site, len(sample),
             *sample.stage_range, *sample.discharge_range)
    return sample


def continuous_discharge(site, start=None, end=None) -> pd.DataFrame:
    """Instantaneous recorded discharge for one gage.

    Not used for fitting - a rating is fitted to field measurements - but useful
    for checking a fitted rating against the gage's own published record.

    Parameters
    ----------
    site : str
        USGS site id.
    start, end : str, optional
        ``YYYY-MM-DD`` bounds.

    Returns
    -------
    pandas.DataFrame
        The API's frame, with ``time`` and ``value`` (cfs) columns.
    """
    from dataretrieval import waterdata
    frame, _ = waterdata.get_continuous(
        monitoring_location_id=_monitoring_location_ids(normalize_site_id(site)),
        parameter_code=PARAM_DISCHARGE_CFS, time=_time_interval(start, end))
    return frame


def continuous_stage(site, start=None, end=None) -> pd.DataFrame:
    """Instantaneous recorded gage height for one gage.

    Parameters
    ----------
    site : str
        USGS site id.
    start, end : str, optional
        ``YYYY-MM-DD`` bounds.

    Returns
    -------
    pandas.DataFrame
        The API's frame, with ``time`` and ``value`` (feet) columns.
    """
    from dataretrieval import waterdata
    frame, _ = waterdata.get_continuous(
        monitoring_location_id=_monitoring_location_ids(normalize_site_id(site)),
        parameter_code=PARAM_GAGE_HEIGHT_FT, time=_time_interval(start, end))
    return frame


def continuous_record(site, start, end, *, approved_only: bool = True,
                      refresh: bool = False) -> pd.DataFrame:
    """Gage height and discharge from the gage's own continuous record, paired.

    Every timestamp at which the gage reported both values - which is the rating the
    gage was operated under, read back out of its published record. It is not a
    sample to fit to: the discharge was computed *from* the stage through the
    rating, so the pairs are the rating's own output, not independent observations
    of it. What they are good for is asking whether that rating was one curve or
    several, and when it moved. Fit to :func:`gage_sample` instead.

    Fetched and cached a calendar year at a time, so a long span can be built up
    over several calls and interrupted without losing what it already has. A year of
    15-minute data is around 35,000 rows per parameter and takes a few seconds.

    Parameters
    ----------
    site : str
        USGS site id.
    start, end : str
        ``YYYY-MM-DD`` bounds. Both are required: the full record at a long-running
        gage is a million points, so how much of it you want is your decision.
    approved_only : bool, default True
        Keep only rows both of whose values are approved. Provisional data has not
        been through the rating review that this analysis is about, so mixing it in
        would answer a different question.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``stage_ft``, ``discharge_cfs``, ``approval_status`` and
        ``year``, sorted by time.
    """
    site = normalize_site_id(site)
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    frames = [_continuous_year(site, year, refresh) for year in years]
    record = pd.concat(frames, ignore_index=True) if frames else _empty_record()
    if record.empty:
        log.warning("%s: no continuous stage-discharge pairs in %s-%s", site,
                    start, end)
        return record

    within = ((record["time"] >= pd.Timestamp(start, tz="UTC"))
              & (record["time"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)))
    record = record[within]
    if approved_only:
        record = record[record["approval_status"] == "Approved"]
    record = record.sort_values("time").reset_index(drop=True)
    log.info("%s: %d paired readings, %s to %s", site, len(record),
             record["time"].min().date(), record["time"].max().date())
    return record


def _empty_record() -> pd.DataFrame:
    """The shape :func:`continuous_record` returns when there is nothing to return."""
    return pd.DataFrame(columns=["time", "stage_ft", "discharge_cfs",
                                 "approval_status", "year"])


def _continuous_year(site, year: int, refresh: bool = False) -> pd.DataFrame:
    """One calendar year of paired readings, cached on its own."""
    def build():
        """Fetch both parameters for the year and pair them on the timestamp."""
        window = (f"{year}-01-01", f"{year}-12-31")
        stage = continuous_stage(site, *window)
        discharge = continuous_discharge(site, *window)
        if stage.empty or discharge.empty:
            return _empty_record()
        columns = ["time", "value", "approval_status"]
        paired = pd.merge(stage[columns].rename(columns={"value": "stage_ft"}),
                          discharge[columns].rename(columns={"value": "discharge_cfs"}),
                          on="time", suffixes=("_stage", "_discharge"))
        # one status for the pair: approved only when both readings are
        paired["approval_status"] = np.where(
            (paired["approval_status_stage"] == "Approved")
            & (paired["approval_status_discharge"] == "Approved"),
            "Approved", "Provisional")
        paired["year"] = year
        return paired[["time", "stage_ft", "discharge_cfs", "approval_status", "year"]]

    return cache.cached("usgs_continuous_year", (site, year), build, refresh)


def rating_stability(record, *, bin_width: float = 0.1, min_readings: int = 500,
                     min_per_year: int = 20) -> pd.DataFrame:
    """Split the discharge spread at each stage into within-year and between-year.

    Answers "was this site operated under one rating?" by asking, at each stage, how
    much the reported discharge varies. A single unchanging rating gives 1.00 for
    both columns.

    The split is what makes it readable. ``within_year`` is the 5th-to-95th
    percentile ratio inside one year, which is loop and backwater: the same stage
    carrying different flow on a rising limb than a falling one, or a downstream
    river holding the gage up. ``between_year`` is the ratio of the highest to the
    lowest yearly median, which is the rating itself being rebuilt. Two very
    different problems, and only the second is a rating change.

    Parameters
    ----------
    record : pandas.DataFrame
        Paired readings from :func:`continuous_record`.
    bin_width : float, default 0.1
        Stage bin width, feet.
    min_readings : int, default 500
        Drop bins with fewer readings than this over the whole record.
    min_per_year : int, default 20
        Drop a year from a bin when it holds fewer readings than this, so one
        afternoon's data does not become a year's median.

    Returns
    -------
    pandas.DataFrame
        Indexed by stage bin, with ``readings``, ``years``, ``median_cfs``,
        ``within_year`` and ``between_year``.
    """
    if record.empty:
        return pd.DataFrame(columns=["readings", "years", "median_cfs",
                                     "within_year", "between_year"])
    binned = record.assign(bin=(record["stage_ft"] / bin_width).round() * bin_width)
    readings = binned.groupby("bin").size()
    binned = binned[binned["bin"].isin(readings[readings >= min_readings].index)]

    per_year = binned.groupby(["bin", "year"])["discharge_cfs"].agg(
        n="size", median="median",
        low=lambda values: values.quantile(0.05),
        high=lambda values: values.quantile(0.95))
    per_year = per_year[per_year["n"] >= min_per_year]
    per_year["within"] = per_year["high"] / per_year["low"]

    table = per_year.groupby("bin").agg(
        readings=("n", "sum"), years=("median", "size"),
        median_cfs=("median", "median"), within_year=("within", "median"),
        between_year=("median", lambda values: values.max() / values.min()))
    return table.round(3)


def rating_deviation(site, sample=None, refresh: bool = False) -> pd.DataFrame:
    """Every field measurement against the published rating.

    The comparison a rating is maintained by. ``percent_diff`` is how far the
    measured discharge sits from the rating at the stage it was measured at, and
    ``shift_ft`` is the same disagreement expressed the way the USGS records it: the
    stage correction that would make the rating pass through the measurement.
    Plotted against time, a drift in either is the rating moving under the gage.

    Parameters
    ----------
    site : str
        USGS site id.
    sample : Sample or pandas.DataFrame, optional
        The measurements. Defaults to the gage's own field measurements.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``stage_ft``, ``discharge_cfs``, ``rating_cfs``,
        ``percent_diff``, ``shift_ft``. Measurements outside the published stage or
        discharge range come back NaN rather than extrapolated. Empty when the site
        publishes no rating.
    """
    site = normalize_site_id(site)
    curve = published_rating(site, refresh=refresh)
    if curve is None or curve.empty:
        log.info("%s publishes no rating, so there is nothing to deviate from", site)
        return pd.DataFrame(columns=["time", "stage_ft", "discharge_cfs", "rating_cfs",
                                     "percent_diff", "shift_ft"])
    if sample is None:
        sample = measurements(site, refresh=refresh)
    frame = sample.to_frame() if hasattr(sample, "to_frame") else pd.DataFrame(sample)

    stage = frame["stage_ft"].to_numpy(float)
    observed = frame["discharge_cfs"].to_numpy(float)
    rating_stage = curve["stage_ft"].to_numpy(float)
    rating_discharge = curve["discharge_cfs"].to_numpy(float)

    predicted = np.interp(stage, rating_stage, rating_discharge,
                          left=np.nan, right=np.nan)
    # the stage the rating would have to be read at to return the measured
    # discharge; the rating is monotone, so it inverts by interpolating the other way
    stage_for_observed = np.interp(observed, rating_discharge, rating_stage,
                                   left=np.nan, right=np.nan)
    return pd.DataFrame({
        "time": frame.get("time"),
        "stage_ft": stage,
        "discharge_cfs": observed,
        "rating_cfs": predicted,
        "percent_diff": 100 * (observed - predicted) / predicted,
        "shift_ft": stage_for_observed - stage})


# ---------------------------------------------------------------------------
# gage datum - for converting an elevation to that gage's gage height
# ---------------------------------------------------------------------------

def site_info(sites, refresh: bool = False) -> pd.DataFrame:
    """Expanded site-file metadata for one or more gages.

    Parameters
    ----------
    sites : str or sequence of str
        Site id(s).
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        Indexed by site id, with ``station_name``, ``latitude``, ``longitude``,
        ``alt_va`` (elevation of gage height zero, feet), ``alt_datum_cd`` (the
        datum that elevation is on), ``alt_acy_va`` (its stated accuracy) and
        ``drainage_area_sqmi``.
    """
    if isinstance(sites, str):
        sites = [sites]
    ids = sorted({normalize_site_id(site) for site in sites})

    def build():
        """Fetch and parse each site's expanded site-file record."""
        import requests
        rows = {}
        for site in ids:
            try:
                text = requests.get(_SITE_FILE_URL.format(site=site), timeout=30).text
                frame = pd.read_csv(io.StringIO(text), sep="\t", comment="#",
                                    header=[0], dtype=str)
                record = frame.iloc[1]          # row 0 is the RDB type descriptor
            except Exception as exc:  # noqa: BLE001
                log.warning("no site file for %s: %s", site, exc)
                continue
            rows[site] = {
                "station_name": record.get("station_nm"),
                "latitude": _as_float(record.get("dec_lat_va")),
                "longitude": _as_float(record.get("dec_long_va")),
                "alt_va": _as_float(record.get("alt_va")),
                "alt_datum_cd": str(record.get("alt_datum_cd") or "").strip(),
                "alt_acy_va": _as_float(record.get("alt_acy_va")),
                "drainage_area_sqmi": _as_float(record.get("drain_area_va")),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    return cache.cached("usgs_site_info", (tuple(ids),), build, refresh)


def _as_float(value) -> float:
    """A site-file field as a float, NaN when blank or unparseable."""
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def gage_datum(site, refresh: bool = False):
    """The vertical datum a gage's gage height is measured from.

    ``alt_va`` is the elevation of gage height zero and ``alt_datum_cd`` is the
    datum that elevation is on. For a water surface whose elevation is on the
    **same** datum,

        gage height = water-surface elevation − alt_va

    When the datums differ the subtraction is invalid without a conversion (VERTCON
    for NGVD29 to NAVD88), which is why the datum code is returned rather than
    assumed.

    Parameters
    ----------
    site : str
        USGS site id.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    StageDatum
        Ready to pass as ``stage_datum=`` when fitting a rating to a
        surveyed water-surface elevation at that gage.

    Raises
    ------
    KeyError
        If the site has no site-file record.

    Examples
    --------
    >>> datum = gage_datum("04176356")                              # doctest: +SKIP
    >>> fit_rating(elevations, flows, stage_datum=datum)             # doctest: +SKIP
    """
    from .datum import StageDatum
    site = normalize_site_id(site)
    info = site_info(site, refresh=refresh)
    if site not in info.index:
        raise KeyError(f"no USGS site-file record for {site!r}")
    return StageDatum.usgs_gage_datum(float(info.loc[site, "alt_va"]),
                                      str(info.loc[site, "alt_datum_cd"]))


def coordinates(sites, refresh: bool = False) -> dict:
    """Gage coordinates, for the map.

    Parameters
    ----------
    sites : str or sequence of str
        Site id(s).
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    dict
        ``{site id: (latitude, longitude)}``, omitting sites with no coordinates.
    """
    info = site_info(sites, refresh=refresh)
    return {site: (float(row["latitude"]), float(row["longitude"]))
            for site, row in info.iterrows()
            if np.isfinite(row["latitude"]) and np.isfinite(row["longitude"])}


# ---------------------------------------------------------------------------
# published ratings - the reference to compare a fit against
# ---------------------------------------------------------------------------

def _parse_rdb_header(lines) -> dict:
    """Pull the useful RATING / STATION fields out of an RDB comment block."""
    meta = {}
    for raw in lines:
        line = raw.lstrip("#/ ").strip()
        offset = _RATING_OFFSET.match(line)
        if offset:
            number, value = int(offset.group(1)), float(offset.group(2))
            meta.setdefault("offsets", {})[number] = value
            if number == 1:
                meta["offset"] = value
        elif line.startswith("RATING EXPANSION="):
            meta["expansion"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("STATION NAME="):
            meta["station_name"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("RATING ID="):
            meta.update({key.lower(): value for key, value
                         in _HEADER_FIELDS.findall(line[len("RATING "):])})
        elif line.startswith("PARAMETER CODE="):
            meta["parameter_code"] = line.split("=", 1)[1].strip().strip('"')
    return meta


def _parse_rdb(text: str) -> pd.DataFrame:
    """Parse an NWIS RDB rating file into a numeric frame with header metadata."""
    header = [line for line in text.splitlines() if line.startswith("#")]
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    frame = pd.read_csv(io.StringIO(body), sep="\t", dtype=str)
    frame = frame.iloc[1:].reset_index(drop=True)     # drop the '16N/1S' format row
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(how="all", axis=1)
    frame.attrs["meta"] = _parse_rdb_header(header)
    return frame


def rating_table(site, file_type: str = "exsa", refresh: bool = False) -> pd.DataFrame:
    """One of a site's published rating files, raw.

    Parameters
    ----------
    site : str
        USGS site id.
    file_type : {'exsa', 'base', 'corr'}, default 'exsa'
        Which file (see ``RATING_FILE_TYPES``).
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        The file's own columns (``INDEP``, ``SHIFT``, ``DEP``, ``STOR``, ...), with
        the parsed header on ``df.attrs["meta"]``.

    Raises
    ------
    ValueError
        On an unknown `file_type`.
    """
    site = normalize_site_id(site)
    if file_type not in RATING_FILE_TYPES:
        raise ValueError(f"file_type must be one of {RATING_FILE_TYPES}, "
                         f"got {file_type!r}")

    def build():
        """Fetch the STAC item, then the RDB file it links to."""
        import requests
        item = requests.get(_RATING_ITEM_URL.format(site=site, file_type=file_type),
                            timeout=30)
        item.raise_for_status()
        payload = item.json()
        text = requests.get(payload["assets"]["data"]["href"], timeout=30).text
        frame = _parse_rdb(text)
        frame.attrs["meta"].update({"site": site, "file_type": file_type})
        return frame

    return cache.cached(f"usgs_rating_{file_type}", (site,), build, refresh)


def published_rating(site, refresh: bool = False) -> pd.DataFrame:
    """The gage's published stage-discharge rating, as a tidy curve.

    This is the USGS's own maintained rating (the expanded, shift-adjusted table),
    the natural reference to hold a fitted curve against. It is a product, not
    data: do not fit to it.

    Parameters
    ----------
    site : str
        USGS site id.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage_ft``, ``shift_ft`` and ``discharge_cfs`` (with the shift
        already applied), ascending in stage. An empty frame when the site
        publishes no rating - which is common, and is not an error.
    """
    site = normalize_site_id(site)

    def build():
        """Tidy the expanded rating table, or an empty frame if there is none."""
        try:
            table = rating_table(site, "exsa", refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.info("no published rating for %s (%s)", site, type(exc).__name__)
            return pd.DataFrame(columns=["stage_ft", "shift_ft", "discharge_cfs"])
        curve = pd.DataFrame({"stage_ft": table["INDEP"],
                              "shift_ft": table.get("SHIFT"),
                              "discharge_cfs": table["DEP"]})
        return (curve.dropna(subset=["discharge_cfs"])
                .sort_values("stage_ft").reset_index(drop=True))

    return cache.cached("usgs_published_rating", (site,), build, refresh)


def published_reference(site, sample, refresh: bool = False):
    """The published rating packaged as a comparison reference for a sample.

    Scores the published curve on the same measurements a fit was made to, so it
    appears in the comparison table as one more row.

    Parameters
    ----------
    site : str
        USGS site id.
    sample : Sample
        The measurements. Its stage must be on the gage's gage-height axis for the
        comparison to be valid - that is what the published curve is indexed by.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    dict or None
        ``{"label", "curve", "metrics"}``, or None when the site publishes no
        rating. Measurements outside the published stage range are ignored rather
        than extrapolated, so ``metrics["n"]`` may be below ``len(sample)``.
    """
    from ..core import fit_metrics
    curve = published_rating(site, refresh=refresh)
    if curve is None or curve.empty or len(sample) == 0:
        return None
    predicted = np.interp(sample.stage_ft, curve["stage_ft"], curve["discharge_cfs"],
                          left=np.nan, right=np.nan)
    return {"label": "USGS published rating",
            "curve": curve[["stage_ft", "discharge_cfs"]],
            "metrics": fit_metrics(sample.discharge_cfs, predicted)}


def segment_breakpoints(site, threshold: float = 0.15,
                        refresh: bool = False) -> pd.DataFrame:
    """Stages where the published rating changes log-log slope.

    A logarithmic rating is piecewise-linear in ``log(h - offset)`` against
    ``log Q``, so the vertices between its segments show up as changes in the local
    exponent. Useful for sanity-checking how many segments a fitted power law
    should have.

    This fits nothing of its own: it reads the base rating points and the single
    published log offset, and flags the points where the slope changes by more than
    `threshold`. Note that the API publishes **one** offset per rating, not one per
    segment. Runs of consecutive flagged points - common at the low end, where
    ``h - offset`` is small and the exponent is numerically unstable - are collapsed
    to the largest change in the run.

    Parameters
    ----------
    site : str
        USGS site id.
    threshold : float, default 0.15
        Smallest change in the log-log slope treated as a breakpoint.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    pandas.DataFrame
        Columns ``stage_ft``, ``discharge_cfs``, ``slope_below``, ``slope_above``,
        ``delta_slope``. Empty when the base rating has fewer than three usable
        points.
    """
    columns = ["stage_ft", "discharge_cfs", "slope_below", "slope_above",
               "delta_slope"]
    base = rating_table(site, "base", refresh=refresh)
    offset = float(base.attrs.get("meta", {}).get("offset", 0.0))
    points = pd.DataFrame({"stage_ft": base["INDEP"],
                           "discharge_cfs": base["DEP"]}).dropna()
    points = points[(points["stage_ft"] - offset > 0) & (points["discharge_cfs"] > 0)]
    points = (points.sort_values("stage_ft").drop_duplicates("stage_ft")
              .reset_index(drop=True))
    if len(points) < 3:
        return pd.DataFrame(columns=columns)

    x = np.log(points["stage_ft"].to_numpy(float) - offset)
    y = np.log(points["discharge_cfs"].to_numpy(float))
    slope = np.diff(y) / np.diff(x)
    slope_change = np.diff(slope)
    candidates = np.where(np.abs(slope_change) > threshold)[0] + 1

    runs, rows = [], []
    for index in candidates:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    for run in runs:
        index = max(run, key=lambda position: abs(slope_change[position - 1]))
        rows.append({"stage_ft": float(points["stage_ft"].iloc[index]),
                     "discharge_cfs": float(points["discharge_cfs"].iloc[index]),
                     "slope_below": float(slope[index - 1]),
                     "slope_above": float(slope[index]),
                     "delta_slope": float(slope[index] - slope[index - 1])})
    return pd.DataFrame(rows, columns=columns)
