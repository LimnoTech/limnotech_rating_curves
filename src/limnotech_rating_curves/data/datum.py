import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import settings

log = logging.getLogger(__name__)

#: How much headroom ``stage_datum="lowest"`` leaves below the stage of zero flow, as
#: a fraction of the observed stage range.
#:
#: The margin exists because of a hard constraint, not as a matter of taste. A rating
#: model estimates the **stage of zero flow** below the lowest measurement, and the
#: power-law implementation restricts that parameter to ``[0, min(stage))``. Two ways
#: to get it wrong, and this constant avoids both:
#:
#: * Reference *at* the lowest measurement and the interval collapses to nothing - the
#:   parameter has nowhere to live and the fit dies at initialization.
#: * Reference so that the true zero flow lands at **0**, the interval's lower bound,
#:   and it dies too. The truncated normal's unconstrained transform sends that bound
#:   to negative infinity, so the optimizer walks off to NaN chasing it.
#:
#: The second is why the margin is measured from the *estimated* stage of zero flow
#: rather than from the lowest measurement: any fixed offset below the lowest
#: measurement is exactly wrong for the data whose zero flow happens to sit there.
#: Measuring from the estimate puts the fitted parameter in the interior of its range
#: for any data. See :meth:`StageDatum.lowest_observed`.
LOWEST_MARGIN_FRACTION = 0.10


@dataclass
class ConvertedStage:
    """The result of a datum conversion.

    Attributes
    ----------
    stage_ft : numpy.ndarray
        Gage height (feet) - what a rating is fitted against.
    label : str
        Description of the resulting axis, for a plot or a table.
    note : str
        What was subtracted and where the number came from. This is the audit
        trail; carry it with any fit made on the converted stage.
    reference_ft : float or numpy.ndarray
        The reference elevation that was subtracted, so the conversion can be
        undone or applied to new data.
    """

    stage_ft: np.ndarray
    label: str
    note: str
    reference_ft: object

    def to_elevation(self, stage_ft=None) -> np.ndarray:
        """Undo the conversion: gage height back to the original reference.

        Parameters
        ----------
        stage_ft : array-like, optional
            Gage heights to convert back. Defaults to this object's own
            `stage_ft`.

        Returns
        -------
        numpy.ndarray
            Values on the original reference (e.g. NAVD88 feet).
        """
        values = self.stage_ft if stage_ft is None else np.asarray(stage_ft, float)
        return values + np.asarray(self.reference_ft, float)


@dataclass
class StageDatum:
    """A named vertical reference to convert stage against.

    You rarely need to build one of these - passing a number, ``"lowest"`` or a
    timeseries to ``stage_datum=`` is equivalent. Use it when you want the
    resulting plots and tables to say *which* datum was used.

    Attributes
    ----------
    kind : {'none', 'constant', 'lowest', 'timeseries', 'pointwise'}
        How the reference elevation is determined.
    elevation_ft : float or None
        The reference elevation, for ``kind="constant"``.
    series : pandas.Series or numpy.ndarray or None
        Reference elevation over time, for ``kind="timeseries"``, or one per
        measurement for ``kind="pointwise"``.
    name : str
        Short name of the datum, e.g. ``"NAVD88"``.
    label : str
        Axis label for the converted stage.
    margin_fraction : float
        For ``kind="lowest"``: how much headroom to leave below the stage of zero
        flow, as a fraction of the observed stage range. See
        :meth:`lowest_observed` for what the margin is measured from and
        :data:`LOWEST_MARGIN_FRACTION` for why it is not zero.
    """

    kind: str = "none"
    elevation_ft: float | None = None
    series: object = None
    name: str = ""
    label: str = ""
    margin_fraction: float = LOWEST_MARGIN_FRACTION

    @classmethod
    def none(cls) -> "StageDatum":
        """Stage is already gage height; convert nothing.

        Returns
        -------
        StageDatum
        """
        return cls(kind="none", label="gage height (ft)")

    @classmethod
    def constant(cls, elevation_ft: float, name: str = "", label: str = "") -> "StageDatum":
        """Subtract one fixed reference elevation from every measurement.

        Parameters
        ----------
        elevation_ft : float
            Elevation of gage height zero, in the same vertical datum as the
            stage values.
        name : str, optional
            Name of that datum, e.g. ``"NAVD88"``.
        label : str, optional
            Axis label. Defaults to a label naming the elevation.

        Returns
        -------
        StageDatum
        """
        reference = f"{elevation_ft:g} ft{' ' + name if name else ''}"
        return cls(kind="constant", elevation_ft=float(elevation_ft), name=name,
                   label=label or f"gage height above {reference} (ft)")

    @classmethod
    def usgs_gage_datum(cls, alt_va: float, alt_datum_cd: str = "NAVD88") -> "StageDatum":
        """The vertical datum a USGS gage's gage height is measured from.

        ``alt_va`` is the elevation of gage height zero, published in the USGS
        expanded site file along with ``alt_datum_cd``, the datum that elevation
        is on. Subtracting it converts a water-surface elevation to that gage's
        gage height - but only if the elevation is on the *same* datum, which is
        why the datum code is carried here and reported in the note.

        Parameters
        ----------
        alt_va : float
            Elevation of gage height zero (feet).
        alt_datum_cd : str, default "NAVD88"
            Vertical datum of `alt_va`.

        Returns
        -------
        StageDatum

        See Also
        --------
        limnotech_rating_curves.data.usgs.gage_datum : fetches these two numbers.
        """
        return cls(kind="constant", elevation_ft=float(alt_va), name=alt_datum_cd,
                   label="USGS gage height (ft)")

    @classmethod
    def lowest_observed(cls, margin_fraction: float = LOWEST_MARGIN_FRACTION
                        ) -> "StageDatum":
        """Reference stage to a little below where the flow would reach zero.

        Use this when a sensor has no surveyed tie: it gives every model a
        well-scaled stage axis without inventing a survey. The result is a *local*
        gage height - fine for fitting and for predicting at that same sensor, and
        not comparable to another site's stage.

        Where the reference goes depends on what is available:

        * with discharge (the normal case, since you are about to fit a rating),
          `margin_fraction` of the stage range **below the estimated stage of zero
          flow**, so the fitted offset lands in the interior of its allowed range;
        * with stage alone, `margin_fraction` below the lowest measurement.

        Both leave the lowest measurement above zero, which the models require. The
        first also keeps the offset off the boundary of its range, where the sampler
        diverges - see :data:`LOWEST_MARGIN_FRACTION` for why that distinction is
        not cosmetic.

        Parameters
        ----------
        margin_fraction : float, optional
            Fraction of the observed stage range to leave as headroom. Defaults to
            :data:`LOWEST_MARGIN_FRACTION`.

        Returns
        -------
        StageDatum
        """
        return cls(kind="lowest", label="local gage height (ft)",
                   margin_fraction=float(margin_fraction))

    @classmethod
    def timeseries(cls, series, name: str = "", label: str = "") -> "StageDatum":
        """A reference elevation that changes over time.

        Use this when the reference itself moves - a sensor that was raised, or a
        station whose control point was resurveyed. Each measurement is converted
        against the reference in force at its timestamp (nearest earlier value,
        so a step takes effect from its own timestamp onward).

        Parameters
        ----------
        series : pandas.Series
            Reference elevation in feet, indexed by timestamp.
        name : str, optional
            Name of the datum.
        label : str, optional
            Axis label.

        Returns
        -------
        StageDatum
        """
        series = pd.Series(series).dropna().astype(float).sort_index()
        series.index = pd.to_datetime(series.index)
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        return cls(kind="timeseries", series=series, name=name,
                   label=label or "gage height above a time-varying reference (ft)")


def as_stage_datum(datum) -> StageDatum:
    """Coerce whatever a caller passed as ``stage_datum=`` into a :class:`StageDatum`.

    Parameters
    ----------
    datum : None or float or str or pandas.Series or array-like or StageDatum
        See ``docs/data/datum.md`` for the accepted forms. A bare array is treated
        as one reference elevation per measurement; ``"lowest"`` (also
        ``"lowest_observed"``, ``"local"``) references the lowest observation.

    Returns
    -------
    StageDatum

    Raises
    ------
    ValueError
        If a string other than the recognized keywords is passed.
    """
    if datum is None:
        return StageDatum.none()
    if isinstance(datum, StageDatum):
        return datum
    if isinstance(datum, str):
        keyword = datum.strip().lower()
        if keyword in ("lowest", "lowest_observed", "local", "baseline"):
            return StageDatum.lowest_observed()
        if keyword in ("none", "gage_height", "as_is"):
            return StageDatum.none()
        raise ValueError(
            f"unknown stage_datum {datum!r}; pass a number (the elevation of gage "
            f"height zero), 'lowest', a timeseries of reference elevations, or a "
            f"StageDatum")
    if isinstance(datum, pd.Series):
        return StageDatum.timeseries(datum)
    if np.isscalar(datum):
        return StageDatum.constant(float(datum))
    values = np.asarray(datum, float).ravel()
    if values.size == 1:
        return StageDatum.constant(float(values[0]))
    return StageDatum(kind="pointwise", series=values,
                      label="gage height above a per-measurement reference (ft)")


def to_gage_height(stage, datum, time=None, discharge=None) -> ConvertedStage:
    """Convert stage onto a gage-height reference.

    Parameters
    ----------
    stage : array-like
        Stage values (feet) - a water-surface elevation, or already gage height.
    datum : None or float or str or pandas.Series or array-like or StageDatum
        The reference to convert against; see ``docs/data/datum.md``.
    time : array-like, optional
        Measurement timestamps. Required only when `datum` is a timeseries.
    discharge : array-like, optional
        Paired discharge (cfs). Used only by ``"lowest"``, which places its
        reference relative to the estimated stage of zero flow when it can - see
        :meth:`StageDatum.lowest_observed`. Without it, ``"lowest"`` falls back to
        referencing the lowest measurement.

    Returns
    -------
    ConvertedStage
        The converted stage, an axis label, a note recording the subtraction, and
        the reference that was subtracted.

    Raises
    ------
    ValueError
        If a timeseries datum is given with no `time`, or if a per-measurement
        reference has the wrong length.
    """
    stage = np.asarray(stage, float).ravel()
    spec = as_stage_datum(datum)

    if spec.kind == "none":
        return ConvertedStage(stage_ft=stage,
                              label=spec.label or "gage height (ft)",
                              note="", reference_ft=0.0)

    if spec.kind == "constant":
        reference = float(spec.elevation_ft)
        note = (f"gage height = stage - {reference:g} ft"
                f"{' ' + spec.name if spec.name else ''}")
        return ConvertedStage(stage_ft=stage - reference, label=spec.label,
                              note=note, reference_ft=reference)

    if spec.kind == "lowest":
        finite = stage[np.isfinite(stage)]
        if not finite.size:
            return ConvertedStage(stage_ft=stage, label=spec.label,
                                  note="no finite stage to reference", reference_ft=0.0)
        lowest = float(np.min(finite))
        observed_range = float(np.max(finite)) - lowest
        margin = spec.margin_fraction * (observed_range or 1.0)
        anchor, anchor_note = _low_water_anchor(stage, discharge, lowest, finite.size)
        reference = anchor - margin
        note = (f"gage height = stage - {reference:.3f} ft ({margin:.3f} ft below "
                f"{anchor_note}, so the fitted stage of zero flow lands inside its "
                f"allowed range rather than on its boundary)")
        return ConvertedStage(stage_ft=stage - reference, label=spec.label,
                              note=note, reference_ft=reference)

    if spec.kind == "pointwise":
        reference = np.asarray(spec.series, float).ravel()
        if reference.size != stage.size:
            raise ValueError(
                f"stage_datum has {reference.size} reference elevations for "
                f"{stage.size} measurements; pass one per measurement, a single "
                f"number, or a timeseries")
        note = ("gage height = stage - a per-measurement reference elevation "
                f"({np.nanmin(reference):.3f}-{np.nanmax(reference):.3f} ft)")
        return ConvertedStage(stage_ft=stage - reference, label=spec.label,
                              note=note, reference_ft=reference)

    # timeseries
    if time is None:
        raise ValueError(
            "stage_datum is a timeseries of reference elevations, so the "
            "measurements need timestamps - pass time= (or use a DataFrame with a "
            "time column)")
    reference = reference_at_times(spec.series, time)
    note = (f"gage height = stage - a time-varying reference elevation"
            f"{' (' + spec.name + ')' if spec.name else ''} "
            f"({np.nanmin(reference):.3f}-{np.nanmax(reference):.3f} ft over "
            f"{len(spec.series)} survey value(s))")
    return ConvertedStage(stage_ft=stage - reference, label=spec.label,
                          note=note, reference_ft=reference)


#: The datetime resolution every timestamp in this package is normalized to.
#:
#: pandas 2.0 and later preserve whatever resolution a source gave them, so a USGS
#: measurement time can arrive as microseconds while a sensor record is nanoseconds.
#: ``merge_asof`` refuses to join two different resolutions ("incompatible merge
#: keys"), and the failure surfaces far from its cause. Normalizing on the way in is
#: cheaper than diagnosing it twice.
_TIME_RESOLUTION = "datetime64[ns]"


def _as_naive_index(values) -> pd.DatetimeIndex:
    """A tz-naive DatetimeIndex at one fixed resolution.

    Parameters
    ----------
    values : array-like
        Timestamps in any form pandas can parse.

    Returns
    -------
    pandas.DatetimeIndex
        tz-naive, at ``_TIME_RESOLUTION``, so two indexes from different sources can
        be joined.
    """
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(values), errors="coerce",
                                            format="mixed"))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.astype(_TIME_RESOLUTION)


def _low_water_anchor(stage, discharge, lowest: float, n: int) -> tuple:
    """What ``"lowest"`` measures its margin down from, and how to describe it.

    Prefers the estimated stage of zero flow, because that is the quantity the model
    will fit and the one that must not end up on the boundary of its range. Falls
    back to the lowest measurement when there is no discharge to estimate it from.

    Parameters
    ----------
    stage : array-like
        Stage values (feet).
    discharge : array-like or None
        Paired discharge (cfs), if available.
    lowest : float
        The lowest finite stage.
    n : int
        How many finite stages there are, for the note.

    Returns
    -------
    anchor : float
        The stage the margin is measured down from.
    note : str
        A phrase describing it, for the conversion's audit note.
    """
    if discharge is not None:
        from .zero_flow import estimate_zero_flow
        try:
            estimate = estimate_zero_flow(np.asarray(stage, float),
                                          np.asarray(discharge, float))
            return (float(estimate.stage_ft),
                    f"the estimated stage of zero flow ({estimate.stage_ft:.3f} ft, "
                    f"{estimate.method})")
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail
            log.debug("could not estimate zero flow for the 'lowest' datum (%s); "
                      "referencing the lowest measurement instead", exc)
    return lowest, f"the lowest of {n} observed stages ({lowest:.3f} ft)"


def reference_at_times(series, times) -> np.ndarray:
    """The value of a stepwise reference series in force at each timestamp.

    A reference elevation is a step function: it holds until the next surveyed
    change. So this takes the most recent value at or before each requested time,
    and the first value for times before the series begins.

    Parameters
    ----------
    series : pandas.Series
        Reference elevation (feet) indexed by timestamp.
    times : array-like
        Timestamps to evaluate at.

    Returns
    -------
    numpy.ndarray
        One reference elevation per requested timestamp.
    """
    series = pd.Series(series).dropna().astype(float)
    series.index = _as_naive_index(series.index)
    series = series.sort_index()
    requested = _as_naive_index(times)
    if series.empty:
        return np.full(len(requested), np.nan)
    aligned = series.reindex(series.index.union(requested)).ffill().bfill()
    return aligned.reindex(requested).to_numpy(float)


def distance_to_stage(distance, reference_elevation, *, distance_units: str = "ft",
                      stage_units: str = "ft") -> np.ndarray:
    """Water-surface elevation from a distance-down-to-the-water reading.

    A mounted sensor measures how far below itself the water surface is, so the
    water-surface elevation is the sensor's reference elevation minus that
    distance. This is the whole conversion; the only thing that makes it
    error-prone in practice is units and a reference that moves, both of which
    are arguments here.

    Parameters
    ----------
    distance : array-like or pandas.Series
        Distance from the reference point down to the water surface.
    reference_elevation : float or array-like or pandas.Series
        Elevation of the point the distance is measured from, in `stage_units`.
        Pass a series (or the output of :func:`stepwise_reference_elevation`) when
        the sensor has been raised or lowered.
    distance_units : {'ft', 'm', 'mm'}, default 'ft'
        Units of `distance`.
    stage_units : {'ft', 'm', 'mm'}, default 'ft'
        Units of `reference_elevation` and of the returned elevation.

    Returns
    -------
    numpy.ndarray or pandas.Series
        Water-surface elevation, in `stage_units`. A Series in, a Series out (with
        its index preserved).

    Examples
    --------
    A sensor whose reference point is at 615.0 ft NAVD88, reading 3.2 ft down to
    the water:

    >>> float(distance_to_stage(3.2, 615.0))
    611.8

    Feed the result to ``stage_datum=`` to get gage height, or pass
    ``stage_datum="lowest"`` if the site has no surveyed tie.
    """
    factor = _length_factor(distance_units, stage_units)
    if isinstance(distance, pd.Series):
        converted = distance.astype(float) * factor
        if isinstance(reference_elevation, pd.Series):
            reference = reference_elevation.reindex(converted.index).ffill().bfill()
        else:
            reference = reference_elevation
        return (reference - converted).rename(distance.name)
    converted = np.asarray(distance, float) * factor
    return np.asarray(reference_elevation, float) - converted


_LENGTH_IN_MM = {"ft": settings.MM_PER_FOOT, "feet": settings.MM_PER_FOOT,
                 "foot": settings.MM_PER_FOOT,
                 "m": settings.MM_PER_METER, "meter": settings.MM_PER_METER,
                 "meters": settings.MM_PER_METER,
                 "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0,
                 "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0}


def _length_factor(from_units: str, to_units: str) -> float:
    """Multiplier converting a length from `from_units` to `to_units`."""
    try:
        return _LENGTH_IN_MM[from_units.lower()] / _LENGTH_IN_MM[to_units.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown length unit {exc.args[0]!r}; use one of "
                         f"{sorted(set(_LENGTH_IN_MM))}") from exc


def stepwise_reference_elevation(index, base_elevation_ft: float,
                                 changes=()) -> pd.Series:
    """A sensor's reference elevation over time, stepping at each recorded move.

    When a sensor is physically raised or lowered, its distance readings jump by
    the move amount. Stage stays continuous only if the reference elevation steps
    with it, which means every move has to be applied from its own timestamp
    onward and cumulatively with earlier moves.

    Parameters
    ----------
    index : array-like
        Timestamps to produce a reference elevation for.
    base_elevation_ft : float
        The reference elevation before any move.
    changes : sequence of (timestamp, delta_ft)
        Recorded moves: how much the reference elevation changed, and when.
        Order does not matter; they are applied in time order and accumulate.

    Returns
    -------
    pandas.Series
        Reference elevation (feet) indexed by `index`.

    Examples
    --------
    >>> idx = pd.to_datetime(["2025-01-01", "2025-06-01"])
    >>> stepwise_reference_elevation(idx, 615.0, [("2025-03-01", 0.5)]).tolist()
    [615.0, 615.5]
    """
    stamps = _as_naive_index(index)
    elevation = np.full(len(stamps), float(base_elevation_ft))
    cumulative = 0.0
    for when, delta in sorted(changes, key=lambda change: pd.Timestamp(change[0])):
        cumulative += float(delta)
        elevation[stamps >= pd.Timestamp(when)] = float(base_elevation_ft) + cumulative
    return pd.Series(elevation, index=stamps, name="reference_elevation_ft")


def stage_at_times(stage_series, times, tolerance="3h") -> pd.DataFrame:
    """Match a continuous stage record to the instants a discharge was measured.

    A manual discharge measurement happens at a moment; the sensor records on its
    own schedule. This takes the nearest reading within `tolerance` for each
    measurement time and reports how far away it was, so a pair matched across a
    two-hour gap on a rising limb can be seen for what it is rather than trusted
    silently.

    Parameters
    ----------
    stage_series : pandas.Series
        Continuous stage (feet), indexed by timestamp.
    times : array-like
        The measurement timestamps to match.
    tolerance : str or pandas.Timedelta, default '3h'
        Largest acceptable gap between a measurement and the stage reading used
        for it. Readings further away are not used and come back NaN.

    Returns
    -------
    pandas.DataFrame
        Indexed by the requested timestamps, with columns ``stage_ft``,
        ``stage_time`` (the reading actually used) and ``lag_min`` (the gap in
        minutes). Rows with no reading in tolerance have NaN stage.
    """
    requested = _as_naive_index(times)

    out = pd.DataFrame(index=requested, columns=["stage_ft", "lag_min"], dtype=float)
    out["stage_time"] = pd.NaT
    out = out[["stage_ft", "stage_time", "lag_min"]]

    stage = pd.Series(stage_series).dropna().astype(float)
    if stage.empty:
        return out
    # both sides normalized to one resolution, or merge_asof refuses to join them
    stage.index = _as_naive_index(stage.index)
    stage = stage[~stage.index.duplicated(keep="first")].sort_index()

    left = pd.DataFrame({"requested_time": requested}).sort_values("requested_time")
    right = pd.DataFrame({"stage_time": stage.index, "stage_ft": stage.to_numpy()})
    matched = pd.merge_asof(left, right, left_on="requested_time",
                            right_on="stage_time", direction="nearest",
                            tolerance=pd.Timedelta(tolerance)).set_index("requested_time")

    out["stage_ft"] = matched["stage_ft"]
    out["stage_time"] = matched["stage_time"]
    gap = (out.index - out["stage_time"]).abs()
    out["lag_min"] = gap.dt.total_seconds() / 60.0
    unmatched = out["stage_ft"].isna()
    out.loc[unmatched, "stage_time"] = pd.NaT
    out.loc[unmatched, "lag_min"] = np.nan
    if unmatched.any():
        log.info("%d of %d measurement time(s) had no stage reading within %s",
                 int(unmatched.sum()), len(out), tolerance)
    return out
