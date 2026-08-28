import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Column names :meth:`Sample.of` recognizes for stage, in order of preference.
STAGE_COLUMNS = ("stage_ft", "stage", "gage_height_ft", "gage_height",
                 "water_level_ft", "wse_ft", "elevation_ft", "h")

#: Column names :meth:`Sample.of` recognizes for discharge, in order of preference.
DISCHARGE_COLUMNS = ("discharge_cfs", "discharge", "q_cfs", "flow_cfs", "flow", "q")

#: Column names :meth:`Sample.of` recognizes for a measurement timestamp.
TIME_COLUMNS = ("time", "timestamp", "datetime", "date")


def padded_stage_grid(stage, points=None, pad_fraction=None) -> np.ndarray:
    """The stage grid a fitted curve is tabulated and drawn on.

    Every model family is given this grid, so their curves cover the same stages
    and are comparable end to end rather than each stopping where its own backend
    happens to stop.

    Parameters
    ----------
    stage : array-like
        Measured stages (feet).
    points : int, optional
        Number of grid points. Defaults to ``settings.GRID_POINTS``.
    pad_fraction : float, optional
        Fraction of the observed stage range to extend past each end. Defaults to
        ``settings.GRID_PAD_FRACTION``.

    Returns
    -------
    numpy.ndarray
        Ascending stage values (feet).
    """
    from . import settings
    points = settings.GRID_POINTS if points is None else int(points)
    pad_fraction = (settings.GRID_PAD_FRACTION if pad_fraction is None
                    else float(pad_fraction))
    stage = np.asarray(stage, float)
    low, high = float(np.nanmin(stage)), float(np.nanmax(stage))
    pad = pad_fraction * ((high - low) or 1.0)
    return np.linspace(low - pad, high + pad, points)


def _pick_column(frame: pd.DataFrame, candidates, what: str, explicit=None) -> str:
    """Resolve which column of `frame` holds `what`.

    Parameters
    ----------
    frame : pandas.DataFrame
        The frame to search.
    candidates : sequence of str
        Accepted names, most-preferred first. Matching is case-insensitive and
        ignores spaces, so ``"Stage (ft)"`` matches ``"stage_ft"`` only if the
        exact normalized name appears in `candidates` - this is a name lookup,
        not a fuzzy search.
    what : str
        Human name of the quantity, used in the error message.
    explicit : str, optional
        A column the caller named. Used if present, and an error if it is absent
        from the frame.

    Returns
    -------
    str
        The resolved column name.

    Raises
    ------
    KeyError
        If no candidate column is present.
    """
    if explicit is not None:
        if explicit not in frame.columns:
            raise KeyError(f"no column {explicit!r} in the data "
                           f"(columns: {list(frame.columns)})")
        return explicit
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in frame.columns}
    for name in candidates:
        if name in normalized:
            return normalized[name]
    raise KeyError(
        f"could not find the {what} column. Rename one of {list(frame.columns)} to "
        f"one of {list(candidates)}, or pass it explicitly "
        f"(e.g. {what}='my_column').")


@dataclass
class Sample:
    """Paired stage-discharge measurements, and where they came from.

    Build one with :meth:`of` rather than the constructor unless you already have
    clean arrays.

    Attributes
    ----------
    stage_ft : numpy.ndarray
        Stage (feet), one value per measurement. Which vertical reference the
        numbers are on is recorded in `stage_label`, not enforced here.
    discharge_cfs : numpy.ndarray
        Discharge (cubic feet per second), same length as `stage_ft`.
    site_id : str
        The site these measurements came from (a USGS gage number, a station
        name), or ``""`` for data you assembled yourself. It titles a plot and
        names an export file, so it stays the bare place - the key a run over many sites
        files this sample under is :attr:`SiteRating.sample_id`, which qualifies
        the place with the construction that produced it.
    source : str
        Which construction produced the sample, for grouping in reports.
    sample_id : str
        This sample's key: ``"<source>:<site>"``, derived from `site_id` and `source`
        at construction. Not passed in - there is one way to set it.

        A rating pairs a stage record with a discharge record, and `site_id` names
        only the stage instrument. Sensor SBR-01 feeds two ratings - its stage
        against MAGL flow-sheet discharge, and the same stage against USGS field
        gagings - and both carry ``site_id="SBR-01_WL"``. `source` is the other half,
        so together they name the rating: ``"magl:SBR-01"``, ``"colocated:SBR-01"``.
    stage_label : str
        Human description of the stage axis, e.g. ``"USGS gage height (ft)"`` or
        ``"water-surface elevation (NAVD88 ft)"``. This is what a plot's x-axis is
        labelled with and the one thing that keeps a fit from being read off the
        wrong datum.
    datum_note : str
        How the stage was tied to that reference (the subtraction applied, the
        baseline used).
    datum_reference_ft : float
        The elevation that was subtracted to reach this stage axis, when it was a
        single number - so ``stage_ft + datum_reference_ft`` returns the original
        elevations, and a fitted rating can be applied to a continuous record that
        is still on the original datum. NaN when the reference varied per
        measurement, 0 when nothing was subtracted.
    time : numpy.ndarray or None
        Measurement timestamps, if known.
    skipped : str
        Non-empty when the sample is deliberately empty, saying why (e.g. a datum
        mismatch that makes the tie invalid). An empty sample with an empty
        `skipped` simply had no data.
    """

    stage_ft: np.ndarray
    discharge_cfs: np.ndarray
    site_id: str = ""
    source: str = ""
    sample_id: str = field(init=False, default="")
    stage_label: str = "stage (ft)"
    datum_note: str = ""
    datum_reference_ft: float = 0.0
    time: object = None
    skipped: str = ""

    def __post_init__(self):
        site = self.site_id.removesuffix("_WL")
        self.sample_id = f"{self.source}:{site}" if site and self.source else site
        self.stage_ft = np.asarray(self.stage_ft, float).ravel()
        self.discharge_cfs = np.asarray(self.discharge_cfs, float).ravel()
        if self.stage_ft.size != self.discharge_cfs.size:
            raise ValueError(
                f"stage and discharge must be the same length, got "
                f"{self.stage_ft.size} and {self.discharge_cfs.size}")

    # -- construction ---------------------------------------------------------

    @classmethod
    def of(cls, data, discharge=None, *, stage=None, time=None,
           site_id: str = "", source: str = "", stage_label=None,
           datum_note: str = "", stage_datum=None, drop_nonpositive: bool = True
           ) -> "Sample":
        """Build a Sample from whatever stage-discharge data you have.

        Parameters
        ----------
        data : Sample or pandas.DataFrame or array-like
            One of:

            * a :class:`Sample` - returned unchanged (any other argument is
              ignored, so this is safe to call on already-prepared data);
            * a :class:`pandas.DataFrame` with a stage column and a discharge
              column (see `stage` / `discharge` and the module's
              ``STAGE_COLUMNS`` / ``DISCHARGE_COLUMNS``);
            * an array of stage values, with `discharge` given separately.
        discharge : array-like or str, optional
            Discharge values when `data` is an array, or the name of the
            discharge column when `data` is a DataFrame.
        stage : str, optional
            Name of the stage column, when `data` is a DataFrame and the column
            is not one of the recognized names.
        time : array-like or str, optional
            Measurement timestamps, or the name of the timestamp column.
        site_id, source, datum_note : str
            Provenance, carried through to reports and plots.
        stage_label : str, optional
            Description of the stage axis. Defaults to a generic label, or to the
            label the applied `stage_datum` produces.
        stage_datum : object, optional
            A datum conversion to apply to the stage values before fitting -
            anything :func:`limnotech_rating_curves.data.datum.to_gage_height`
            accepts: a number to subtract, ``"lowest"`` to reference the lowest
            observation, a timeseries of reference elevations, or a
            :class:`~limnotech_rating_curves.data.datum.StageDatum`.
        drop_nonpositive : bool, default True
            Drop rows whose discharge is zero or negative. Every rating model in
            this package works in log-discharge space, so a non-positive
            discharge cannot be fitted; dropping is the only alternative to
            failing.

        Returns
        -------
        Sample

        Examples
        --------
        >>> Sample.of(pd.DataFrame({"stage_ft": [1.2, 2.0], "discharge_cfs": [5, 22]}))
        Sample(n=2, site='')
        >>> Sample.of([1.2, 2.0], [5, 22])
        Sample(n=2, site='')
        """
        if isinstance(data, Sample):
            return data

        if isinstance(data, pd.DataFrame):
            frame = data
            stage_col = _pick_column(frame, STAGE_COLUMNS, "stage", stage)
            discharge_col = _pick_column(
                frame, DISCHARGE_COLUMNS, "discharge",
                discharge if isinstance(discharge, str) else None)
            stage_values = frame[stage_col]
            discharge_values = frame[discharge_col]
            time_values = None
            time_col = time if isinstance(time, str) else None
            if time_col is None:
                try:
                    time_col = _pick_column(frame, TIME_COLUMNS, "time")
                except KeyError:
                    time_col = None
            if time_col is not None:
                time_values = frame[time_col]
            elif time is not None and not isinstance(time, str):
                time_values = pd.Series(time, index=frame.index)
            if stage_label is None:
                stage_label = f"{stage_col} (ft)" if "ft" not in stage_col else stage_col
            # keep the source frame's provenance if it carries any
            attrs = getattr(frame, "attrs", {}) or {}
            site_id = site_id or str(attrs.get("site_id") or attrs.get("station")
                                     or attrs.get("gage") or "")
            source = source or str(attrs.get("source", ""))
            datum_note = datum_note or str(attrs.get("datum_note", ""))
        else:
            if discharge is None:
                raise ValueError(
                    "pass discharge as well when `data` is an array of stage "
                    "values - Sample.of(stage_ft, discharge_cfs)")
            stage_values = pd.Series(np.asarray(data, float).ravel())
            discharge_values = pd.Series(np.asarray(discharge, float).ravel())
            time_values = None if time is None else pd.Series(np.asarray(time).ravel())

        tidy = pd.DataFrame({
            "stage_ft": pd.to_numeric(stage_values, errors="coerce").to_numpy(float),
            "discharge_cfs": pd.to_numeric(discharge_values,
                                           errors="coerce").to_numpy(float),
        })
        if time_values is not None:
            tidy["time"] = pd.to_datetime(np.asarray(time_values), errors="coerce",
                                          format="mixed", utc=False)

        note = datum_note
        reference = 0.0
        if stage_datum is not None:
            from .data import datum as datum_module
            # discharge goes in too: the "lowest" datum places its reference relative
            # to the estimated stage of zero flow, which needs both columns
            converted = datum_module.to_gage_height(
                tidy["stage_ft"], stage_datum,
                time=tidy["time"] if "time" in tidy else None,
                discharge=tidy["discharge_cfs"])
            tidy["stage_ft"] = converted.stage_ft
            note = converted.note if not note else f"{note}; {converted.note}"
            stage_label = converted.label
            reference = (float(converted.reference_ft)
                         if np.ndim(converted.reference_ft) == 0 else float("nan"))

        before = len(tidy)
        tidy = tidy.dropna(subset=["stage_ft", "discharge_cfs"])
        if drop_nonpositive:
            tidy = tidy[tidy["discharge_cfs"] > 0]
        dropped = before - len(tidy)
        if dropped:
            log.info("dropped %d of %d measurement(s) with missing or "
                     "non-positive values", dropped, before)

        return cls(
            stage_ft=tidy["stage_ft"].to_numpy(float),
            discharge_cfs=tidy["discharge_cfs"].to_numpy(float),
            time=tidy["time"].to_numpy() if "time" in tidy else None,
            site_id=site_id, source=source,
            stage_label=stage_label or "stage (ft)", datum_note=note,
            datum_reference_ft=reference)

    # -- basics ---------------------------------------------------------------

    def __len__(self) -> int:
        """Number of measurements."""
        return int(self.stage_ft.size)

    def __repr__(self) -> str:
        return f"Sample(n={len(self)}, site={self.site_id!r})"

    def to_frame(self) -> pd.DataFrame:
        """The sample as a tidy DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns ``stage_ft``, ``discharge_cfs`` and, when known, ``time``.
            Provenance is on ``df.attrs``.
        """
        columns = {"stage_ft": self.stage_ft, "discharge_cfs": self.discharge_cfs}
        if self.time is not None:
            columns["time"] = self.time
        frame = pd.DataFrame(columns)
        frame.attrs.update(site_id=self.site_id, source=self.source,
                           stage_label=self.stage_label, datum_note=self.datum_note,
                           datum_reference_ft=self.datum_reference_ft)
        return frame

    @property
    def stage_range(self) -> tuple[float, float]:
        """``(lowest, highest)`` observed stage in feet, or ``(nan, nan)`` if empty."""
        if not len(self):
            return (float("nan"), float("nan"))
        return (float(np.nanmin(self.stage_ft)), float(np.nanmax(self.stage_ft)))

    @property
    def discharge_range(self) -> tuple[float, float]:
        """``(lowest, highest)`` observed discharge in cfs, or ``(nan, nan)``."""
        if not len(self):
            return (float("nan"), float("nan"))
        return (float(np.nanmin(self.discharge_cfs)),
                float(np.nanmax(self.discharge_cfs)))

    def support(self) -> dict:
        """How much of a rating these measurements can actually identify.

        A poor score has two very different causes and a reader has to be able to
        tell them apart: the model is wrong, or the measurements never
        constrained a rating in the first place. A handful of points spanning less
        than a decade of discharge is the second case, and no stage-discharge
        curve is identifiable from it however good the sampler is.

        Returns
        -------
        dict
            ``n``, ``distinct_discharge``, ``discharge_min``, ``discharge_max``,
            ``discharge_span_ratio`` (max / min), ``stage_range_ft`` and
            ``identifiable`` - the last being True when there are at least four
            distinct discharges spanning at least one decade, the rough threshold
            below which a power-law exponent is barely constrained.
        """
        if not len(self):
            return {"n": 0, "identifiable": False}
        q = pd.Series(self.discharge_cfs).dropna()
        h = pd.Series(self.stage_ft).dropna()
        span = float(q.max() / q.min()) if q.min() > 0 else None
        distinct = int(q.round(4).nunique())
        return {
            "n": int(len(q)),
            "distinct_discharge": distinct,
            "discharge_min": float(q.min()),
            "discharge_max": float(q.max()),
            "discharge_span_ratio": span,
            "stage_range_ft": float(h.max() - h.min()),
            "identifiable": bool(distinct >= 4 and (span or 0) >= 10),
        }

    def stage_grid(self, points=None, pad_fraction=None) -> np.ndarray:
        """A stage grid spanning the observations, padded a little on each side.

        Parameters
        ----------
        points : int, optional
            Number of grid points. Defaults to ``settings.GRID_POINTS``.
        pad_fraction : float, optional
            Fraction of the observed stage range to extend past each end.
            Defaults to ``settings.GRID_PAD_FRACTION``.

        Returns
        -------
        numpy.ndarray
            Ascending stage values (feet).
        """
        return padded_stage_grid(self.stage_ft, points=points,
                                 pad_fraction=pad_fraction)

    def time_range(self, pad_fraction=None) -> tuple:
        """The window these measurements span, padded a little on each side.

        Parameters
        ----------
        pad_fraction : float, optional
            Fraction of the measured span to extend past each end. Defaults to
            ``settings.GRID_PAD_FRACTION``, the same padding :meth:`stage_grid` uses
            on the stage axis. A sample whose measurements share one instant is
            padded by a day instead, since a fraction of nothing is nothing.

        Returns
        -------
        tuple
            ``(start, end)`` as pandas Timestamps, or ``(None, None)`` when the
            sample has no timestamps.
        """
        from . import settings
        if self.time is None or len(self.time) == 0:
            return None, None
        pad_fraction = (settings.GRID_PAD_FRACTION if pad_fraction is None
                        else float(pad_fraction))
        times = pd.to_datetime(pd.Series(self.time)).dropna()
        if times.empty:
            return None, None
        first, last = times.min(), times.max()
        pad = pad_fraction * (last - first) or pd.Timedelta(days=1)
        return first - pad, last + pad

    # -- fitting convenience --------------------------------------------------

    def fit(self, model="power_law", **kwargs):
        """Fit one rating model to this sample.

        A thin alias for :func:`limnotech_rating_curves.fit_rating`, so a loaded
        sample can be fitted without a second import.

        Parameters
        ----------
        model : str or RatingModel
            Which model to fit. See :func:`limnotech_rating_curves.fit_rating`.
        **kwargs
            Forwarded to :func:`limnotech_rating_curves.fit_rating`.

        Returns
        -------
        RatingModel
            The fitted estimator.
        """
        from . import ratings
        return ratings.fit_rating(self, model=model, **kwargs)

    def compare(self, models=None, **kwargs):
        """Fit several models to this sample and return them for comparison.

        Parameters
        ----------
        models : sequence, optional
            Which models to fit. See :func:`limnotech_rating_curves.compare`.
        **kwargs
            Forwarded to :func:`limnotech_rating_curves.compare`.

        Returns
        -------
        RatingSet
        """
        from . import ratings
        return ratings.compare(self, models=models, **kwargs)

    def cross_validate(self, **kwargs):
        """Cross-validate models on this sample.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`limnotech_rating_curves.cross_validate`.

        Returns
        -------
        CrossValidation
        """
        from .model_selection import crossval
        return crossval.cross_validate(self, **kwargs)

    def plot_record(self, station, *, variable: str = "stage", pad_fraction=None,
                    discharge=None, figsize=(12, 7)):
        """The continuous record these measurements were drawn from, with them on it.

        The window comes from :meth:`time_range`, so it is the measured span with a
        little either side rather than dates you have to look up. The measurements are
        drawn on the record's own datum, by adding back what was subtracted to make
        the sample.

        Parameters
        ----------
        station : fb_pagaia.core.Station
            The station to fetch the record from.
        variable : str, default 'stage'
            Which variable to fetch. See
            :func:`limnotech_rating_curves.data.pagaia.raw_station_series`.
        pad_fraction : float, optional
            How far past the measurements to fetch. See :meth:`time_range`.
        discharge : pandas.DataFrame, optional
            All the discharge measurements, for the lower panel. Defaults to this
            sample's own - which is the subset that was matched to a stage reading, so
            passing the full set is what shows any that were dropped.
        figsize : tuple, default (12, 7)
            Figure size.

        Returns
        -------
        numpy.ndarray of matplotlib.axes.Axes
            The two axes, record above and discharge below.
        """
        from .data import datum, pagaia, pagaia_corrections
        from .export import plots
        start, end = self.time_range(pad_fraction)
        # Fetch, correct the units, convert - see
        # limnotech_rating_curves.data.pagaia_corrections.
        raw = pagaia.raw_station_series(station, variable=variable, start=start,
                                        end=end)
        meters = pagaia_corrections.to_meters(raw, pagaia.station_name(station),
                                              variable=variable)
        series = datum.in_units(meters, "ft")
        return plots.plot_record(self, series, discharge=discharge, figsize=figsize)


@dataclass
class Fit:
    """One model fitted to one sample

    Attributes
    ----------
    key : str
        Stable model identifier, e.g. ``"power_law_2seg"`` or ``"bdrc_gplm0"``.
    label : str
        How the model reads in a legend or table.
    family : str
        Which backend fitted it: ``"ratingcurve"`` or ``"bdrc"``.
    n : int
        Measurements it was fitted on.
    status : {'ok', 'skipped', 'failed'}
        ``"skipped"`` means the sample was too small for this model;
        ``"failed"`` means the fit was attempted and raised.
    reason : str
        Why it was skipped or how it failed.
    config : dict
        The model configuration actually used (segments, knots, sampler).
    metrics : dict
        In-sample scores, from :func:`~limnotech_rating_curves.model_selection.metrics.fit_metrics`.
    curve : pandas.DataFrame or None
        The fitted rating: ``stage_ft``, ``discharge_cfs`` (posterior mean),
        ``discharge_median_cfs``, ``lower``, ``upper``.
    predicted : numpy.ndarray or None
        Predicted discharge at each observed stage.
    rating : object or None
        The live ``ratingcurve`` model object, for the ratingcurve family only.
        Its presence is what lets :meth:`RatingModel.predict` evaluate the model
        exactly rather than interpolating its curve.
    bayes : dict
        Bayesian comparison scores, filled in on first request (see
        :mod:`limnotech_rating_curves.model_selection.elpd`).
    log_likelihood : numpy.ndarray or None
        Pointwise log-likelihood, shape ``(draws, observations)``, in raw
        log-discharge space. Produced by the bdrc family at fit time; the
        ratingcurve family computes it on demand from `idata`.
    native : dict
        The backend's own information criteria, kept as a cross-check.
    idata : arviz.InferenceData or None
        The posterior. Present for both families, which is what makes the NetCDF
        export and the convergence diagnostics work identically for each.
    """

    key: str
    label: str
    family: str
    n: int
    status: str
    reason: str = ""
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    curve: pd.DataFrame = None
    predicted: np.ndarray = None
    rating: object = None
    bayes: dict = field(default_factory=dict)
    log_likelihood: np.ndarray = None
    native: dict = field(default_factory=dict)
    idata: object = None

    @property
    def ok(self) -> bool:
        """True when the model fitted successfully."""
        return self.status == "ok"


@dataclass
class FoldCurve:
    """One cross-validation fold's fitted curve, in one shape for every family.

    The median curve and its 95% band are evaluated over the fold's stage grid;
    the training and held-out points travel with it, along with the held-out
    scores, so the map's cross-validation pane and the per-fold figures read one
    structure regardless of which backend produced the fit.

    Attributes
    ----------
    model_key : str
        Which model this fold belongs to.
    split : int
        Fold index.
    stage_ft : list of float
        Stage grid the curve is evaluated on.
    discharge_median_cfs, lower_cfs, upper_cfs : list of float
        Posterior median and 2.5 / 97.5 percent bounds on that grid.
    train_stage, train_discharge : list of float
        The measurements this fold was fitted on.
    test_stage, test_discharge : list of float
        The measurements held out from it.
    test_metrics : dict
        Held-out scores, from :func:`~limnotech_rating_curves.model_selection.metrics.fit_metrics`.
    """

    model_key: str
    split: int
    stage_ft: list
    discharge_median_cfs: list
    lower_cfs: list
    upper_cfs: list
    train_stage: list
    train_discharge: list
    test_stage: list
    test_discharge: list
    test_metrics: dict = field(default_factory=dict)

    def predict_test(self) -> np.ndarray:
        """This fold's predicted discharge at its held-out stages.

        Recovered by interpolating the fold's own median curve, so it tracks the
        curve that gets drawn. Stages outside the curve's support come back NaN.

        Returns
        -------
        numpy.ndarray
        """
        grid = np.asarray(self.stage_ft, float)
        if grid.size < 2:
            return np.full(len(self.test_stage), np.nan)
        return np.interp(np.asarray(self.test_stage, float), grid,
                         np.asarray(self.discharge_median_cfs, float),
                         left=np.nan, right=np.nan)

    def to_map_dict(self) -> dict:
        """The fold as the plain dict the interactive map's payload uses.

        ``test_metrics`` travels with it: a fold's held-out scores are the whole point
        of showing the fold, and a map that draws the curve without them shows the
        reader a picture and withholds the number.
        """
        return {"fold": self.split, "stage_ft": self.stage_ft,
                "q_median": self.discharge_median_cfs,
                "q_lower": self.lower_cfs, "q_upper": self.upper_cfs,
                "train_stage": self.train_stage, "train_q": self.train_discharge,
                "test_stage": self.test_stage, "test_q": self.test_discharge,
                "n_train": len(self.train_stage), "n_test": len(self.test_stage),
                "test_metrics": {name: (None if value is None
                                        or (isinstance(value, float)
                                            and not np.isfinite(value))
                                        else value)
                                 for name, value in (self.test_metrics or {}).items()}}


@dataclass
class ExternalCurve:
    """A rating this package did not fit, attached to a site for comparison.

    Two very different things arrive at a site as "a curve somebody else drew", and
    typing them the same way is how a report ends up lying:

    ``kind="published_reference"``
        The **USGS published rating** for a gage - a curve maintained by a federal
        agency and the closest thing to ground truth a site can have.
    ``kind="spreadsheet"``
        The **literal equation typed into a MAGL field workbook**, with the
        coefficients its author entered. Not a published rating, not fitted here, and
        the whole point of the MAGL comparison.

    A site can carry both, and several of either, so they live in a list
    (:attr:`SiteRating.extra_curves`) rather than in one slot. The distinction is
    carried in `kind` and travels into the results table's ``family`` column and into
    the map's legend, so no reader has to infer it from a label.

    Attributes
    ----------
    key : str
        Stable identifier for the results table, e.g. ``"spreadsheet"`` or
        ``"published_reference"``.
    label : str
        How it reads in a legend or a table row.
    kind : str
        What sort of external curve this is - see above.
    curve : pandas.DataFrame
        ``stage_ft`` and ``discharge_cfs``, and ``lower`` / ``upper`` when the source
        supplies an interval. Most do not: a typed spreadsheet equation is a bare
        line and drawing a band around it would be an invention.
    metrics : dict
        How it scores on the site's own measurements, from :func:`~limnotech_rating_curves.model_selection.metrics.fit_metrics`, so it
        lands in the comparison table on the same footing as a fit.
    color : str
        Hex color it is drawn in. Empty means "use the style this ``kind`` is drawn
        with everywhere", which is the normal case - see
        ``settings.MAP_EXTERNAL_STYLE``.
    dash : str
        Plotly line style it is drawn with. Empty means the same.
    detail : dict
        Anything provenance-shaped worth showing: the workbook's declared ``form``,
        the ``axis`` the stored equation consumes, the ``equation`` text.
    """

    key: str
    label: str
    kind: str
    curve: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    color: str = ""
    dash: str = ""
    detail: dict = field(default_factory=dict)

    def __repr__(self):
        nse = self.metrics.get("nse", float("nan"))
        return f"ExternalCurve({self.key}, kind={self.kind!r}, nse={nse:.3f})"
