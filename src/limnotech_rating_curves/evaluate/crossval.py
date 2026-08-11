import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models import catalog
from .. import settings
from ..core import Sample, fit_metrics

log = logging.getLogger(__name__)

#: Scheme names :func:`cross_validate` accepts.
SCHEMES = ("auto", "holdout", "loo")


def scheme_for(n_points: int, scheme: str = "auto", *, holdout: float = None,
               n_splits: int = None) -> dict:
    """Resolve which cross-validation scheme to run and with what settings.

    Parameters
    ----------
    n_points : int
        Measurements in the sample.
    scheme : {'auto', 'holdout', 'loo'}, default 'auto'
        ``"auto"`` picks the holdout sweep at ``settings.CV_HOLDOUT_MIN_POINTS``
        measurements or more and leave-one-out below.
    holdout : float, optional
        Fraction held out per fold, for the holdout scheme. Defaults to
        ``settings.CV_HOLDOUT`` (0.90).
    n_splits : int, optional
        Number of folds. Defaults to ``settings.CV_SPLITS`` (12) for the holdout
        scheme, and to ``n_points`` for leave-one-out, where the fold count is not
        a choice.

    Returns
    -------
    dict
        ``name``, ``holdout``, ``n_splits``, ``n_train`` and a human-readable
        ``description``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if scheme == "auto":
        scheme = "holdout" if n_points >= settings.CV_HOLDOUT_MIN_POINTS else "loo"

    if scheme == "holdout":
        holdout = settings.CV_HOLDOUT if holdout is None else float(holdout)
        n_splits = settings.CV_SPLITS if n_splits is None else int(n_splits)
        n_train = max(settings.CV_MIN_TRAIN, int(round((1 - holdout) * n_points)))
        return {"name": "holdout", "holdout": holdout, "n_splits": n_splits,
                "n_train": n_train,
                "description": (f"{holdout:.0%} holdout, {n_splits} random "
                                f"stage-stratified splits training on {n_train} of "
                                f"{n_points} measurements")}

    return {"name": "loo", "holdout": None,
            "n_splits": int(n_points if n_splits is None else n_splits),
            "n_train": max(1, n_points - 1),
            "description": (f"leave-one-out, {n_points} folds training on "
                            f"{n_points - 1} and predicting 1")}


def _as_frame(sample) -> pd.DataFrame:
    """A clean stage/discharge frame from a Sample or frame-like."""
    frame = sample.to_frame() if isinstance(sample, Sample) else pd.DataFrame(sample)
    return (frame.dropna(subset=["stage_ft", "discharge_cfs"])
            .reset_index(drop=True))


def stratified_train_index(stage, n_train: int, rng) -> np.ndarray:
    """Indices of a training set that spans the stage range.

    Sort by stage, cut the order into `n_train` roughly equal bins, and take one
    random index from each. This is what keeps a small random training draw from
    collapsing onto one end of the rating, which would make the fold's score a
    measure of the draw rather than of the model.

    Parameters
    ----------
    stage : array-like
        Stage values (feet).
    n_train : int
        Training-set size.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    numpy.ndarray
        Training indices into the original ordering.
    """
    order = np.argsort(np.asarray(stage, float))
    bins = np.array_split(order, n_train)
    return np.array([rng.choice(chunk) for chunk in bins if len(chunk) > 0])


def holdout_splits(frame, holdout: float = None, n_splits: int = None,
                   seed: int = settings.SEED, n_train: int = None) -> list:
    """Stage-stratified train/test partitions for the holdout scheme.

    Parameters
    ----------
    frame : pandas.DataFrame
        The measurements.
    holdout : float, optional
        Fraction held out per fold. Defaults to ``settings.CV_HOLDOUT``.
    n_splits : int, optional
        Number of folds. Defaults to ``settings.CV_SPLITS``.
    seed : int
        Base seed; fold *k* uses ``seed + k``, so the partition is reproducible.
    n_train : int, optional
        Explicit training-set size, overriding `holdout`.

    Returns
    -------
    list of tuple
        ``(split index, train indices, test indices)``. The split index is the
        original position, preserved even when a degenerate split with no test
        points is dropped. Empty when the sample is too small to hold anything out.
    """
    stage = frame["stage_ft"].to_numpy(float)
    n = len(frame)
    holdout = settings.CV_HOLDOUT if holdout is None else float(holdout)
    n_splits = settings.CV_SPLITS if n_splits is None else int(n_splits)
    if n_train is None:
        n_train = max(settings.CV_MIN_TRAIN, int(round((1 - holdout) * n)))
    n_train = max(1, int(n_train))
    if n_train >= n or n - n_train < 1:
        return []

    splits = []
    for split in range(n_splits):
        rng = np.random.default_rng(seed + split)
        train = np.unique(stratified_train_index(stage, n_train, rng))
        held_out = np.setdiff1d(np.arange(n), train)
        if held_out.size == 0:
            continue
        splits.append((split, train, held_out))
    return splits


def leave_one_out_splits(frame) -> list:
    """The exhaustive leave-one-out partition: fold *i* holds out measurement *i*.

    Parameters
    ----------
    frame : pandas.DataFrame
        The measurements.

    Returns
    -------
    list of tuple
        ``(split index, train indices, test indices)``, one fold per measurement.
        Empty when there are too few measurements to train on after removing one.
    """
    n = len(frame)
    if n - 1 < settings.CV_MIN_TRAIN:
        return []
    everything = np.arange(n)
    return [(i, np.delete(everything, i), np.array([i])) for i in range(n)]


@dataclass
class CrossValidation:
    """The result of a cross-validation sweep.

    Attributes
    ----------
    scheme : dict
        Which scheme ran, from :func:`scheme_for`.
    folds : pandas.DataFrame
        One row per (model, fold): ``model``, ``split``, ``n_train``, ``n_test``,
        ``status``, and the held-out scores prefixed ``test_``.
    curves : list of FoldCurve
        Every fold's fitted curve, for plotting.
    pooled : pandas.DataFrame
        Per model, the scores over *all* held-out predictions pooled into one
        vector. **This is the headline under leave-one-out**, where per-fold NSE is
        undefined.
    per_fold : pandas.DataFrame
        Per model, the mean and standard deviation of the per-fold scores. The
        headline under the holdout scheme; the standard deviation is the number that
        says how much a model swings between training draws.
    support : dict
        What the sample could identify at all, from :meth:`Sample.support`.
    failures : dict
        ``{model key: reason}`` for models that raised outright, so a model that
        failed every fold does not simply vanish.
    seed : int
        The base seed used.
    fold_method : str
        How folds were fitted - ``"advi"`` by default. Folds carry no ELPD.
    """

    scheme: dict
    folds: pd.DataFrame
    curves: list = field(default_factory=list)
    pooled: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_fold: pd.DataFrame = field(default_factory=pd.DataFrame)
    support: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)
    seed: int = settings.SEED
    fold_method: str = "advi"

    @property
    def headline(self) -> pd.DataFrame:
        """The table to read for this scheme - pooled for LOO, per-fold otherwise."""
        return self.pooled if self.scheme.get("name") == "loo" else self.per_fold

    def curves_by_model(self) -> dict:
        """The fold curves grouped as ``{model key: {fold index: FoldCurve}}``."""
        grouped: dict = {}
        for curve in self.curves:
            grouped.setdefault(curve.model_key, {})[int(curve.split)] = curve
        return grouped

    def to_map_payload(self) -> dict:
        """The fold data *and its summary statistics* in the map pane's shape.

        The curves alone are a picture of how much the rating moves between refits;
        the summary tables are the numbers that say whether that movement matters.
        Both go across, along with which scheme ran and how many folds it had, because
        "mean test NSE over 12 folds of a 90% holdout" and "pooled NSE over 8
        leave-one-out folds" are not the same claim and a reader has to be told which
        one is on screen.

        Returns
        -------
        dict
            ``models`` - ``{model key: {"label", "color", "dash", "folds": [...],
            "pooled": {...}, "per_fold": {...}}}``; ``scheme``; ``headline`` - the
            name of the table to read for this scheme (``"pooled"`` or
            ``"per_fold"``); ``support`` - what the sample could identify at all.
        """
        def records(table) -> dict:
            if table is None or table.empty:
                return {}
            clean = table.replace({np.nan: None})
            return {str(key): value for key, value in clean.to_dict(orient="index").items()}

        pooled = records(self.pooled)
        per_fold = records(self.per_fold)

        models: dict = {}
        for curve in self.curves:
            entry = models.setdefault(curve.model_key, {
                "label": catalog.label(curve.model_key),
                "color": catalog.color(curve.model_key),
                "dash": catalog.dash(curve.model_key), "folds": [],
                "pooled": pooled.get(curve.model_key, {}),
                "per_fold": per_fold.get(curve.model_key, {})})
            entry["folds"].append(curve.to_map_dict())
        return {"models": models, "scheme": dict(self.scheme),
                "headline": "pooled" if self.scheme.get("name") == "loo"
                            else "per_fold",
                "fold_method": self.fold_method,
                "support": {key: (None if isinstance(value, float)
                                  and not np.isfinite(value) else value)
                            for key, value in (self.support or {}).items()}}

    def plot(self, model=None, ax=None):
        """Overlay every fold's curve, so the spread between refits is visible.

        Parameters
        ----------
        model : str, optional
            Restrict to one model. All models are drawn if omitted, each in its own
            color, which shows which family is stable and which is not.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8.5, 6))
        drawn = set()
        for curve in self.curves:
            if model is not None and curve.model_key != model:
                continue
            color = catalog.color(curve.model_key)
            label = (None if curve.model_key in drawn
                     else catalog.label(curve.model_key))
            drawn.add(curve.model_key)
            ax.plot(curve.stage_ft, curve.discharge_median_cfs, color=color, lw=1.4,
                    alpha=0.65, label=label)
        if self.curves:
            first = self.curves[0]
            all_stage = list(first.train_stage) + list(first.test_stage)
            all_discharge = list(first.train_discharge) + list(first.test_discharge)
            ax.scatter(all_stage, all_discharge, s=45, facecolor="white",
                       edgecolor="black", zorder=5, label="measurements")
        ax.set_yscale("log")
        ax.set_xlabel("stage (ft)")
        ax.set_ylabel("discharge (cfs)")
        ax.set_title(f"cross-validation folds - {self.scheme.get('description', '')}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize="small")
        return ax

    def __repr__(self):
        return (f"CrossValidation({self.scheme.get('name')}, "
                f"{len(self.curves)} fold fits, "
                f"{self.folds['model'].nunique() if len(self.folds) else 0} models)")


def cross_validate(sample, models=None, *, discharge=None, stage=None,
                   stage_datum=None, scheme: str = "auto", holdout=None,
                   n_splits=None, n_train=None, seed: int = settings.SEED,
                   fold_method: str = "advi", zero_flow=None,
                   nuts_sampler=None) -> CrossValidation:
    """Refit each model on subsets of the measurements and score the held-out ones.

    Every model is scored on the *same* folds. That is structural, not hoped for:
    the partition depends only on the measurements, the seed and the split settings
    - never on the model set - so it is computed once and the model loop nests
    inside it.

    Parameters
    ----------
    sample : Sample or pandas.DataFrame or array-like
        The measurements.
    models : None or str or sequence, optional
        Which models to cross-validate. ``None`` uses the curated default set.
    discharge : array-like or str, optional
        Discharge (cfs) when `sample` is an array of stage values, or the discharge
        column name when it is a DataFrame.
    stage : str, optional
        Name of the stage column, when `sample` is a DataFrame whose stage column is
        not one of the recognized names.
    stage_datum : optional
        Convert stage onto a gage-height reference first; see
        :mod:`limnotech_rating_curves.data.datum`.
    scheme : {'auto', 'holdout', 'loo'}, default 'auto'
        See ``docs/evaluate/crossval.md``. ``"auto"`` picks by sample size.
    holdout : float, optional
        Fraction held out per fold, for the holdout scheme.
    n_splits : int, optional
        Number of folds.
    n_train : int, optional
        Explicit training-set size, overriding `holdout`.
    seed : int
        Base seed. Fold *k* is fitted with ``seed + k``.
    fold_method : {'advi', 'nuts'}, default 'advi'
        How to fit each fold. ADVI because a sweep is many refits; note that folds
        therefore carry no ELPD.
    zero_flow : optional
        Stage-of-zero-flow handling, forwarded to each fit. See
        :meth:`limnotech_rating_curves.ratings.RatingModel.fit`.
    nuts_sampler : str, optional
        Which NUTS implementation, for the models whose folds are NUTS chains
        (bdrc's are, even when `fold_method` is ADVI for the others).

    Returns
    -------
    CrossValidation

    Examples
    --------
    >>> result = cross_validate(gage_measurements, models="all")   # doctest: +SKIP
    >>> result.headline                                             # doctest: +SKIP
    >>> result.plot()                                               # doctest: +SKIP
    """
    sample = Sample.of(sample, discharge, stage=stage, stage_datum=stage_datum)
    frame = _as_frame(sample)
    n = len(frame)
    resolved = scheme_for(n, scheme, holdout=holdout, n_splits=n_splits)

    if n < settings.CV_MIN_POINTS:
        log.warning("%d measurement(s) is too few to hold any out (need %d)",
                    n, settings.CV_MIN_POINTS)
        return CrossValidation(scheme=resolved, folds=pd.DataFrame(),
                               support=sample.support(), seed=seed,
                               fold_method=fold_method)

    if resolved["name"] == "loo":
        splits = leave_one_out_splits(frame)
    else:
        splits = holdout_splits(frame, holdout=resolved["holdout"],
                                n_splits=resolved["n_splits"], seed=seed,
                                n_train=n_train or resolved["n_train"])
    if not splits:
        log.warning("no usable folds for %d measurement(s) under the %s scheme",
                    n, resolved["name"])
        return CrossValidation(scheme=resolved, folds=pd.DataFrame(),
                               support=sample.support(), seed=seed,
                               fold_method=fold_method)

    grid = sample.stage_grid()
    entries = catalog.select(models)
    log.info("cross-validating %d model(s) over %d fold(s): %s", len(entries),
             len(splits), resolved["description"])

    rows, curves, failures = [], [], {}
    for split, train_index, test_index in splits:
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].reset_index(drop=True)
        for entry in entries:
            # one model failing must not sink the sweep: the spline in particular
            # dies on a single-row test set, because ratingcurve hands PyMC a 1-D
            # basis matrix where a 2-D one is required
            try:
                curve = entry.fit_fold(train, test, grid, split, seed=seed + split,
                                       zero_flow=zero_flow,
                                       nuts_sampler=nuts_sampler,
                                       method=fold_method if entry.family
                                       == "ratingcurve" else "nuts")
            except Exception as exc:  # noqa: BLE001
                failures.setdefault(entry.key, f"{type(exc).__name__}: {exc}"[:200])
                curve = None
            row = {"model": entry.key, "split": split, "n_train": len(train),
                   "n_test": len(test), "status": "ok" if curve else "failed",
                   "reason": "" if curve else failures.get(entry.key, "")}
            if curve is not None:
                row.update({f"test_{name}": value
                            for name, value in curve.test_metrics.items()})
                curves.append(curve)
            rows.append(row)

    folds = pd.DataFrame(rows)
    for key, reason in failures.items():
        log.warning("%s raised on at least one fold: %s", key, reason)

    # A model that failed every fold produces no curve and so no row in the summary
    # tables. Say so, or it simply vanishes from the comparison - which on a small
    # sample is the normal outcome for the more complex models, since a fold trains
    # on fewer points than the model needs.
    fitted = {curve.model_key for curve in curves}
    for entry in entries:
        if entry.key in fitted:
            continue
        attempted = folds[folds["model"] == entry.key] if len(folds) else folds
        log.warning("%s produced no fold curves (%d fold(s) attempted, all failed) - "
                    "a fold trains on %d point(s) and this model needs %d",
                    entry.key, len(attempted), splits[0][1].size, entry.min_points)

    return CrossValidation(
        scheme=resolved, folds=folds, curves=curves,
        pooled=pooled_metrics(curves), per_fold=per_fold_metrics(folds),
        support=sample.support(), failures=failures, seed=seed,
        fold_method=fold_method)


def pooled_metrics(curves) -> pd.DataFrame:
    """Scores over all held-out predictions pooled into one vector, per model.

    This is the number to quote for leave-one-out, where each fold contributes one
    held-out prediction and a per-fold NSE is undefined.

    Parameters
    ----------
    curves : sequence of FoldCurve
        Every fold's fitted curve.

    Returns
    -------
    pandas.DataFrame
        Indexed by model, with ``n_predictions``, ``n_dropped`` (predictions that
        fell outside the fold curve's support), ``rmse``, ``nse``, ``pbias_pct`` and
        ``r2_log``.
    """
    grouped: dict = {}
    for curve in curves:
        observed, predicted = grouped.setdefault(curve.model_key, ([], []))
        observed.extend(np.asarray(curve.test_discharge, float).tolist())
        predicted.extend(curve.predict_test().tolist())

    rows = {}
    for model, (observed, predicted) in grouped.items():
        observed = np.asarray(observed, float)
        predicted = np.asarray(predicted, float)
        usable = np.isfinite(observed) & np.isfinite(predicted)
        scores = fit_metrics(observed[usable], predicted[usable])
        rows[model] = {"n_predictions": int(usable.sum()),
                       "n_dropped": int((~usable).sum()),
                       "rmse": scores["rmse"], "nse": scores["nse"],
                       "pbias_pct": scores["pbias_pct"], "r2_log": scores["r2_log"]}
    table = pd.DataFrame.from_dict(rows, orient="index")
    if not table.empty:
        table.index.name = "model"
        table = table.sort_values("r2_log", ascending=False)
    return table


def per_fold_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of the per-fold held-out scores, per model.

    The headline under the holdout scheme, where each fold holds out many points.
    The standard deviation is the interesting column: it is how much a model's
    predictions swing from one training draw to the next.

    Parameters
    ----------
    folds : pandas.DataFrame
        The per-(model, fold) table from :func:`cross_validate`.

    Returns
    -------
    pandas.DataFrame
        Indexed by model, with ``n_folds_ok``, ``n_train``, ``n_test`` and
        ``mean``/``std`` for ``nse``, ``rmse`` and ``r2_log``. Empty when nothing
        succeeded.
    """
    if folds.empty:
        return pd.DataFrame()
    succeeded = folds[folds["status"] == "ok"]
    if succeeded.empty:
        return pd.DataFrame()
    grouped = succeeded.groupby("model")
    table = pd.DataFrame({
        "n_folds_ok": grouped.size(),
        "n_train": grouped["n_train"].median(),
        "n_test": grouped["n_test"].median(),
    })
    for name in ("nse", "rmse", "r2_log"):
        column = f"test_{name}"
        if column in succeeded.columns:
            table[f"{name}_mean"] = grouped[column].mean()
            table[f"{name}_std"] = grouped[column].std()
    table.index.name = "model"
    sort_column = "r2_log_mean" if "r2_log_mean" in table else "n_folds_ok"
    return table.sort_values(sort_column, ascending=False)


def high_flow_skill(result: CrossValidation, quantile: float = 0.75) -> pd.DataFrame:
    """How well each model predicted the *highest* held-out measurements.

    Extrapolation to high flow is where a rating earns or loses its keeping: the
    high end is where the measurements are sparsest, where the consequences are
    largest, and where a flexible model that fits beautifully in the middle can be
    badly wrong. An overall score hides that, because the low-flow measurements
    outnumber the high-flow ones.

    So this scores only the held-out predictions above a stage quantile, and reports
    the ratio of predicted to observed discharge - which is the interpretable form
    of high-flow error, in a way that RMSE in cfs is not.

    Parameters
    ----------
    result : CrossValidation
        A completed sweep.
    quantile : float, default 0.75
        Stage quantile above which a held-out measurement counts as high flow.

    Returns
    -------
    pandas.DataFrame
        Indexed by model, with ``n_high``, ``median_ratio`` (predicted / observed;
        1.0 is unbiased, below 1 under-predicts high flow), ``worst_ratio`` (the
        ratio furthest from 1), ``r2_log_high`` and ``rmse_high``.
    """
    if not result.curves:
        return pd.DataFrame()
    all_stage = np.concatenate([np.asarray(curve.test_stage, float)
                               for curve in result.curves])
    if all_stage.size == 0:
        return pd.DataFrame()
    threshold = float(np.nanquantile(all_stage, quantile))

    rows = {}
    for curve in result.curves:
        stage = np.asarray(curve.test_stage, float)
        observed = np.asarray(curve.test_discharge, float)
        predicted = curve.predict_test()
        high = stage >= threshold
        bucket = rows.setdefault(curve.model_key, {"observed": [], "predicted": []})
        bucket["observed"].extend(observed[high].tolist())
        bucket["predicted"].extend(predicted[high].tolist())

    summary = {}
    for model, bucket in rows.items():
        observed = np.asarray(bucket["observed"], float)
        predicted = np.asarray(bucket["predicted"], float)
        usable = np.isfinite(observed) & np.isfinite(predicted) & (observed > 0)
        if usable.sum() == 0:
            summary[model] = {"n_high": 0}
            continue
        ratio = predicted[usable] / observed[usable]
        scores = fit_metrics(observed[usable], predicted[usable])
        worst = ratio[np.argmax(np.abs(np.log(np.where(ratio > 0, ratio, np.nan))))]
        summary[model] = {"n_high": int(usable.sum()),
                          "median_ratio": float(np.nanmedian(ratio)),
                          "worst_ratio": float(worst),
                          "r2_log_high": scores["r2_log"],
                          "rmse_high": scores["rmse"]}
    table = pd.DataFrame.from_dict(summary, orient="index")
    table.index.name = "model"
    table.attrs["stage_threshold_ft"] = threshold
    table.attrs["quantile"] = quantile
    return table
