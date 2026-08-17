"""
Rating-curve fitting engine: stage <-> discharge, from measured USGS field data.

No plotting, no web framework, no PyMC/Bayesian dependency - only numpy/scipy/pandas.
This is the piece that runs on every dashboard fetch/upload, so it has to be fast and
dependency-light.

Model family
------------
A rating curve relates stage (gage height, ft) to discharge (cfs). USGS gage
records typically span multiple hydraulic controls (e.g. a natural channel
control at low flow, an overbank control at high flow), so a single power law
does not always fit the full range. This module fits a *compounding segmented
power law* with 0, 1, or 2 breakpoints:

    Q(h) = C * (h - h0)^b0 * (1 + max(h - bp1, 0))^b1 * (1 + max(h - bp2, 0))^b2

- ``h0`` is the stage of zero flow.
- Each factor is exactly 1 below its breakpoint, so the curve is continuous
  (no jump) at every breakpoint by construction - only the curvature changes.
- With 0 breakpoints this is the standard single power law Q = C(h-h0)^b0.

Fitting is done in log space (log Q vs log(h-h0) and log1p(h-bp)), which is
standard practice for rating curves because discharge typically spans one or
more orders of magnitude and a plain least-squares fit on raw cfs would be
dominated entirely by the highest flows. Given the breakpoints and h0, the
remaining coefficients are a linear regression (closed form); h0 and the
breakpoints themselves are found with a bounded nonlinear search.

Model selection (how many breakpoints) is decided empirically per dataset by
cross-validation, not assumed - see ``select_best_model``.
"""

from __future__ import annotations

import os

# Must be set before numpy/scipy load their BLAS backend. On some Windows +
# conda-forge MKL builds, threaded LAPACK calls (lstsq/svd) from a second
# library sharing the process can crash the whole interpreter instead of
# raising a Python exception. Forcing MKL to run single-threaded avoids the
# conflict and costs nothing on problems this small (a handful of columns,
# at most a few hundred rows).
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass, field
import json

import numpy as np
from scipy.optimize import minimize

MIN_OFFSET = 1e-6  # smallest allowed (h - h0), keeps log() finite


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #

@dataclass
class FittedRating:
    """A fitted stage-discharge rating curve, ready to predict or serialize."""

    n_segments: int          # 1, 2, or 3
    zero_flow: float         # h0, stage of zero flow (ft)
    breakpoints: list        # [] , [bp1], or [bp1, bp2] (ft)
    log_coef: list           # [log(C), b0, b1, b2, ...] matching design matrix
    stage_range: tuple       # (min, max) stage in the fitting data (ft)
    discharge_range: tuple   # (min, max) discharge in the fitting data (cfs)
    n_obs: int
    metrics: dict = field(default_factory=dict)  # filled in after fitting/CV

    # -- prediction ---------------------------------------------------------

    def predict(self, stage):
        """Discharge (cfs) for one stage or an array of stages (ft)."""
        stage_arr = np.asarray(stage, dtype=float)
        scalar_input = stage_arr.ndim == 0
        X = _design_matrix(stage_arr, self.zero_flow, self.breakpoints)
        log_q = X @ np.asarray(self.log_coef)
        q = np.exp(log_q)
        return float(q[0]) if scalar_input else q

    def predict_stage(self, discharge, stage_grid=None):
        """
        Inverse lookup: stage (ft) for one discharge or an array of discharges (cfs),
        via interpolation on a fine predicted curve. Requires the curve to be
        monotonically increasing over ``stage_grid`` (checked; raises if not).
        """
        discharge = np.asarray(discharge, dtype=float)
        if stage_grid is None:
            stage_grid = np.linspace(
                self.zero_flow + MIN_OFFSET, self.stage_range[1] * 1.5, 5000
            )
        q_grid = self.predict(stage_grid)
        if np.any(np.diff(q_grid) <= 0):
            raise ValueError(
                "Rating curve is not strictly increasing over the requested stage "
                "range, so discharge cannot be uniquely inverted to a stage there. "
                "Narrow stage_grid to a monotonic portion of the curve."
            )
        stage = np.interp(discharge, q_grid, stage_grid)
        return float(stage) if stage.ndim == 0 else stage

    def is_extrapolation(self, stage):
        """True where a stage falls outside the range of the fitting measurements."""
        stage = np.asarray(stage, dtype=float)
        return (stage < self.stage_range[0]) | (stage > self.stage_range[1])

    def equation(self) -> str:
        c = np.exp(self.log_coef[0])
        terms = [f"{c:.5g} * (h - {self.zero_flow:.4g})^{self.log_coef[1]:.4g}"]
        for bp, b in zip(self.breakpoints, self.log_coef[2:]):
            terms.append(f"(1 + max(h - {bp:.4g}, 0))^{b:.4g}")
        return "Q = " + " * ".join(terms)

    def curve_table(self, stage_min=None, stage_max=None, step=0.01):
        """A stage -> discharge table over the fitted (or given) stage range."""
        import pandas as pd

        lo = self.zero_flow + MIN_OFFSET if stage_min is None else stage_min
        hi = self.stage_range[1] if stage_max is None else stage_max
        grid = np.round(np.arange(lo, hi + step, step), 6)
        q = self.predict(grid)
        return pd.DataFrame(
            {
                "stage_ft": grid,
                "discharge_cfs": q,
                "extrapolated": self.is_extrapolation(grid),
            }
        )

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_segments": self.n_segments,
            "zero_flow": self.zero_flow,
            "breakpoints": list(self.breakpoints),
            "log_coef": list(self.log_coef),
            "stage_range": list(self.stage_range),
            "discharge_range": list(self.discharge_range),
            "n_obs": self.n_obs,
            "metrics": self.metrics,
            "equation": self.equation(),
        }

    def to_json(self, path=None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_dict(cls, d: dict) -> "FittedRating":
        return cls(
            n_segments=d["n_segments"],
            zero_flow=d["zero_flow"],
            breakpoints=list(d["breakpoints"]),
            log_coef=list(d["log_coef"]),
            stage_range=tuple(d["stage_range"]),
            discharge_range=tuple(d["discharge_range"]),
            n_obs=d["n_obs"],
            metrics=d.get("metrics", {}),
        )

    @classmethod
    def from_json(cls, path) -> "FittedRating":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# Fitting internals
# --------------------------------------------------------------------------- #

def _design_matrix(stage, h0, breakpoints):
    base = np.log(np.clip(stage - h0, MIN_OFFSET, None))
    cols = [np.ones_like(base), base]
    for bp in breakpoints:
        cols.append(np.log1p(np.clip(stage - bp, 0, None)))
    return np.column_stack(cols)


def _fit_linear_given_knots(stage, discharge, h0, breakpoints):
    X = _design_matrix(stage, h0, breakpoints)
    y = np.log(discharge)
    # A near-singular X (e.g. h0 far enough below the data that every (stage - h0)
    # column is nearly constant) can make the underlying LAPACK routine abort the
    # whole process on Windows instead of raising - guard on the condition number
    # rather than relying on lstsq to fail softly.
    if not np.all(np.isfinite(X)) or np.linalg.cond(X) > 1e10:
        return None, np.full_like(y, 1e6)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, resid


def _knot_objective(knots, stage, discharge, n_breaks, stage_min, stage_span):
    h0 = knots[0]
    breakpoints = sorted(knots[1 : 1 + n_breaks])
    if h0 >= stage_min - MIN_OFFSET:
        return 1e12
    if n_breaks and (breakpoints[0] <= h0):
        return 1e12
    # Breakpoints closer together than 2% of the stage span are numerically
    # indistinguishable and produce degenerate, huge-canceling-exponent fits.
    if n_breaks == 2 and (breakpoints[1] - breakpoints[0] < 0.02 * stage_span):
        return 1e12
    coef, resid = _fit_linear_given_knots(stage, discharge, h0, breakpoints)
    if coef is not None and np.max(np.abs(coef)) > 1e3:
        return 1e12
    return float(np.sum(resid**2))


def _knot_bounds(stage_min, stage_max, n_breaks):
    span = stage_max - stage_min
    bounds = [(stage_min - 5 * span - 5, stage_min - MIN_OFFSET)]  # h0
    bounds += [(stage_min, stage_max)] * n_breaks  # breakpoints
    return bounds


def fit_rating(stage, discharge, n_segments: int = 1) -> FittedRating:
    """
    Fit a segmented power-law rating curve with ``n_segments`` in {1, 2, 3}
    (i.e. 0, 1, or 2 breakpoints) to measured stage/discharge pairs.
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    if stage.shape != discharge.shape or stage.ndim != 1:
        raise ValueError("stage and discharge must be 1-D arrays of the same length")
    if np.any(discharge <= 0):
        raise ValueError("discharge must be positive (zero/negative flows can't be log-fit)")

    n_breaks = n_segments - 1
    stage_min, stage_max = float(stage.min()), float(stage.max())
    span = stage_max - stage_min

    # Nelder-Mead can converge to a degenerate (ill-conditioned) local optimum from
    # one starting guess but a well-conditioned one from another, so try a handful
    # of different starting zero-flow offsets and keep the best-fitting result that
    # is actually well-conditioned, rather than giving up after a single attempt.
    best = None
    for h0_frac in (0.1, 0.02, 0.3, 0.6, 1.0):
        h0_init = stage_min - h0_frac * span - 0.1
        if n_breaks == 0:
            x0 = [h0_init]
        elif n_breaks == 1:
            x0 = [h0_init, stage_min + 0.5 * span]
        else:
            x0 = [h0_init, stage_min + 0.33 * span, stage_min + 0.67 * span]

        result = minimize(
            _knot_objective,
            x0=x0,
            args=(stage, discharge, n_breaks, stage_min, span),
            method="Nelder-Mead",
            bounds=_knot_bounds(stage_min, stage_max, n_breaks),
            options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000},
        )
        h0 = result.x[0]
        breakpoints = sorted(result.x[1:].tolist())
        coef, resid = _fit_linear_given_knots(stage, discharge, h0, breakpoints)
        if coef is None:
            continue
        ssr = float(np.sum(resid**2))
        if best is None or ssr < best[0]:
            best = (ssr, h0, breakpoints, coef)

    if best is None:
        raise RuntimeError(
            f"Could not find a well-conditioned {n_segments}-segment fit for this data "
            "(the optimizer only found degenerate breakpoints/zero-flow stage)."
        )
    _, h0, breakpoints, coef = best

    fitted = FittedRating(
        n_segments=n_segments,
        zero_flow=float(h0),
        breakpoints=[float(b) for b in breakpoints],
        log_coef=[float(c) for c in coef],
        stage_range=(stage_min, stage_max),
        discharge_range=(float(discharge.min()), float(discharge.max())),
        n_obs=len(stage),
    )
    fitted.metrics = _fit_metrics(discharge, fitted.predict(stage))
    return fitted


def residual_trend(model: "FittedRating", time, stage, discharge):
    """Percent residual (observed vs. predicted) over time - reveals rating drift.

    A model fit to the whole record can look inaccurate not because the curve
    shape is wrong, but because the physical rating itself changed partway
    through the record (channel or sediment change, vegetation, a USGS
    re-rating) - fitting one static curve across data governed by genuinely
    different physical relationships caps accuracy no matter the curve family.
    A trend or step change here, rather than random scatter centered on zero,
    is that signal - narrowing the fitted date range to the recent, stable
    era is usually a bigger accuracy win than trying a different curve shape.

    Parameters
    ----------
    model : FittedRating
        A fitted rating curve.
    time : array-like
        Timestamp for each measurement, same length and order as ``stage``.
    stage, discharge : array-like
        The measurements the model was fit to (or any measurements to check
        it against).

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``stage_ft``, ``discharge_cfs``, ``predicted_cfs``,
        ``percent_diff``, sorted by time.
    """
    import pandas as pd

    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    predicted = model.predict(stage)
    return pd.DataFrame({
        "time": pd.to_datetime(pd.Series(time).to_numpy()),
        "stage_ft": stage,
        "discharge_cfs": discharge,
        "predicted_cfs": predicted,
        "percent_diff": 100 * (discharge - predicted) / predicted,
    }).sort_values("time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _fit_metrics(observed, predicted) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    resid = observed - predicted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    log_obs, log_pred = np.log(observed), np.log(predicted)
    log_resid = log_obs - log_pred
    ss_res_log = float(np.sum(log_resid**2))
    ss_tot_log = float(np.sum((log_obs - log_obs.mean()) ** 2))
    r2_log = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else float("nan")

    return {
        "n": len(observed),
        "r2": r2,
        "r2_log": r2_log,
        "rmse_cfs": float(np.sqrt(np.mean(resid**2))),
        "rmse_log": float(np.sqrt(np.mean(log_resid**2))),
        "mae_cfs": float(np.mean(np.abs(resid))),
        "pbias_pct": float(100 * np.sum(resid) / np.sum(observed)),
        "nse": r2,  # NSE and R^2 coincide for this residual definition
    }


# --------------------------------------------------------------------------- #
# Cross-validation and model selection
# --------------------------------------------------------------------------- #

def cross_validate(stage, discharge, n_segments: int, n_splits: int = 10,
                    n_repeats: int = 5, seed: int = 42) -> dict:
    """
    Repeated k-fold cross-validation using only the measured field data:
    for each fold, refit on the training rows and predict the held-out rows,
    then pool every held-out prediction across all repeats/folds to score
    against the actual measured discharge. Falls back to leave-one-out when
    there are too few measurements for the requested number of folds.
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    n = len(stage)
    n_splits = min(n_splits, n) if n < n_splits else n_splits
    if n < 2 * n_splits:
        n_splits = max(2, n // 2)
    rng = np.random.default_rng(seed)

    pooled_obs, pooled_pred = [], []
    for rep in range(n_repeats):
        order = rng.permutation(n)
        folds = np.array_split(order, n_splits)
        for k, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(order, test_idx)
            try:
                model = fit_rating(stage[train_idx], discharge[train_idx], n_segments)
                pred = model.predict(stage[test_idx])
            except Exception:
                continue
            pooled_obs.append(discharge[test_idx])
            pooled_pred.append(np.atleast_1d(pred))

    pooled_obs = np.concatenate(pooled_obs)
    pooled_pred = np.concatenate(pooled_pred)
    pooled_pred = np.clip(pooled_pred, 1e-6, None)
    metrics = _fit_metrics(pooled_obs, pooled_pred)
    metrics["n_folds"] = n_splits
    metrics["n_repeats"] = n_repeats
    metrics["n_held_out_predictions"] = len(pooled_obs)
    return metrics


def select_best_model(stage, discharge, max_segments: int = 3,
                       parsimony_margin: float = 0.02, **cv_kwargs):
    """
    Fit 1-, 2-, and (if enough data) 3-segment power laws, cross-validate each
    on the measured data, and pick the simplest model whose cross-validated
    RMSE (log space) is within ``parsimony_margin`` of the best one - i.e. more
    breakpoints must earn their keep on held-out data, not just fit training
    data more closely. Every candidate is returned too (each with its own
    cross-validation attached to ``.metrics["cv"]``), so a caller can compare
    or display a candidate other than the one auto-selected.

    Returns (best_model: FittedRating, comparison: pandas.DataFrame,
    fitted: dict[int, FittedRating]).
    """
    import pandas as pd

    n = len(stage)
    candidates = [1, 2]
    if max_segments >= 3 and n >= 20:
        candidates.append(3)

    rows = []
    fitted = {}
    cv_results = {}
    for n_seg in candidates:
        model = fit_rating(stage, discharge, n_seg)
        cv = cross_validate(stage, discharge, n_seg, **cv_kwargs)
        model.metrics["cv"] = cv
        fitted[n_seg] = model
        cv_results[n_seg] = cv
        rows.append(
            {
                "model": f"{n_seg}-segment power law" if n_seg > 1 else "power law (1 segment)",
                "n_segments": n_seg,
                "breakpoints_ft": ", ".join(f"{b:.2f}" for b in model.breakpoints) or "-",
                "fit_r2": model.metrics["r2"],
                "fit_r2_log": model.metrics["r2_log"],
                "cv_r2": cv["r2"],
                "cv_r2_log": cv["r2_log"],
                "cv_rmse_cfs": cv["rmse_cfs"],
                "cv_rmse_log": cv["rmse_log"],
                "cv_pbias_pct": cv["pbias_pct"],
            }
        )

    comparison = pd.DataFrame(rows)
    # Selection runs on log-space RMSE, not raw cfs RMSE: discharge spans orders
    # of magnitude, so a linear-space RMSE is dominated by the highest flows and
    # would pick whichever model fits the flood peaks best, regardless of how it
    # does across the more commonly measured low-to-mid flow range.
    best_rmse_log = comparison["cv_rmse_log"].min()
    comparison["selected"] = False
    for n_seg in sorted(candidates):
        row = comparison.loc[comparison["n_segments"] == n_seg].iloc[0]
        if row["cv_rmse_log"] <= best_rmse_log * (1 + parsimony_margin):
            comparison.loc[comparison["n_segments"] == n_seg, "selected"] = True
            chosen = n_seg
            break

    best_model = fitted[chosen]
    return best_model, comparison, fitted
