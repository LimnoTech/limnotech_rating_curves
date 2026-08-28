import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .. import settings

log = logging.getLogger(__name__)


@dataclass
class ZeroFlowEstimate:
    """A stage-of-zero-flow estimate and the evidence for it.

    Attributes
    ----------
    stage_ft : float
        The estimated stage of zero flow, ``e``, in feet on the same axis as the
        stage that was passed in. Always below the lowest observed stage.
    method : {'johnson', 'below_lowest'}
        How it was obtained. ``"below_lowest"`` means Johnson's method could not
        be solved and the value is the fallback, which is a placeholder rather
        than an estimate.
    points : pandas.DataFrame or None
        The three points Johnson's method used, indexed ``low``/``mid``/``high``
        with columns ``discharge_cfs`` and ``stage_ft``. ``None`` for the
        fallback.
    r2_log_raw : float
        Straightness (R-squared) of ``log Q`` against ``log h`` with no offset.
    r2_log_offset : float
        The same after subtracting ``e``. A clear improvement over `r2_log_raw` is
        the check that the estimate is doing its job; no improvement means the
        rating was already log-linear and the offset is not identified.
    note : str
        Human summary, including any fallback reason.
    """

    stage_ft: float
    method: str
    points: object = None
    r2_log_raw: float = float("nan")
    r2_log_offset: float = float("nan")
    note: str = ""

    def __float__(self) -> float:
        """The estimate itself, so it can be used anywhere a number is expected."""
        return float(self.stage_ft)

    def __repr__(self) -> str:
        return (f"ZeroFlowEstimate(stage_ft={self.stage_ft:.4f}, "
                f"method={self.method!r}, r2_log {self.r2_log_raw:.4f} -> "
                f"{self.r2_log_offset:.4f})")


def gage_height_of_discharge(stage, discharge, smooth_window=None):
    """The smooth "median curve" Johnson's method reads its three points off.

    Builds a monotone, non-decreasing estimate of gage height as a function of
    discharge. Reading off this rather than off raw measurements is what makes the
    geometric-mean identity hold (see ``docs/data/zero_flow.md``).

    Parameters
    ----------
    stage, discharge : array-like
        Measured gage height (feet) and discharge (cfs). Non-positive and
        non-finite values are dropped.
    smooth_window : int, optional
        Width of the centered rolling median applied to stage after sorting by
        discharge. ``None`` (the default) chooses by sample size: a
        ``settings.ZERO_FLOW_SMOOTH_WINDOW``-point median at
        ``settings.ZERO_FLOW_MIN_POINTS_TO_SMOOTH``
        measurements or more, and no smoothing below that, where a median window
        would flatten real curvature rather than remove noise. Pass 1 or 0 to force
        smoothing off, or a number to force it on.

        Measurements near either end, whose window is not complete, keep their
        measured stage. A median taken over a one-sided window is drawn from the
        interior side, so on a record where stage rises with discharge it moves
        both extremes inward on every sample, noisy or not. Those extremes are
        :math:`h_1` and :math:`h_3`, the two heights Johnson's method is most
        sensitive to, and pairing them with a neighbour's stage biases the solved
        offset upward - far enough, on a clean sample, to push it above the lowest
        measurement and make it unusable.

    Returns
    -------
    interpolator : scipy.interpolate.PchipInterpolator
        Maps discharge (cfs) to gage height (feet).
    discharge_sorted : numpy.ndarray
        The distinct discharges the interpolator was built on, ascending. Its
        first and last values are the default :math:`Q_1` and :math:`Q_3`.

    Raises
    ------
    ValueError
        If fewer than three distinct positive measurements remain.
    """
    pairs = pd.DataFrame({"stage_ft": np.asarray(stage, float).ravel(),
                          "discharge_cfs": np.asarray(discharge, float).ravel()})
    pairs = pairs[np.isfinite(pairs["stage_ft"]) & np.isfinite(pairs["discharge_cfs"])
                  & (pairs["discharge_cfs"] > 0)]
    # repeated discharges would break the interpolator's strict monotonicity
    pairs = (pairs.groupby("discharge_cfs", as_index=False)["stage_ft"].mean()
             .sort_values("discharge_cfs"))
    if len(pairs) < 3:
        raise ValueError(f"Johnson's method needs at least 3 distinct positive "
                         f"(stage, discharge) pairs, got {len(pairs)}")
    if smooth_window is None:
        smooth_window = (settings.ZERO_FLOW_SMOOTH_WINDOW
                         if len(pairs) >= settings.ZERO_FLOW_MIN_POINTS_TO_SMOOTH
                         else 1)
    if smooth_window and smooth_window > 1:
        smoothed = pairs["stage_ft"].rolling(smooth_window, center=True).median()
        pairs["stage_ft"] = smoothed.fillna(pairs["stage_ft"])
    # stage rises with discharge; enforce it so the interpolation is monotone
    heights = np.maximum.accumulate(pairs["stage_ft"].to_numpy(float))
    discharges = pairs["discharge_cfs"].to_numpy(float)
    return PchipInterpolator(discharges, heights), discharges


def johnson_three_points(stage, discharge, discharge_low=None, discharge_high=None,
                         smooth_window=None) -> pd.DataFrame:
    """The three points Johnson's method selects, in geometric progression.

    Parameters
    ----------
    stage, discharge : array-like
        Measured gage height (feet) and discharge (cfs).
    discharge_low, discharge_high : float, optional
        :math:`Q_1` and :math:`Q_3`, the discharges bracketing the part of the
        rating you want the offset to describe. Default to the smallest and
        largest observed discharge. Narrow them to the range you actually care
        about - for a rating you intend to extrapolate upward, anchoring on the
        upper measurements gives an offset better suited to high flow.
    smooth_window : int, optional
        Passed to :func:`gage_height_of_discharge`; ``None`` chooses by sample size.

    Returns
    -------
    pandas.DataFrame
        Three rows indexed ``low``, ``mid``, ``high``, with columns
        ``discharge_cfs`` and ``stage_ft``. The middle discharge is
        :math:`\\sqrt{Q_1 Q_3}` and all three gage heights are read off the smooth
        curve.

    Raises
    ------
    ValueError
        If the bracketing discharges are not ``0 < low < high``.
    """
    curve, discharges = gage_height_of_discharge(stage, discharge, smooth_window)
    low = float(discharges.min()) if discharge_low is None else float(discharge_low)
    high = float(discharges.max()) if discharge_high is None else float(discharge_high)
    if not (0 < low < high):
        raise ValueError(f"need 0 < discharge_low < discharge_high, got "
                         f"{low} and {high}")
    middle = float(np.sqrt(low * high))          # the geometric progression
    selected = np.array([low, middle, high])
    return pd.DataFrame({"discharge_cfs": selected, "stage_ft": curve(selected)},
                        index=["low", "mid", "high"])


def johnson_offset(data, discharge=None, *, discharge_low=None, discharge_high=None,
                   smooth_window=None, on_error="fallback") -> ZeroFlowEstimate:
    """Estimate the stage of zero flow by Johnson's three-point method.

    Parameters
    ----------
    data : Sample or pandas.DataFrame or array-like
        The measurements - anything :meth:`Sample.of` accepts, so a
        :class:`~limnotech_rating_curves.core.Sample`, a DataFrame with stage
        and discharge columns, or an array of stage with `discharge` given
        separately.
    discharge : array-like, optional
        Discharge (cfs) when `data` is an array of stage values.
    discharge_low, discharge_high : float, optional
        The bracketing discharges :math:`Q_1` and :math:`Q_3` (see
        :func:`johnson_three_points`).
    smooth_window : int, optional
        Rolling-median window for the smooth curve; ``None`` chooses by sample size
        (see :func:`gage_height_of_discharge`).
    on_error : {'fallback', 'raise'}, optional
        What to do when the method cannot be solved on this data. The default
        ``"fallback"`` returns a ``method="below_lowest"`` estimate, placing zero
        flow ``settings.ZERO_FLOW_FALLBACK_PAD_FRACTION`` of the stage range below
        the lowest
        measurement, with the reason in ``note``. ``"raise"`` re-raises instead.

    Returns
    -------
    ZeroFlowEstimate
        The estimate, the three points it came from, and the log-log straightness
        before and after applying it.

    Raises
    ------
    ValueError
        With ``on_error="raise"``, if the method cannot be solved on this data -
        degenerate points, or a solution not below the lowest observed stage.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> stage = np.array([2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    >>> measured = pd.DataFrame({"stage_ft": stage,
    ...                          "discharge_cfs": 12.0 * (stage - 1.5) ** 2.1})
    >>> estimate = johnson_offset(measured)
    >>> round(estimate.stage_ft, 2)
    1.5
    """
    if on_error not in ("fallback", "raise"):
        raise ValueError(f"on_error must be 'fallback' or 'raise', got {on_error!r}")
    from ..core import Sample
    sample = Sample.of(data, discharge)
    try:
        return _solve_johnson(sample, discharge_low, discharge_high, smooth_window)
    except Exception as exc:  # noqa: BLE001 - any failure means use the fallback
        if on_error == "raise":
            raise
        return _below_lowest(sample, exc)


def _solve_johnson(sample, discharge_low, discharge_high,
                   smooth_window) -> ZeroFlowEstimate:
    """Johnson's method with no fallback: raises if it cannot be solved."""
    stage, flow = sample.stage_ft, sample.discharge_cfs

    points = johnson_three_points(stage, flow, discharge_low, discharge_high,
                                  smooth_window)
    h1, h2, h3 = points["stage_ft"].to_numpy(float)
    denominator = h1 + h3 - 2.0 * h2
    if abs(denominator) < 1e-12:
        raise ValueError(
            "Johnson's method is degenerate on this sample: h1 + h3 - 2*h2 is ~0, "
            "which means log Q is already linear in log h and no offset is "
            "identified (e is effectively 0)")
    offset = float((h1 * h3 - h2 ** 2) / denominator)

    lowest = float(np.nanmin(stage))
    if offset >= lowest:
        raise ValueError(
            f"Johnson's method solved e = {offset:.4f} ft, which is not below the "
            f"lowest observed stage {lowest:.4f} ft; (h - e) would be non-positive "
            f"so the power law is undefined. Narrow discharge_low/discharge_high, "
            f"or pass on_error='fallback' for a usable estimate.")

    raw = loglog_r2(stage, flow, 0.0)
    corrected = loglog_r2(stage, flow, offset)
    return ZeroFlowEstimate(
        stage_ft=offset, method="johnson", points=points,
        r2_log_raw=raw, r2_log_offset=corrected,
        note=(f"Johnson three-point method on Q = {points['discharge_cfs'].iloc[0]:.3g}"
              f" / {points['discharge_cfs'].iloc[1]:.3g}"
              f" / {points['discharge_cfs'].iloc[2]:.3g} cfs; "
              f"log-log R^2 {raw:.4f} -> {corrected:.4f}"))


def _below_lowest(sample, exc) -> ZeroFlowEstimate:
    """The fallback: zero flow placed a little below the lowest measurement."""
    low, high = sample.stage_range
    span = (high - low) or 1.0
    pad = settings.ZERO_FLOW_FALLBACK_PAD_FRACTION
    fallback = float(low - pad * span)
    log.debug("Johnson's method unavailable (%s); placing zero flow at %.4f ft",
              exc, fallback)
    return ZeroFlowEstimate(
        stage_ft=fallback, method="below_lowest",
        r2_log_raw=loglog_r2(sample.stage_ft, sample.discharge_cfs, 0.0),
        r2_log_offset=loglog_r2(sample.stage_ft, sample.discharge_cfs, fallback),
        note=(f"Johnson's method unavailable ({type(exc).__name__}: {exc}); "
              f"placed zero flow {pad:.0%} of the stage range "
              f"below the lowest measurement"))


def estimate_zero_flow(data, discharge=None, **kwargs) -> ZeroFlowEstimate:
    """Estimate the stage of zero flow, falling back rather than failing.

    Tries Johnson's method; if that cannot be solved, places zero flow a little
    below the lowest observed stage. This is what the fitting code calls when
    building an offset prior, so a fit is never blocked by an offset that could not
    be estimated. Same as :func:`johnson_offset` with its default
    ``on_error="fallback"``.

    Parameters
    ----------
    data : Sample or pandas.DataFrame or array-like
        The measurements.
    discharge : array-like, optional
        Discharge (cfs) when `data` is an array of stage values.
    **kwargs
        Forwarded to :func:`johnson_offset`.

    Returns
    -------
    ZeroFlowEstimate
        With ``method="johnson"`` on success, or ``method="below_lowest"`` and the
        reason in ``note`` on fallback.
    """
    kwargs.setdefault("on_error", "fallback")
    return johnson_offset(data, discharge, **kwargs)


def loglog_r2(stage, discharge, zero_flow: float) -> float:
    """Straightness of ``log Q`` against ``log(h - e)``, as an R-squared.

    The diagnostic for an offset: subtracting the right ``e`` should straighten
    the log-log relation. Fitted here by ordinary least squares on the pairs where
    ``h - e`` and ``Q`` are both positive.

    Parameters
    ----------
    stage, discharge : array-like
        Gage height (feet) and discharge (cfs).
    zero_flow : float
        The offset ``e`` to test. Pass 0 for the uncorrected case.

    Returns
    -------
    float
        R-squared of the log-log line, or NaN if fewer than three usable pairs
        remain.
    """
    stage = np.asarray(stage, float).ravel()
    discharge = np.asarray(discharge, float).ravel()
    usable = (np.isfinite(stage) & np.isfinite(discharge)
              & (stage - zero_flow > 0) & (discharge > 0))
    if usable.sum() < 3:
        return float("nan")
    x = np.log(stage[usable] - zero_flow)
    y = np.log(discharge[usable])
    line = np.polyfit(x, y, 1)
    residual = y - np.polyval(line, x)
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return float(1.0 - np.sum(residual ** 2) / total)


def plot_johnson(data, discharge=None, ax=None, **kwargs):
    """Show what Johnson's method did: the smooth curve and its three points.

    Two panels - stage against discharge with the smoothed curve and the three
    selected points marked, and the log-log relation before and after subtracting
    the estimated offset, which is where you can see whether the offset helped.

    Parameters
    ----------
    data : Sample or pandas.DataFrame or array-like
        The measurements.
    discharge : array-like, optional
        Discharge (cfs) when `data` is an array of stage values.
    ax : sequence of matplotlib.axes.Axes, optional
        Two axes to draw into. Created if not given.
    **kwargs
        Forwarded to :func:`johnson_offset`.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The two axes.
    """
    import matplotlib.pyplot as plt
    from ..core import Sample

    sample = Sample.of(data, discharge)
    estimate = estimate_zero_flow(sample, **kwargs)
    curve, discharges = gage_height_of_discharge(sample.stage_ft, sample.discharge_cfs)

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = np.atleast_1d(ax)

    grid = np.geomspace(discharges.min(), discharges.max(), 200)
    ax[0].plot(sample.discharge_cfs, sample.stage_ft, "o", mfc="white", mec="black",
               label="measurements")
    ax[0].plot(grid, curve(grid), "-", color="#4c72b0", lw=2,
               label="smoothed median curve")
    if estimate.points is not None:
        ax[0].plot(estimate.points["discharge_cfs"], estimate.points["stage_ft"],
                   "D", color="#c44e52", ms=9, label="three selected points")
    ax[0].axhline(estimate.stage_ft, color="#c44e52", ls="--",
                  label=f"stage of zero flow = {estimate.stage_ft:.3f} ft")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("discharge (cfs, log)")
    ax[0].set_ylabel(f"stage - {sample.stage_label}")
    ax[0].set_title(f"Johnson's method ({estimate.method})")
    ax[0].legend(fontsize="small")

    for offset, style, name in ((0.0, "o", "no offset"),
                                (estimate.stage_ft, "s", "offset applied")):
        usable = (sample.stage_ft - offset > 0) & (sample.discharge_cfs > 0)
        r2 = loglog_r2(sample.stage_ft, sample.discharge_cfs, offset)
        ax[1].plot(sample.stage_ft[usable] - offset, sample.discharge_cfs[usable],
                   style, mfc="none", label=f"{name}: R²log = {r2:.4f}")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("h − e (ft, log)")
    ax[1].set_ylabel("discharge (cfs, log)")
    ax[1].set_title("does the offset straighten the log-log relation?")
    ax[1].legend(fontsize="small")
    return ax
