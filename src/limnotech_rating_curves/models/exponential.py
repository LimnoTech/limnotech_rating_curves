import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import settings
from ..core import Fit
from ..model_selection.metrics import fit_metrics

log = logging.getLogger(__name__)


@dataclass
class ExponentialRating:
    """A fitted exponential rating ``Q = amplitude * exp(rate * stage)``.

    Attributes
    ----------
    amplitude : float
        ``A`` - the fitted discharge at :attr:`stage_offset`, which is where the
        exponential is anchored. Not the discharge at stage zero: on an elevation axis
        that number is ``e^-470`` and does not exist in floating point (see the module
        docstring).
    stage_offset : float
        ``h0`` - the mean measured stage, subtracted before exponentiating.
    rate : float
        ``B`` - the log-discharge gradient per foot of stage. Positive on a stage
        axis; the workbooks' stored rates are negative because they consume radar
        distance to water, which falls as stage rises.
    stage_ft, discharge_cfs : numpy.ndarray
        The measurements it was fitted to, after dropping non-positive discharge.
    r_squared : float
        Coefficient of determination on **discharge**, comparable with the
        polynomial's.
    r_squared_log : float
        The same on log discharge - the number Excel would print. Always the larger
        of the two.
    effective_range : tuple of float
        ``(low, high)`` stage the curve may be used over: the measured range.
    observed_range : tuple of float
        ``(min, max)`` measured stage. The same as `effective_range`, kept for
        interface symmetry with :class:`~limnotech_rating_curves.models.polynomial.PolynomialRating`.
    """

    amplitude: float
    rate: float
    stage_offset: float
    stage_ft: np.ndarray
    discharge_cfs: np.ndarray
    r_squared: float
    r_squared_log: float
    effective_range: tuple
    observed_range: tuple
    _residual_sd_log: float = field(default=float("nan"), repr=False)
    _gram_inverse: np.ndarray = field(default=None, repr=False)
    _dof: int = field(default=0, repr=False)

    def predict(self, stage, clip: bool = True):
        """Discharge (cfs) at `stage`.

        Parameters
        ----------
        stage : float or array-like
            Stage, on whatever axis the fit was made.
        clip : bool, default True
            Return NaN outside :attr:`effective_range`. An exponential is well
            behaved everywhere, so this is a statement about where the *fit* is
            supported rather than about where the function is defined - but a rating
            extrapolated exponentially goes wrong fast, so clipping is the default.

        Returns
        -------
        float or numpy.ndarray
        """
        scalar = np.ndim(stage) == 0
        stages = np.atleast_1d(np.asarray(stage, float))
        predicted = self.amplitude * np.exp(self.rate * (stages - self.stage_offset))
        if clip:
            low, high = self.effective_range
            predicted = np.where((stages >= low) & (stages <= high), predicted, np.nan)
        return float(predicted[0]) if scalar else predicted

    def interval(self, stage, level: float = 0.95):
        """The least-squares **prediction** interval, computed in log space.

        The regression is on ``log Q``, so the interval is the ordinary Student-t
        prediction interval there, exponentiated. It is therefore multiplicative and
        asymmetric in cfs: the upper half is wider than the lower. That is the honest
        shape for a rating error, and it is why this is not simply the polynomial's
        interval with a different mean.

        Returns
        -------
        tuple
            ``(lower, upper)`` in cfs - floats for a scalar stage, arrays otherwise.
        """
        from scipy import stats

        stages = np.atleast_1d(np.asarray(stage, float))
        centred = stages - self.stage_offset
        design = np.vander(centred, 2)                 # [stage, 1], matching the fit
        leverage = np.einsum("ij,jk,ik->i", design, self._gram_inverse, design)
        se = self._residual_sd_log * np.sqrt(1.0 + np.clip(leverage, 0.0, None))
        half = (stats.t.ppf((1 + level) / 2, self._dof) * se if self._dof > 0
                else se * 0)
        center = np.log(self.amplitude) + self.rate * centred
        lower, upper = np.exp(center - half), np.exp(center + half)
        if np.ndim(stage) == 0:
            return float(lower[0]), float(upper[0])
        return lower, upper

    @property
    def display_range(self) -> tuple:
        """``(low, high)`` stage the curve is tabulated over when none is given.

        The padded grid every family is drawn on
        (:func:`limnotech_rating_curves.core.padded_stage_grid`). An exponential is
        defined everywhere, so nothing cuts it back - which is also why the part of
        it outside the measurements is drawn as extrapolation.
        """
        from ..core import padded_stage_grid

        padded = padded_stage_grid(self.stage_ft)
        return (float(padded.min()), float(padded.max()))

    def table(self, stage=None, level: float = 0.95, points=None) -> pd.DataFrame:
        """The fitted curve over its display range, with the log-space band."""
        if stage is None:
            low, high = self.display_range
            stage = np.linspace(low, high, points or settings.GRID_POINTS)
        stage = np.atleast_1d(np.asarray(stage, float))
        discharge = self.predict(stage, clip=False)
        lower, upper = self.interval(stage, level=level)
        return pd.DataFrame({"stage_ft": stage, "discharge_cfs": discharge,
                             "discharge_median_cfs": discharge,
                             "lower": np.atleast_1d(lower),
                             "upper": np.atleast_1d(upper)})

    def __repr__(self):
        low, high = self.effective_range
        return (f"ExponentialRating(rate={self.rate:.4g}, r2={self.r_squared:.4f}, "
                f"r2_log={self.r_squared_log:.4f}, {low:.2f}-{high:.2f} ft)")


def fit_exponential(stage, discharge, min_points: int = 3) -> ExponentialRating:
    """Fit ``Q = A exp(B h)`` the way Excel's exponential trendline does.

    Ordinary least squares of ``log(Q)`` on stage, which is what
    ``trendlineType="exp"`` computes. Discharge that is zero or negative cannot be
    logged and is dropped.

    Parameters
    ----------
    stage, discharge : array-like
        Measured stage (ft, on whatever axis) and discharge (cfs).
    min_points : int, default 3
        Fewest usable measurements. Two points determine a line through
        ``log Q`` exactly and say nothing about fit quality, so three is the floor.

    Returns
    -------
    ExponentialRating

    Raises
    ------
    ValueError
        If fewer than `min_points` positive measurements remain.
    """
    stage = np.asarray(stage, float).ravel()
    discharge = np.asarray(discharge, float).ravel()
    usable = np.isfinite(stage) & np.isfinite(discharge) & (discharge > 0)
    dropped = int((~usable).sum())
    stage, discharge = stage[usable], discharge[usable]
    if dropped:
        log.info("dropped %d measurement(s) whose discharge could not be logged",
                 dropped)
    if stage.size < min_points:
        raise ValueError(f"an exponential fit needs at least {min_points} positive "
                         f"measurements, and {stage.size} were given")

    log_discharge = np.log(discharge)
    # centred, or an elevation axis overflows the exponential - see the module docstring
    stage_offset = float(np.mean(stage))
    centred = stage - stage_offset
    rate, log_amplitude = np.polyfit(centred, log_discharge, 1)
    amplitude = float(np.exp(log_amplitude))
    rate = float(rate)

    predicted = amplitude * np.exp(rate * centred)
    total = float(np.sum((discharge - discharge.mean()) ** 2))
    r_squared = (float(1.0 - np.sum((discharge - predicted) ** 2) / total)
                 if total > 0 else float("nan"))

    log_residuals = log_discharge - (log_amplitude + rate * centred)
    log_total = float(np.sum((log_discharge - log_discharge.mean()) ** 2))
    r_squared_log = (float(1.0 - np.sum(log_residuals ** 2) / log_total)
                     if log_total > 0 else float("nan"))

    dof = stage.size - 2
    residual_sd_log = (float(np.sqrt(np.sum(log_residuals ** 2) / dof)) if dof > 0
                       else float("nan"))
    design = np.vander(centred, 2)
    gram_inverse = np.linalg.pinv(design.T @ design)
    observed = (float(stage.min()), float(stage.max()))

    return ExponentialRating(
        amplitude=amplitude, rate=rate, stage_offset=stage_offset,
        stage_ft=stage, discharge_cfs=discharge,
        r_squared=r_squared, r_squared_log=r_squared_log, effective_range=observed,
        observed_range=observed, _residual_sd_log=residual_sd_log,
        _gram_inverse=gram_inverse, _dof=dof)


def fit(sample, *, key: str, label: str, min_points: int = 3,
        enforce_min_points: bool = True, level: float = 0.95, **ignored) -> Fit:
    """Fit an exponential rating and package it as a :class:`Fit`.

    Mirrors :func:`limnotech_rating_curves.models.polynomial.fit` so the catalog can
    treat every family the same way. Sampler keywords (``method``, ``seed``,
    ``zero_flow``, ``nuts_sampler``, ...) are accepted and ignored: a log-linear
    least-squares fit has none of them.
    """
    frame = sample.to_frame() if hasattr(sample, "to_frame") else pd.DataFrame(sample)
    stage = frame["stage_ft"].to_numpy(float)
    discharge = frame["discharge_cfs"].to_numpy(float)

    fit = Fit(
        key=key, label=label, family="exponential", n=len(frame),
        status="skipped", reason="",
        config={"method": "least_squares_log", "min_points": min_points,
                "interval": f"{level:.0%} prediction interval, computed on log Q"})
    if enforce_min_points and len(frame) < min_points:
        fit.reason = (f"{len(frame)} measurements, fewer than the {min_points} an "
                         f"exponential fit needs")
        return fit
    try:
        fitted = fit_exponential(stage, discharge, min_points=min_points)
    except Exception as exc:  # noqa: BLE001
        fit.status = "failed"
        fit.reason = f"{type(exc).__name__}: {exc}"
        return fit

    fit.status = "ok"
    fit.rating = fitted
    fit.curve = fitted.table(level=level)
    fit.predicted = fitted.predict(stage, clip=False)
    fit.metrics = fit_metrics(discharge, fit.predicted)
    fit.config.update({
        "amplitude": fitted.amplitude, "rate": fitted.rate,
        "stage_offset": fitted.stage_offset,
        # both, deliberately: the log-space number is what Excel shows and it is
        # always the flattering one
        "r_squared": fitted.r_squared, "r_squared_log": fitted.r_squared_log,
        "effective_range": list(fitted.effective_range)})
    return fit
