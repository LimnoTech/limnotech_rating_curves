import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import settings
from ..core import Fit
from ..model_selection.metrics import fit_metrics

log = logging.getLogger(__name__)

#: Degree names, for labels.
DEGREE_NAMES = {1: "linear", 2: "quadratic", 3: "cubic"}


@dataclass
class PolynomialRating:
    """A fitted least-squares polynomial rating.

    Attributes
    ----------
    coefficients : numpy.ndarray
        Highest power first, the order Excel prints them and
        :func:`numpy.polyfit` returns them.
    degree : int
        Polynomial degree - 2 for the quadratic.
    stage_ft, discharge_cfs : numpy.ndarray
        The measurements it was fitted to.
    r_squared : float
        Coefficient of determination on discharge, which is the number the
        spreadsheet reports next to the equation.
    effective_range : tuple of float
        ``(low, high)`` stage over which the curve is usable - increasing, positive,
        and within the measured range.
    observed_range : tuple of float
        ``(min, max)`` of the measured stage.
    turning_point : float or None
        Stage of the parabola's vertex, where the fitted curve stops rising. None for
        a straight line.
    """

    coefficients: np.ndarray
    degree: int
    stage_ft: np.ndarray
    discharge_cfs: np.ndarray
    r_squared: float
    effective_range: tuple
    observed_range: tuple
    turning_point: float | None = None
    _residual_sd: float = field(default=float("nan"), repr=False)
    _gram_inverse: np.ndarray = field(default=None, repr=False)
    _dof: int = field(default=0, repr=False)

    def predict(self, stage, clip: bool = True):
        """Discharge (cfs) at `stage`.

        Parameters
        ----------
        stage : float or array-like
            Gage height or elevation, on whatever axis the fit was made.
        clip : bool, default True
            Return NaN outside :attr:`effective_range`. Pass False to see what the
            raw polynomial does out there, which is mostly a warning about why the
            range exists.

        Returns
        -------
        float or numpy.ndarray
        """
        scalar = np.ndim(stage) == 0
        stages = np.atleast_1d(np.asarray(stage, float))
        predicted = np.polyval(self.coefficients, stages)
        if clip:
            low, high = self.effective_range
            predicted = np.where((stages >= low) & (stages <= high), predicted, np.nan)
        return float(predicted[0]) if scalar else predicted

    def interval(self, stage, level: float = 0.95):
        """The least-squares **prediction** interval at `stage`.

        Not a credible interval: there is no posterior here. This is the ordinary
        regression interval for one new observation,
        ``se = s * sqrt(1 + x' (X'X)^-1 x)`` on Student-t degrees of freedom, so it
        is directly comparable in spirit to the Bayesian models' predictive band but
        is a different object. Reported so a spreadsheet curve can be drawn with
        honest uncertainty rather than as a bare line.

        Returns
        -------
        tuple
            ``(lower, upper)``, floats for a scalar stage and arrays otherwise.
        """
        from scipy import stats

        stages = np.atleast_1d(np.asarray(stage, float))
        predicted = self.predict(stages, clip=False)
        design = np.vander(stages, self.degree + 1)
        leverage = np.einsum("ij,jk,ik->i", design, self._gram_inverse, design)
        se = self._residual_sd * np.sqrt(1.0 + np.clip(leverage, 0.0, None))
        half = stats.t.ppf((1 + level) / 2, self._dof) * se if self._dof > 0 else se * 0
        lower, upper = predicted - half, predicted + half
        if np.ndim(stage) == 0:
            return float(lower[0]), float(upper[0])
        return lower, upper

    @property
    def display_range(self) -> tuple:
        """``(low, high)`` stage the curve is tabulated over when none is given.

        The padded grid every family is drawn on
        (:func:`limnotech_rating_curves.core.padded_stage_grid`), cut back to where
        this polynomial is still rising in the direction the data runs and still
        positive. A quadratic past its vertex turns over and is not a rating, so the
        padding stops there rather than at a fixed fraction.
        """
        from ..core import padded_stage_grid

        padded = padded_stage_grid(self.stage_ft)
        rising = bool(np.corrcoef(self.stage_ft, self.discharge_cfs)[0, 1] >= 0)
        display, _ = _effective_range(self.coefficients, self.degree,
                                      (float(padded.min()), float(padded.max())),
                                      rising)
        return display

    def table(self, stage=None, level: float = 0.95, points=None) -> pd.DataFrame:
        """The fitted curve over its display range, with a prediction band."""
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
        return (f"PolynomialRating(degree={self.degree}, r2={self.r_squared:.4f}, "
                f"effective {low:.2f}-{high:.2f} ft)")


def _effective_range(coefficients, degree, observed_range, rising: bool = True):
    """Where a fitted polynomial is usable: monotone, positive, and measured.

    Parameters
    ----------
    coefficients, degree
        The fitted polynomial.
    observed_range : tuple
        ``(min, max)`` of the fitted x.
    rising : bool, default True
        Whether discharge should *increase* with x. True for a rating against stage.
        False when x runs the other way - fitting against a sensor's distance down to
        the water surface, say, where discharge falls as the reading grows. Taken from
        the data rather than assumed, because picking the wrong limb of the parabola
        would throw away the half of the range that is actually usable.

    Returns
    -------
    tuple
        ``((low, high), turning_point)``.
    """
    low, high = observed_range
    turning = None
    if degree == 2:
        a, b, _ = coefficients
        if a != 0:
            turning = float(-b / (2 * a))
            # an upward parabola rises to the right of its vertex and falls to the
            # left; keep whichever limb matches the direction the data runs
            wants_right_limb = (a > 0) == rising
            if wants_right_limb:
                low = max(low, turning)
            else:
                high = min(high, turning)
    if high <= low:
        # the measured range sits entirely on the wrong limb: there is no usable band
        # at all, and saying so is better than returning an inverted one
        log.warning("the fitted quadratic is %s across the whole measured range "
                    "(vertex at %.4f), so it has no usable band",
                    "falling" if rising else "rising", turning)
        return (float(observed_range[0]), float(observed_range[1])), turning

    # trim any leading stretch where the curve is still negative
    grid = np.linspace(low, high, 512)
    positive = np.polyval(coefficients, grid) > 0
    if positive.any():
        low = float(grid[positive][0])
        high = float(grid[positive][-1])
    return (float(low), float(high)), turning


def fit_polynomial(stage, discharge, degree: int = 2,
                   stage_range=None) -> PolynomialRating:
    """Fit ``Q`` as a polynomial in stage by ordinary least squares.

    The same computation as an Excel chart trendline of type ``poly``: no transform
    of either axis, no weighting, no offset. :func:`numpy.polynomial.polynomial.polyfit`
    does it through an SVD least squares, which is the stable way to solve what Excel
    solves with normal equations.

    Parameters
    ----------
    stage, discharge : array-like
        Measured stage (ft, on whatever axis - gage height or elevation) and
        discharge (cfs). Non-finite pairs are dropped.
    degree : int, default 2
        Polynomial degree. 2 reproduces the usual spreadsheet curve.
    stage_range : tuple of float, optional
        Override the effective range, when you know the curve is usable over a band
        the data alone does not imply. Given as ``(low, high)`` in stage units.

    Returns
    -------
    PolynomialRating

    Raises
    ------
    ValueError
        If fewer than ``degree + 1`` usable measurements remain - the fit would be
        exactly determined or undetermined, and either way tells you nothing.
    """
    stage = np.asarray(stage, float).ravel()
    discharge = np.asarray(discharge, float).ravel()
    usable = np.isfinite(stage) & np.isfinite(discharge)
    stage, discharge = stage[usable], discharge[usable]
    degree = int(degree)
    if stage.size < degree + 2:
        raise ValueError(
            f"a degree-{degree} fit needs at least {degree + 2} measurements to say "
            f"anything about fit quality, and {stage.size} were given")

    # numpy.polynomial's least squares (SVD) rather than the legacy polyfit, which
    # warns about exactly the conditioning this data has
    ascending = np.polynomial.polynomial.polyfit(stage, discharge, degree)
    coefficients = ascending[::-1].copy()          # highest power first, as Excel

    predicted = np.polyval(coefficients, stage)
    residuals = discharge - predicted
    total = float(np.sum((discharge - discharge.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residuals ** 2) / total) if total > 0 else float("nan")

    dof = stage.size - (degree + 1)
    residual_sd = float(np.sqrt(np.sum(residuals ** 2) / dof)) if dof > 0 else float("nan")
    design = np.vander(stage, degree + 1)
    gram_inverse = np.linalg.pinv(design.T @ design)

    observed = (float(stage.min()), float(stage.max()))
    # which way the data runs, so the usable limb of the parabola is chosen from the
    # measurements rather than assumed
    rising = bool(np.corrcoef(stage, discharge)[0, 1] >= 0)
    effective, turning = _effective_range(coefficients, degree, observed, rising)
    if stage_range is not None:
        effective = (float(stage_range[0]), float(stage_range[1]))

    return PolynomialRating(
        coefficients=coefficients, degree=degree, stage_ft=stage,
        discharge_cfs=discharge, r_squared=r_squared, effective_range=effective,
        observed_range=observed, turning_point=turning,
        _residual_sd=residual_sd, _gram_inverse=gram_inverse, _dof=dof)


def fit(sample, *, key: str, label: str, degree: int = 2, min_points: int = 4,
        enforce_min_points: bool = True, stage_range=None, level: float = 0.95,
        **ignored) -> Fit:
    """Fit a polynomial rating and package it as a :class:`Fit`.

    Mirrors the Bayesian backends' entry point so the catalog can treat every family
    the same way. Keyword arguments that only mean something to a sampler (``method``,
    ``seed``, ``nuts_sampler``, ``zero_flow``, ...) are accepted and ignored: an
    ordinary least-squares fit has no sampler and no offset parameter.
    """
    frame = sample.to_frame() if hasattr(sample, "to_frame") else pd.DataFrame(sample)
    stage = frame["stage_ft"].to_numpy(float)
    discharge = frame["discharge_cfs"].to_numpy(float)

    fit = Fit(key=key, label=label, family="polynomial", n=len(frame),
              status="skipped", reason="",
              config={"degree": degree, "method": "least_squares",
                      "stage_range": None if stage_range is None
                                     else list(stage_range),
                      "interval": f"{level:.0%} prediction interval"})
    if enforce_min_points and len(frame) < min_points:
        fit.reason = (f"{len(frame)} measurements, fewer than the {min_points} a "
                      f"degree-{degree} fit needs")
        return fit
    try:
        fitted = fit_polynomial(stage, discharge, degree=degree,
                                stage_range=stage_range)
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
        "coefficients": [float(value) for value in fitted.coefficients],
        "r_squared": fitted.r_squared,
        "effective_range": list(fitted.effective_range),
        "turning_point": fitted.turning_point})
    return fit
