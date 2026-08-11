# limnotech_rating_curves.data.zero_flow

Estimating the stage of zero flow by Johnson's three-point method.

## Background
A stage-discharge rating is a power law in effective head rather than in stage:

$$
Q = C\,(h - e)^{\beta}
$$

`e` is the stage of zero flow: the gage height at which discharge would reach zero,
usually the elevation of the streambed or of the control's low point. With `e`
correct, `log Q` against `log(h - e)` is a straight line, so the rating fits by
ordinary regression and the exponent `beta` retains its physical meaning. With `e`
wrong the relation curves, and the fit compensates by distorting `beta`, which is
the parameter that governs extrapolation to high flow.

Every Bayesian model in this package estimates `e` itself, so supplying it is
optional. Johnson's method provides a starting point for that estimate, which matters
most where `e` is hardest to identify: few measurements over a narrow stage range.
It can be passed as a prior,

>>> offset = johnson_offset(measurements)                # doctest: +SKIP
>>> rating = fit_rating(measurements, zero_flow=offset)  # doctest: +SKIP

or left to the fit, which calls it by default (`zero_flow="johnson"`).

## The method, and its point-selection rule
Johnson's method (Rantz and others, 1982, *Measurement and Computation of Streamflow*,
USGS WSP 2175, v.2, p. 337; Kennedy, 1984, TWRI 3-A10) solves for `e` in closed form
from three points on the rating. The selection rule is the part most often
misapplied, so it is given in full:

1. Draw a smooth curve through the measurements, the "median curve" of the USGS text.
   This step is required, for the reason given below.
2. Choose three discharges in geometric progression, so that the middle discharge is
   the geometric mean of the outer two,

$$
Q_2 = \sqrt{Q_1 Q_3}
with :math:`Q_1` and :math:`Q_3` bracketing the stage range the rating is to
describe. The progression is geometric, meaning equal ratios in :math:`Q` rather
than equal differences, which is why the middle point is not the middle
measurement.
$$

3. Read the three gage heights from the smooth curve rather than from the nearest raw
   measurements.

Because $Q = C(h-e)^{\beta}$, a geometric progression in $Q$ implies one
in $(h - e)$:

$$
(h_2 - e)^2 = (h_1 - e)(h_3 - e)
$$

which is one equation in one unknown:

$$
e = \frac{h_1 h_3 - h_2^2}{h_1 + h_3 - 2 h_2}
$$

Step 1 is what makes step 3 valid. The identity holds for points on the rating, and a
raw measurement is the rating plus measurement error, so reading $h_2$ from a
noisy point propagates that error into a difference of nearly equal numbers in the
denominator. Smoothing is part of the method rather than a convenience, which is why
this implementation smooths and why `gage_height_of_discharge` is exposed
separately, so the curve it reads from can be inspected.

## This implementation
`gage_height_of_discharge` builds the smooth curve: sort by discharge, average
repeated discharges, take a centered rolling median, force the result non-decreasing
because stage rises with discharge, then interpolate with a shape-preserving PCHIP
spline. The median window is set by sample size, using five points on records long
enough for that to average noise, and no smoothing below `MIN_POINTS_TO_SMOOTH`,
where it would average away real curvature instead.

`johnson_three_points` selects the three points. By default $Q_1$ and
$Q_3$ are the smallest and largest observed discharge, which anchors the estimate
on the full observed range. Narrowing them targets one part of the rating; anchoring on
the upper measurements yields an offset better suited to extrapolating high flow.
`johnson_offset` solves for `e` and reports the accompanying diagnostic: the
log-log straightness before and after subtracting `e`.

## Failure cases
The method cannot be solved when

* fewer than three distinct measurements are available;
* the denominator $h_1 + h_3 - 2h_2$ is approximately zero, meaning the data are
  already log-linear and no offset is resolvable (`e` near 0 is the answer);
* the solved `e` is not below the lowest observed stage, which would make
  $(h - e)$ non-positive somewhere and leave the power law undefined.

In each case `johnson_offset` returns a fallback estimate, with zero flow placed
just below the lowest observation, flagged `method="below_lowest"` and with the
reason in `note`, so a fit is never blocked by an unestimable offset prior. Pass
`on_error="raise"` to surface the failure instead. `estimate_zero_flow` is the
same computation under an explicit name, and is what the fitting code calls.
