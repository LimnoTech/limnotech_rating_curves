# limnotech_rating_curves.models.exponential

Exponential ratings: the third form the field spreadsheets use.

Three MAGL workbooks (SBR-06, SBR-07, SBR-08) carry `Q = A e^(B d)` rather than a
polynomial, fitted as an Excel chart trendline of type `exp`. This module reproduces
that computation, so those sensors can be compared against the form their own field
curve uses rather than against a quadratic.

    rating = lrc.fit_rating(measurements, model="exponential")
    rating.equation()             # 'Q = 41.2 exp(1.284 (h - 5.5))'
    rating.rate, rating.amplitude, rating.stage_offset

How Excel fits it, and therefore how this module fits it: `trendlineType="exp"` is
ordinary least squares of `log Q` on stage, not a nonlinear least-squares fit of
`Q` itself. The two give different answers, and the log-space fit weights the low
flows considerably more heavily. Matching the spreadsheet requires matching that
choice, which is what `fit_exponential` does.

Two consequences follow:

* The interval is computed in log space and exponentiated, so it is asymmetric in
  discharge. It is a different quantity from the polynomial's symmetric prediction
  interval and from the Bayesian models' credible band, and the three are not
  interchangeable.
* R-squared is reported on discharge rather than on log discharge. Excel prints the
  log-space value beside an exponential trendline, which overstates the fit: a curve
  can fit `log Q` well and be substantially wrong in cfs at the high end. Both are
  available (`nse` and `r_squared` on Q, `r2_log` on log Q) and both are
  reported.

This module has no effective-range logic, unlike the polynomial: `A e^(Bh)` with
`A > 0` is positive and monotone everywhere, so the usable band is the measured
range. There is no vertex and no limb to select.

## Centring the stage axis
MAGL stage is a NAVD88 elevation, so `h` is around 800 ft. With a fitted rate near
0.6 per foot, `e^(B h)` is `e^480`, which overflows to infinity, while the matching
amplitude `e^(-470)` underflows to zero and their product is NaN. The curve is well
defined; the parameterization is not.

The fit is therefore carried out and evaluated on `h - h0`, where `h0` is the mean
measured stage. That is the same least-squares problem, since shifting the predictor
changes only the intercept, and the same curve, evaluated over a range a float can
represent. `amplitude` is consequently the discharge at `h0` rather than at
stage zero, and `equation` prints the shift explicitly instead of folding it into
a coefficient that cannot be represented.
