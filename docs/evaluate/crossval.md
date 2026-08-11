# limnotech_rating_curves.evaluate.crossval

Cross-validation: scoring a rating on measurements it was not fitted to.

The predictive score in the metrics table (`elpd_loo`) approximates leave-one-out
from a single fit by reweighting its posterior draws. This module performs the literal
computation instead: refit the model without some measurements and score its
prediction of them. The two estimate the same quantity, and where they disagree the
explicit refits are the more reliable.

## Two schemes, chosen by sample size

```
================  =============  =================================================
sample            scheme         what it does
================  =============  =================================================
tens of points    ``"holdout"``  Train on 10% of the measurements and predict the
(a USGS gage)                    other 90%, over twelve random draws. A demanding
                                 test by design: it measures whether the model
                                 recovers a rating from few points, and how much
                                 it varies between draws.
a handful         ``"loo"``      Leave one measurement out at a time, ``n`` folds,
(a MAGL sensor)                  training on ``n-1``.
================  =============  =================================================
```

`scheme="auto"`, the default, selects by sample size: the holdout sweep at
`settings.CV_HOLDOUT_MIN_POINTS` measurements or more, leave-one-out below that.

Leave-one-out is required rather than preferred at small sample sizes. The holdout
scheme trains on `max(3, round(0.10 n))`, so on six measurements it fits three, and
the segmented power laws and the spline fail on every fold.

## Two details this module handles
Leave-one-out is scored pooled rather than per fold. NSE divides by the variance of the
held-out observations, and a single point has none, so a per-fold NSE under
leave-one-out is NaN by construction. The quantity to report is the pooled one:
collect every fold's single held-out prediction into one vector of `n` predictions
and score it once. That is the standard meaning of "leave-one-out cross-validated
NSE", and it is what `CrossValidation.pooled` reports. Under the holdout scheme
each fold holds out many points, so per-fold scores are meaningful and both are
reported.

A leave-one-out split is not stratified. The holdout scheme draws its training set
stratified across the stage range, so that a fold is not handed an all-low-flow
training set. Applying the same machinery with `n_train = n-1` does not produce
leave-one-out: sorting by stage into `n-1` bins leaves most bins holding one point,
which is therefore always selected for training, so only the single over-full bin can
contribute a test point and the high-flow measurements are never tested. At `n-1`
training points stratification protects against nothing, so
`leave_one_out_splits` constructs the exhaustive partition directly.

## Fold fits use the fast method
Folds are fitted with ADVI by default (`fold_method="advi"`), because a sweep is
many refits and its purpose is to show how the curve moves when the data change rather
than to produce an ELPD. Folds therefore carry no ELPD: the Bayesian block comes only
from the all-data NUTS fit. A table placing `test_nse` beside `elpd_loo` mixes two
fitting regimes and should say so.
