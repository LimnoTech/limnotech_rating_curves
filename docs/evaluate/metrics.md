# limnotech_rating_curves.evaluate.metrics

Bayesian model comparison: placing every rating on one predictive score.

The question a metrics table has to answer is not which curve passes closest to the
measurements, since a spline with enough knots always wins that comparison, but which
curve would best predict a measurement it has not seen. That quantity is the expected
log pointwise predictive density (ELPD), estimated here two ways from each model's
pointwise log-likelihood.

PSIS-LOO is the headline estimate: approximate leave-one-out, which for each
measurement reweights the posterior draws to what they would have been without that
measurement and scores the model's prediction of it. Pareto-smoothed importance
sampling stabilizes those weights and reports a per-observation diagnostic, the
Pareto-k, identifying where it could not. A k above 0.7 means that measurement is too
influential for the reweighting to hold, which in a rating almost always means the
highest-flow point is determining the curve on its own. The worst k is reported beside
every ELPD for that reason.

WAIC is reported as a cross-check: a different approximation to the same quantity.
Material disagreement between the two is a reason to distrust both.

Higher ELPD is better. The scale is nats of log predictive density summed over the
measurements, so it has no absolute interpretation. Only differences between models on
the same sample are meaningful, and a difference smaller than about two standard errors
(`se_loo`) is not resolvable by that sample.

## Comparability across the two backends
ELPD is comparable only if every model's log-likelihood is a density over the same
observations in the same space. bdrc's likelihood is Gaussian on raw log discharge.
The ratingcurve package fits a Gaussian on standardized log discharge,
`z = (log q - mean) / std`, so its pointwise log-likelihood is a density of `z`
rather than of `log q`. Moving it into bdrc's space adds `-log(std)` to every
pointwise value, the Jacobian of the standardization, since `log q = mean + std * z`.

That offset is the same constant for every observation, so it shifts `elpd_loo` and
`elpd_waic` by `n * (-log std)` and leaves `p_loo`, `p_waic`, the standard
errors and every Pareto-k unchanged. This module therefore scores the z-space
InferenceData as it stands and applies the offset to the ELPD point estimates
afterwards, without modifying the posterior. Without the correction the two packages'
ELPDs differ by `n * log(std)` and the ranking is meaningless.

The discharge unit, cfs against cms, is a location shift in log space and requires no
correction.

## Two constraints
The ratingcurve models store no `log_likelihood` group, because their fit calls
`approx.sample` or `pm.sample` without one, so `ensure_log_likelihood`
computes it before scoring. That requires the PyMC model in memory, which is why the
metrics pass runs in the same process as the fit.

PSIS-LOO's Pareto-k diagnostic is meaningful only for a well-explored posterior. Fits
scored here should be sampled with NUTS rather than ADVI.
