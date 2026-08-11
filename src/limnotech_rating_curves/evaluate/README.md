# evaluate

Scoring fitted ratings: cross-validation, the Bayesian predictive scores, exact
leave-one-out where the PSIS approximation failed, and sampler convergence.

```python
from limnotech_rating_curves import evaluate

cv = evaluate.cross_validate(sample)   # or sample.cross_validate()
cv.headline
cv.plot()
print(evaluate.convergence_report(rating))
```

## `crossval.py` — refit and score held-out measurements

| Name | Purpose |
| --- | --- |
| `cross_validate(sample, models=None, scheme="auto", ...)` | Refit each model on subsets and score what it did not see |
| `CrossValidation` | `.headline` (pooled for LOO, per-fold otherwise), `.curves_by_model`, `.to_map_payload`, `.plot(model=None)` |
| `pooled_metrics(curves)` | Scores over all held-out predictions pooled |
| `per_fold_metrics(folds)` | Mean and standard deviation across folds |
| `high_flow_skill(result, quantile=0.75)` | How well each model did on the highest held-out points |
| `scheme_for(n_points, scheme="auto")` | Which scheme runs, and with what settings |
| `holdout_splits`, `leave_one_out_splits`, `stratified_train_index` | The partitions themselves |

Small samples get leave-one-out; larger ones get stage-stratified holdout splits.

## `exact_loo.py` — repair `elpd_loo` where PSIS broke down

PSIS-LOO flags high-influence measurements with Pareto k > 0.7. These refit
without those points and score them exactly.

| Name | Purpose |
| --- | --- |
| `reloo(rating, k_threshold=None, ...)` | Refit the flagged measurements and rescore them |
| `refine(fit, sample, entry, ...)` | Repair a `FitResult`'s `elpd_loo` in place |
| `refit_budget(n_points, ...)` | How many refits a sample is allowed |
| `BdrcRefitWrapper`, `RatingCurveRefitWrapper` | The per-family refit machinery ArviZ calls |
| `holdout_log_likelihood(fitted, stage_m, discharge_cms)` | Log density of a point a bdrc fit never saw |
| `ordered_frame(sample, family)` | Measurements in the order the log-likelihood is indexed by |

## `metrics.py` — the Bayesian scores

| Name | Purpose |
| --- | --- |
| `elpd_from_idata(idata)` | PSIS-LOO and WAIC from an InferenceData |
| `elpd_from_log_likelihood(log_likelihood)` | The same from a raw pointwise matrix |
| `relative_efficiency(log_likelihood)` | Relative MCMC efficiency (`reff`, needed for loglik-only idata) |
| `ensure_log_likelihood(rating)` | Attach a pointwise log-likelihood group to a ratingcurve fit |
| `pareto_k_per_point(fit)` | Per-measurement Pareto k |
| `comparison_rows(site)`, `metric_hover(column)` | Table rows and their tooltips |
| `unavailable(note)`, `refused_for_advi(fit)` | NaN score dicts with a reason (ADVI fits are not scored, by policy) |

## `diagnostics.py` — did the sampler converge

| Name | Purpose |
| --- | --- |
| `convergence(fit_or_model)` | R-hat and ESS per parameter, with a pass/fail flag |
| `convergence_report(fit_or_model)` | A one-line verdict |
| `convergence_table(rating_set)` | Verdicts for every model in a comparison |
| `posterior_summary(fit_or_model)` | ArviZ posterior summary |
| `plot_convergence`, `plot_effective_sample_size`, `plot_trace`, `plot_pareto_k` | The diagnostic figures |
