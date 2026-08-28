# bdrc

A port of the R `bdrc` package: Bayesian generalized power-law rating curves.
Four models — `plm0`, `plm` (power law), `gplm0`, `gplm` (exponent varies with
stage); the ones without `0` also let the error variance vary with stage.

**Units here are SI** (`discharge` in m³/s, `stage` in m), unlike the rest of the
package. `fit_predict` is the ft/cfs wrapper.

```python
from limnotech_rating_curves.models import bdrc

fit = bdrc.fit(discharge_cms, stage_m, model="gplm0")
fit.predict(newdata)
```

## Fitting

| Name | Purpose |
| --- | --- |
| `fit(discharge, stage, model="gplm0", method="nuts", ...)` | Fit a rating; SI units |
| `fit_predict(stage_ft, discharge_cfs, grid_stage_ft, ...)` | Fit and predict in ft/cfs |
| `BdrcFit` | The fit object; field names follow R's so the two can be compared. `.predict(newdata)` gives discharge quantiles |
| `MODELS` | The four model names |

## The compile cache (`compiled.py`)

One compiled pytensor graph per `(model, known_c)` pair, rebound to each new
dataset rather than recompiled.

| Name | Purpose |
| --- | --- |
| `compiled_for(C)` | The compiled model for this dataset, bound and ready |
| `CompiledModel` | `.model`, `.hessian`, `.logp_graph(theta)`, `.bind(C)`, `.design_at(theta)` |
| `cached_models()`, `clear_cache()` | Inspect and free the cache |
| `posterior_mode(compiled)` | The marginal posterior mode, by L-BFGS-B |
| `curvature_at(compiled, theta)` | Minus the Hessian — the precision matrix |
| `c_upper_bound(y_obs, stage, forcepoint)` | Bound on `c` when nothing was measured near zero flow |
| `finalize_grid(compiled, h_max)` | Attach the prediction grid once the mode is known |

## Model structure

| Module | Contents |
| --- | --- |
| `defaults.py` | `MODELS`, `theta_names`, `varying_exponent`, `varying_variance`, `prior_covariance`, `target_accept_for`. The numbers themselves - sampler budgets, priors, the spline basis size - are `settings.BDRC_*`, imported here under the short names the paper uses |
| `design.py` | `components(model, y_obs, stage, ...)` → `Components`; `b_splines`, `unique_stage_matrix`, `prediction_stages`, `matern52_correlation`, `distance_matrix` |
| `graphs.py` | `SharedDataset` (the shared variables a graph binds to), `marginal_logp_graph`, `design_graph`, `log_prior_graph` |
| `sampling.py` | `sample_hyperparameters(compiled, method=...)` — NUTS or ADVI |
| `posterior.py` | `assemble(...)` builds the fit from hyperparameter draws; `mcmc_summary`, `latent_draw`, `unobserved_draw`, `param_summary`, `sampler_diagnostics`, `stable_cholesky` |
| `criteria.py` | `pointwise_log_likelihood`, `waic_from_log_likelihood`, `add_information_criteria`, `deviance_at_median`, `attach_log_likelihood` |
| `differences.py` | `DIFFERENCES_FROM_R` — where this port deliberately departs from `bdrc` |

Importing this package sets `MKL_THREADING_LAYER=SEQUENTIAL` and
`KMP_DUPLICATE_LIB_OK=TRUE`; MKL's threaded BLAS segfaults on this stack and
sequential costs nothing at these problem sizes.
