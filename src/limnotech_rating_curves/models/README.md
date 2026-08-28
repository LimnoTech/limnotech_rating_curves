# models

The estimator backends, plus the catalog that gives every model its key, label,
color and minimum point count. Callers normally go through
[`ratings.py`](../ratings.py); reach in here to add a model or to look up how one
is drawn.

## `catalog.py` — the registry

`MODELS` is the ordered list of every entry. Model keys:

| Key | Model | Backend |
| --- | --- | --- |
| `power_law` | power law, 1 segment | ratingcurve |
| `power_law_2seg`, `power_law_3seg` | segmented power law | ratingcurve |
| `spline` | natural cubic spline, log space | ratingcurve |
| `bdrc_gplm0`, `bdrc_gplm`, `bdrc_plm0`, `bdrc_plm` | bdrc generalized power law | bdrc |
| `linear`, `quadratic` | least-squares polynomial | polynomial |
| `exponential` | log-linear exponential | exponential |

The last three are the forms the MAGL field workbooks use, **refitted here** —
not the workbook's own typed coefficients. Groups `select()` understands:
`all`, `default`, `bdrc`, `ratingcurve`, `power_law_family`, `polynomial`,
`exponential_family`, and the spreadsheet-form group.

| Name | Purpose |
| --- | --- |
| `select(spec)` | Resolve a `models=` argument to catalog entries, in catalog order |
| `get(key)`, `all_keys()` | Look up one entry, or list them |
| `label(key)`, `short_label(key)`, `color(key)`, `dash(key)`, `role(key)` | Presentation, consistent everywhere |
| `ModelEntry` | The common surface: `.fit(sample)`, `.fit_fold(train, test, grid, split)`, `.bayes_metrics(fit)`, `.save_posterior(...)` |
| `RatingCurveEntry`, `BdrcEntry`, `PolynomialEntry`, `ExponentialEntry` | One subclass per backend |
| `save_posterior(fit, directory, sample_id)` | Write a posterior to ArviZ NetCDF |

Also exported: `MODEL_KEYS`, `SPREADSHEET_FORM_KEYS`, `GROUPS`,
`RATINGCURVE_KEYS`, `BDRC_KEYS`, `FORM_KEYS`. Which models `select(None)` returns is
`settings.DEFAULT_MODEL_KEYS`.

## `ratingcurve.py` — the PyMC `ratingcurve` package

| Name | Purpose |
| --- | --- |
| `fit(sample, ...)` | Fit one model, never raises; returns a `FitResult` |
| `BayesianRating` | `.fit`, `.predict`, `.table`, `.equation`, `.summary`, `.residuals`, `.save`, `.fitted` |
| `available_algorithms()` | Which estimator families this wrapper can build |
| `breakpoint_prior(stage, discharge, segments)` | The normal prior on power-law breakpoints |
| `adapt_config(...)` | Whether `n` measurements can support a model, and how |

## `bdrc/` — see [`bdrc/README.md`](bdrc/)

## `polynomial.py` — least squares in stage

`fit_polynomial(stage, discharge, degree=2)` → `PolynomialRating` with
`.predict`, `.interval` (prediction interval), `.table`,
`.display_range`. `fit(sample, key=..., label=...)` wraps it as a `FitResult`.

## `exponential.py` — Excel's exponential trendline

`fit_exponential(stage, discharge)` fits `Q = A exp(B h)` by least squares on log
discharge → `ExponentialRating` with the same surface as above (its `.interval`
is computed in log space). `fit(sample, ...)` wraps it as a `FitResult`.
