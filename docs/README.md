# limnotech_rating_curves - module reference

One page per module, holding the text that used to open its source file.
The API reference proper is the docstrings on the classes and functions
themselves; these pages are the why, not the signatures.

## The package

| page | what it covers |
| --- | --- |
| [`__init__`](__init__.md) | Bayesian stage-discharge rating curves |
| [`ratings`](ratings.md) | The estimators: fit a rating curve, predict from it, compare several |
| [`core`](core.md) | The data a rating is fitted to, and the records a fit produces |
| [`exports`](exports.md) | Writing a fitted rating to disk, and reading it back |
| [`settings`](settings.md) | Package-wide defaults: seeds, sampler budgets, grid sizes, file locations |
| [`cli`](cli.md) | Command line for the batch pipeline and for the ratings it writes |
| [`__main__`](__main__.md) | Entry point for `python -m limnotech_rating_curves` |

## `data` - where measurements come from

| page | what it covers |
| --- | --- |
| [`data/__init__`](data/__init__.md) | Measurement sources, and the vertical reference each one reports |
| [`data/datum`](data/datum.md) | Converting reported stage onto the vertical reference a rating requires |
| [`data/magl`](data/magl.md) | Rating curves for the LimnoTech MAGL sensor network |
| [`data/magl_workbook`](data/magl_workbook.md) | Reading the rating that each MAGL field workbook already contains |
| [`data/pagaia`](data/pagaia.md) | Fitting a rating to a pagaia station |
| [`data/usgs`](data/usgs.md) | USGS data: field measurements, gage datums, and published rating curves |
| [`data/zero_flow`](data/zero_flow.md) | Estimating the stage of zero flow by Johnson's three-point method |

## `models` - the fitting backends and the catalog

| page | what it covers |
| --- | --- |
| [`models/__init__`](models/__init__.md) | The model catalog, and the two packages that fit the models |
| [`models/bdrc/__init__`](models/bdrc/__init__.md) | Bayesian discharge rating curves: a native PyMC port of the R `bdrc` package |
| [`models/bdrc/compiled`](models/bdrc/compiled.md) | Compiled graphs, cached: eight functions serve every fit in a process |
| [`models/bdrc/criteria`](models/bdrc/criteria.md) | Information criteria: the pointwise log-likelihood, WAIC and DIC |
| [`models/bdrc/defaults`](models/bdrc/defaults.md) | bdrc's own constants: the priors from the paper, and the sampling budget |
| [`models/bdrc/design`](models/bdrc/design.md) | Everything about one model and dataset that does not depend on theta |
| [`models/bdrc/differences`](models/bdrc/differences.md) | Where this port's numbers can differ from the R bdrc package, and why |
| [`models/bdrc/fitting`](models/bdrc/fitting.md) | The two entry points: `fit` in SI units, `fit_predict` in ft and cfs |
| [`models/bdrc/graphs`](models/bdrc/graphs.md) | The marginal log-density as one pytensor graph, over shared dataset inputs |
| [`models/bdrc/posterior`](models/bdrc/posterior.md) | The closed-form half of the fit: latent draws, predictions, and the fit object |
| [`models/bdrc/sampling`](models/bdrc/sampling.md) | Walking the marginal posterior: NUTS, or the fast variational fit |
| [`models/catalog`](models/catalog.md) | The catalog of rating models: one entry per model, behind one interface |
| [`models/exponential`](models/exponential.md) | Exponential ratings: the third form the field spreadsheets use |
| [`models/polynomial`](models/polynomial.md) | Least-squares polynomial ratings: the spreadsheet curve, in code |
| [`models/ratingcurve`](models/ratingcurve.md) | Fitting backend: the USGS `ratingcurve` package (power law and spline) |

## `evaluate` - scoring a fitted rating

| page | what it covers |
| --- | --- |
| [`evaluate/__init__`](evaluate/__init__.md) | Evaluating a fitted rating: cross-validation and sampler diagnostics |
| [`evaluate/crossval`](evaluate/crossval.md) | Cross-validation: scoring a rating on measurements it was not fitted to |
| [`evaluate/diagnostics`](evaluate/diagnostics.md) | Convergence diagnostics for a fitted rating |
| [`evaluate/exact_loo`](evaluate/exact_loo.md) | Exact leave-one-out for the measurements PSIS could not handle |
| [`evaluate/metrics`](evaluate/metrics.md) | Bayesian model comparison: placing every rating on one predictive score |

## `workflows` - many sites at once

| page | what it covers |
| --- | --- |
| [`workflows/__init__`](workflows/__init__.md) | Many sites and many models in one call |
| [`workflows/batch`](workflows/batch.md) | Fitting, scoring and mapping many sites in one run |
| [`workflows/parallel`](workflows/parallel.md) | Fitting many sites and many models across processes |

## `view` - figures and the map

| page | what it covers |
| --- | --- |
| [`view/__init__`](view/__init__.md) | Viewing a rating: static figures and the interactive map |
| [`view/mapview`](view/mapview.md) | The interactive map: every site, with an inspector for the site under the cursor |
| [`view/plots`](view/plots.md) | Static rating figures, as matplotlib PNGs |

## `support` - logging and the fetch cache

| page | what it covers |
| --- | --- |
| [`support/__init__`](support/__init__.md) | Internal support code: the fetch cache and logging setup |
| [`support/cache`](support/cache.md) | Pickle-through cache for the slow fetches (USGS web services, pagaia) |
| [`support/logging_setup`](support/logging_setup.md) | Logging configuration for the package |
