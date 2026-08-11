# limnotech_rating_curves

Fit stage-discharge rating curves, compare model families, cross-validate, save.
Units are feet (stage) and cfs (discharge) everywhere in the public API.

```python
import limnotech_rating_curves as lrc

sample  = lrc.gage_sample("04166500")     # USGS field measurements
rating  = lrc.fit_rating(sample)          # one model
ratings = lrc.compare(sample)             # several models, ranked
ratings.metrics()
ratings.best.plot()
```

## Where things live

| Folder | What it holds |
| --- | --- |
| [`data/`](data/) | Loading measurements: USGS, pagaia, MAGL workbooks; stage datums; stage of zero flow |
| [`models/`](models/) | The estimator backends and the model catalog (keys, labels, colors) |
| [`models/bdrc/`](models/bdrc/) | The bdrc generalized power law, ported from R |
| [`evaluate/`](evaluate/) | Cross-validation, exact leave-one-out, convergence diagnostics |
| [`view/`](view/) | Matplotlib figures and the interactive Plotly map |
| [`workflows/`](workflows/) | Many sites at once: parallel sweeps and the batch report |
| [`support/`](support/) | Disk cache and logging setup |

## Top-level modules

**`ratings.py`** — the estimator classes and the two entry points.

| Name | Purpose |
| --- | --- |
| `fit_rating(data, model="power_law")` | Fit one model, return a `RatingModel` |
| `compare(data, models=None)` | Fit several, return a `RatingSet` (alias: `lrc.fit`) |
| `rating_model(name)`, `resolve_models(models)` | Build unfitted estimators from names |
| `PowerLaw`, `Spline`, `Bdrc`, `Quadratic`, `Exponential` | The estimator classes |
| `RatingModel` | Shared surface: `.predict`, `.interval`, `.curve`, `.metrics`, `.equation`, `.summary`, `.diagnostics`, `.pareto_k`, `.zero_flow`, `.cross_validate`, `.save_posterior`, `.plot`, `.plot_residuals`, `.plot_check` |
| `RatingSet` | `.best`, `.metrics`, `.ranking`, `.curve`, `.predict`, `.interval`, `.equation`, `.summary`, `.diagnostics`, `.cross_validate`, `.plot`, `.plot_residuals`, `.plot_ranking` |

`Quadratic` and `Exponential` add spreadsheet-facing extras: `.coefficients`,
`.r_squared`, `.r_squared_log`, `.effective_range`, `.turning_point`.

**`core.py`** — the plain records every backend produces.

| Name | Purpose |
| --- | --- |
| `Sample` | Paired measurements plus provenance. `Sample.of(...)` builds one from anything; `.fit`, `.compare`, `.cross_validate`, `.to_frame`, `.support`, `.stage_range`, `.discharge_range`, `.stage_grid`, `.time_range`, `.plot_record` |
| `FitResult` | One model fitted to one sample; `.ok` |
| `Metrics`, `fit_metrics(observed, predicted)` | Goodness-of-fit scores |
| `FoldCurve` | One cross-validation fold's curve; `.predict_test` |
| `SiteRating` | Every fit at one site; `.successful_fits`, `.external_curves` |
| `ExternalCurve` | A curve this package did not fit, for comparison |
| `padded_stage_grid(stage)` | The grid curves are tabulated and drawn on |

**`exports.py`** — saving and reloading fits.

| Name | Purpose |
| --- | --- |
| `save_rating`, `save_fit`, `save_ratings`, `save_site` | Write manifests (JSON) plus posteriors (NetCDF) |
| `load_rating(path)`, `load_ratings(directory)` | Read them back as `SavedRating` |
| `SavedRating` | `.predict`, `.interval`, `.curve`, `.measurements`, `.metrics`, `.convergence`, `.parameters`, `.sampler`, `.posterior`, `.verify`, `.describe`, `.plot` |
| `curve_table`, `manifest_index` | Many saved ratings as one table |
| `export_directory(source, destination)` | Bundle a directory for another machine |
| `rating_manifest`, `orphan_posteriors`, `canonical_model_key` | Lower-level helpers |

**`cli.py` / `__main__.py`** — the `rating-curves` command (`fit`, `export`,
`inspect`/`show`, `list`). `main(argv=None)` is the entry point.

**`settings.py`** — every default in one place: paths, seed, grid size, sampler
settings. `apply_numerical_workarounds()` runs at import to set the environment
variables PyMC and MKL need before numpy loads.

## Re-exported at the top level

Everything most callers need is on `lrc` directly: `fit_rating`, `compare`, `fit`,
`rating_model`, the five estimator classes, `RatingSet`, `fit_many`,
`RatingCollection`, `report`, `Sample`, `Metrics`, `FitResult`, `FoldCurve`,
`SiteRating`, `ExternalCurve`, `gage_sample`, `station_sample`, `sensor_sample`,
`published_rating`, `rating_curve_sheet`, `flow_sheet_sample`, `StageDatum`,
`to_gage_height`, `distance_to_stage`, `johnson_offset`, `estimate_zero_flow`,
`ZeroFlowEstimate`, `cross_validate`, `CrossValidation`, `convergence`,
`convergence_report`, and the whole `exports` set above.
