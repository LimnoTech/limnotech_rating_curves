# workflows

Many sites at once. `fit_many` is the general parallel sweep over any samples;
`run` is the MAGL-network report that fits, scores, cross-validates and writes
the map in one call.

```python
import limnotech_rating_curves as lrc

collection = lrc.fit_many({"04166500": sample}, directory="output/sweep")
collection.best_per_site()

report = lrc.report(sources=["magl"])   # workflows.batch.run
```

## `parallel.py` — the sweep

| Name | Purpose |
| --- | --- |
| `fit_many(samples, models=None, directory=None, workers=None, ...)` | Fit every model to every sample in parallel, saving as it goes |
| `RatingCollection` | Every rating a sweep produced, indexed by site and model |
| `RatingCollection.from_directory(directory)` | Rebuild a collection from a previous sweep's files |
| `collect_samples(samples)` | Normalize whatever was passed into `{site_id: Sample}` |
| `default_workers(n_tasks=None)` | One worker per core bar one, capped by memory |

`RatingCollection` surface: `.site(site)`, `.get(site, model)`, `.ratings`,
`.sites`, `.model_keys`, `.metrics()`, `.failures()`, `.best(site)`,
`.best_per_site()`, `.curves()`.

## `batch.py` — the end-to-end report

`run(models=None, samples=None, sites=None, sources=None, ...)` is re-exported as
`lrc.report(...)`. It returns a `Report` with `.cv_table` and the assembled sites.

The stages it runs, each usable on its own:

| Name | Step |
| --- | --- |
| `assemble_magl_sites(sites=None, sources=None)` | Build an unfitted `SiteRating` per selected site |
| `sites_from_samples(samples)` | Wrap plain samples so they can go through `run` instead |
| `fit_sites(sites, entries)` | Fit every model on all of each site's sample, in place |
| `attach_references(sites)` | Attach each USGS gage's published rating as a comparison curve |
| `score_sites(sites)` | Compute the Bayesian comparison scores |
| `cross_validate_sites(sites, models=None)` | Cross-validate every site with enough measurements |
| `results_frame(sites)` | One row per curve per site, including the fits that failed |
