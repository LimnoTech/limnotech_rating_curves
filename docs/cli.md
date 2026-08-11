# limnotech_rating_curves.cli

Command line for the batch pipeline and for the ratings it writes.

Installed as `rating-curves`; also runnable as
`python -m limnotech_rating_curves`.

Fitting, the default with no subcommand:

```
rating-curves --list-sites                 the available sites
rating-curves -v                           every site, curated models, then the map
rating-curves --site SBR-09 SR --cv        one sensor and one cluster, with the
                                           cross-validation pane
rating-curves --models bdrc --png          the four bdrc variants, plus the static
                                           figures
rating-curves --data measurements.csv      fit a CSV of measurements
```

Working with what a run saved:

```
rating-curves list output/fitted_curves            the contents of a directory
rating-curves inspect output/fitted_curves/*.rating.json    one rating in full
rating-curves export output/fitted_curves --to ship/        a portable copy + CSVs
```

`list` and `inspect` read the manifests only, so they run on a machine with
neither PyMC nor a compiler. Every subcommand is a thin wrapper over
`limnotech_rating_curves.workflows.batch.run`,
`limnotech_rating_curves.compare` and `limnotech_rating_curves.exports`,
each of which can be called directly.
