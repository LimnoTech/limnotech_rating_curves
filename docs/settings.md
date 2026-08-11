# limnotech_rating_curves.settings

Package-wide defaults: seeds, sampler budgets, grid sizes, file locations.

Every tunable number the package uses lives here rather than being repeated in
the module that happens to need it, so a caller can read one file to see what a
default run does and override any of it in one place.

## Environment variables read at import
`LRC_DATA_DIR`
    Directory holding the local data files (the MAGL survey CSVs, the
    spreadsheet pickle, the manual-discharge CSVs). Defaults to `data/` in the
    surrounding repository.
`LRC_CACHE_DIR`
    Directory for the pickle-through fetch cache. Defaults to `cache/` in the
    surrounding repository.
`LRC_OUTPUT_DIR`
    Directory for generated maps, figures and tables. Defaults to `output/` in
    the distribution root.
