# limnotech_rating_curves.models.bdrc.compiled

Compiled graphs, cached: eight functions serve every fit in a process.

A compiled `CompiledModel` is keyed on `(model, c known or inferred)` and
nothing else. The dataset lives in shared variables (see
`graphs`), so one compiled function evaluates
a 27-measurement gage and an 18-measurement fold without recompiling.
`compiled_for` returns one, building it on first use, binding the dataset, and
returning the same object on later calls.

Compilation takes roughly 40 seconds and dominates a bdrc fit, while the arithmetic
takes milliseconds. Without this cache every distinct sample size pays that cost again,
so a batch run over 20 sites, 4 variants and 13 folds paid it about 1000 times. A
process now pays it at most eight times.
