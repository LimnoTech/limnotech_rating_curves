# limnotech_rating_curves.evaluate.exact_loo

Exact leave-one-out for the measurements PSIS could not handle.

`elpd_loo` in the metrics table is PSIS-LOO: it reweights one fitted posterior to
approximate what the model would have predicted without each measurement in turn. The
Pareto-k beside it says where that reweighting failed. The standard remedy is not to
abandon the approximation but to repair it - keep the PSIS value at every measurement
whose k is acceptable, and refit the model without the handful whose k is not, scoring
those exactly.

That procedure is `arviz.reloo`, and this module supplies the two sampling
wrappers it needs, one per Bayesian family. The polynomial and exponential ratings have
no posterior and are not eligible.

Measured on this package's own samples, the repair is worth roughly what it costs. On
USGS gage 04176356 (26 measurements, `bdrc_gplm0`) exactly one measurement exceeded
k = 0.7, and that one carried 1.69 of the 1.79 nats by which PSIS disagreed with a full
26-refit exact leave-one-out - so one refit recovered 94% of the error for 4% of the
compute. Every measurement below the threshold agreed to within 0.04 nats.

## Two ways to ask for it
Once, for one fit:

```
rating = lrc.fit_rating(measurements, model="bdrc_gplm0")
rating.metrics                              # PSIS, fast
refined = lrc.evaluate.reloo(rating)        # refits the flagged measurements
refined["elpd_loo"], refined["n_refits"]
```

Or everywhere, with one flag:

```
lrc.settings.RELOO = True
```

which makes `rating.metrics` and the batch report return the repaired `elpd_loo`,
so every table, map and manifest downstream carries it. It is off by default because a
refit is a full NUTS chain, about twenty seconds on these models, and `rating.metrics`
is a property that callers expect to return promptly.

## What it does not do
It does not clear the warning that prompted it. A Pareto-k above the threshold
means the posterior leans heavily on one measurement, which in a rating usually means
the highest flow, or an outlier. Refitting gives the honest predictive score; it does
not make the curve less dependent on that point. The returned scores therefore carry
the **original** Pareto-k values, not the zeros `arviz.reloo` writes over the
refitted entries.

## The budget
The number of refits is capped at `settings.RELOO_MAX_FRACTION` of the sample, at
least one, because the cost is unbounded otherwise: a badly conditioned sample can flag
half its measurements and turn a repair into a full exact leave-one-out. Measurements
are refitted worst-k first, so the budget is spent where it buys most, and any that
remain approximate are counted in `n_over_budget` and named in the note. A result
with `n_over_budget` above zero is partly repaired, not repaired.

## Keeping the two families in one space
`arviz.reloo` substitutes exact values into the pointwise vector PSIS produced, so
an exact value has to arrive in the space that vector is already in.

bdrc's log-likelihood is a density on raw log discharge and needs no conversion. The
ratingcurve family fits on standardized log discharge, and **each refit standardizes
differently**, because `LogZTransform` is built from the training discharges. So a
fold's log-likelihood is a density of that fold's `z`, and reaching the all-data
`z` the PSIS terms live in takes `log(all-data std) - log(fold std)` per draw. The
constant `-n log(std)` that `limnotech_rating_curves.evaluate.metrics` applies to
the total is then still the right one. Skipping the per-fold half of that correction
silently biases every refitted measurement.
