# limnotech_rating_curves.__init__

Bayesian stage-discharge rating curves.

Fit a rating to paired stage and discharge measurements, compare model families on a
common predictive score, cross-validate them, fit many sites in parallel, and save the
result in a form that can be read back.

    import limnotech_rating_curves as lrc

    rating = lrc.fit_rating(measurements)          # a DataFrame, or two arrays
    rating.predict(5.0)                            # discharge at 5 ft
    rating.interval(5.0)                           # 95% credible interval
    rating.metrics                                 # in-sample and predictive scores
    rating.plot()

    comparison = lrc.compare(measurements, models="all")
    comparison.metrics                             # ranked by predictive score
    comparison.best.predict(5.0)

    lrc.cross_validate(measurements)               # refit on subsets and score the
                                                   # held-out measurements

## Many sites at once
    results = lrc.fit_many(samples, models=["power_law", "bdrc_gplm0"])
    results.metrics                                # one row per (site, model)
    results["04176356", "power_law"].predict(5.0)
    results.best("04176356")

Each fit is written to disk as it completes, because a fitted model cannot cross a
process boundary. See `parallel`.

## Saving and reloading
    lrc.save_rating(rating, "out/")                # manifest and posterior, as a pair
    again = lrc.load_rating("out/04176356__power_law.rating.json")
    again.predict(5.0)                             # from the stored curve
    again.posterior()                              # the draws, read on demand

The manifest is JSON: model, sampler, seed, measurements, curve, scores and R-hat. The
`.nc` file beside it holds the posterior draws. See
`exports`.

## Loading data

```
==========================================  ===================================
``lrc.gage_sample("04176356")``             a USGS gage's field measurements
``lrc.sensor_sample("SBR-09")``             a MAGL sensor
``lrc.station_sample(station, discharge)``  a pagaia station and its discharge
==========================================  ===================================
```

Stage reported on another vertical reference is converted by `stage_datum=`, which
accepts the elevation of gage height zero, `"lowest"`, or a timeseries of reference
elevations. See `limnotech_rating_curves.data.datum`.

## Package layout

```
=================  ==========================================================
``lrc.data``       measurement sources, and the vertical datum they are on
``lrc.evaluate``   cross-validation, convergence diagnostics, ELPD
``lrc.view``       static figures and the interactive map
``lrc.models``     the model catalog and the two fitting backends
``lrc.workflows``  many-site sweeps and the MAGL batch report
=================  ==========================================================
```

At the top level: `core` (`Sample` and the fit records), `ratings` (the
estimators), `exports` (save and load) and `settings` (tunable defaults).
