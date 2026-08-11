# limnotech_rating_curves.ratings

The estimators: fit a rating curve, predict from it, compare several.

The interface follows the shape of scikit-learn, constructing a model and calling
`fit` and `predict`, with the additions a Bayesian rating requires: an interval, a
fitted curve, model-comparison scores and convergence diagnostics.

    import limnotech_rating_curves as lrc

    rating = lrc.fit_rating(measurements)        # one model, fitted
    rating.predict(5.0)                          # discharge at 5 ft
    rating.interval(5.0)                         # 95% credible interval
    rating.metrics                               # in-sample and predictive scores
    rating.equation()                            # 'Q = 182.16 (h - 5.12323)^1.46971'
    rating.coefficients                          # the same numbers, as a dict
    rating.plot()

    comparison = lrc.compare(measurements, models="all")
    comparison.metrics                           # ranked by predictive score
    comparison.ranking()                         # with the differences and their errors
    comparison.equation()                        # which models can be written down
    comparison.best.predict(5.0)

`equation()` is the interpretable form, and it exists for the families that have one:
the power laws and the two least-squares forms. A spline and a bdrc model return an
empty string, because a spline's shape is a basis matrix with its weights and bdrc's
exponent varies continuously with stage; read those through `curve()`. The coefficients
are taken at the posterior mean, so the written curve tracks `curve()`'s
`discharge_median_cfs` rather than the posterior mean `predict()` returns, and near a
segmented power law's breakpoint the two part company - see
`BayesianRating.coefficients`.

`measurements` may be a DataFrame with a stage and a discharge column, two arrays, a
`Sample` from one of the loaders, or a pagaia
`Station` (see `limnotech_rating_curves.data.pagaia`). Stage given as a
water-surface elevation rather than gage height is converted on the way in by
`stage_datum=`; see `limnotech_rating_curves.data.datum`.

## The three model families
`PowerLaw`
    $Q = C(h-e)^{\beta}$, the default. It follows the physics of open-channel
    flow over a control, so its upward extrapolation is defensible. `segments > 1`
    allows the control to change with stage.
`Spline`
    A natural cubic spline in log space. It is more flexible than the power law and
    will always fit the measurements at least as well, which is why its in-sample fit
    is not evidence of quality: nothing constrains it above the highest measurement,
    so it suits interpolation rather than extrapolation.
`Bdrc`
    The bdrc generalized power law, in which the exponent varies smoothly with stage.
    It is intermediate between the other two: more flexible than a fixed exponent,
    still a power law at every stage, and with priors chosen to behave on very short
    records.
