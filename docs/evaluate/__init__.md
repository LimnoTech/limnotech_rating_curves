# limnotech_rating_curves.evaluate.__init__

Evaluating a fitted rating: cross-validation and sampler diagnostics.

    from limnotech_rating_curves.evaluate import cross_validate, convergence

    cross_validate(sample)          # refit on subsets, score the held-out points
    convergence(rating)             # R-hat and effective sample size per parameter
    convergence_report(rating)      # the same result as one sentence

`metrics` holds the predictive score itself
(PSIS-LOO and WAIC, and the correction that makes the two model families comparable).
It is called by `rating.metrics` and is rarely needed directly.
