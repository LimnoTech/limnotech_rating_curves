# limnotech_rating_curves.workflows.__init__

Many sites and many models in one call.

    from limnotech_rating_curves.workflows import fit_many, run

    fit_many(samples, models="all")   # fit in parallel, one artifact pair per fit
    run()                             # the MAGL pipeline: fit, score, map, tables

`fit_many` is the general entry point: it takes samples and model keys and fans
out across processes. `run` is the MAGL-specific batch report, built on the same
catalog.
