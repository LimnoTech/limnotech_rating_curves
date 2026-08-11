# limnotech_rating_curves.workflows.parallel

Fitting many sites and many models across processes.

    results = lrc.fit_many(samples, models=["power_law", "bdrc_gplm0"])

    results.metrics                     # one row per (site, model), best first
    results["04176356", "power_law"]    # one saved rating
    results["04176356"]                 # every model fitted for that site
    results.best("04176356")            # its best by predictive score
    results.failures                    # what did not fit, and why

## Accepted forms of `samples`

```
=================================  ============================================
``{"04176356": frame, ...}``       a dict of measurement frames or Samples,
                                   keyed by the caller's chosen names
``[sample, sample, ...]``          Samples, keyed by their own ``site_id``
``sample`` or a single frame       one site
an ``fb_pagaia`` ``Network``       every station in it, paired with the
                                   discharge passed in ``discharge=``
=================================  ============================================
```

## Why the result is built from files
A fitted rating cannot cross a process boundary: the spline carries a patsy
`DesignInfo` that does not pickle, and neither a PyMC model nor an `InferenceData`
survives being sent between processes. Each worker therefore writes its own manifest
and posterior (see `limnotech_rating_curves.exports`) and returns only a path and
a summary. The collection is built from those files, which makes it identical to what
reloading a previous run from disk produces: one shape regardless of when the fits
happened.

This is also why `workers=1` behaves the same way. A single-process run writes the
same files, so a script does not change shape when it is widened.
