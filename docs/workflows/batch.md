# limnotech_rating_curves.workflows.batch

Fitting, scoring and mapping many sites in one run.

This is the batch layer over the library. For a single site or for arbitrary data, use
the library directly (`limnotech_rating_curves.fit_rating`). Use this module when
the question spans sites, since its primary output is the interactive map.

`run` performs the following steps, which are logged under `-v`:

1. Assemble a sample per selected site, from whichever sources apply.
2. Fit every selected model on all of each sample, with NUTS by default so that the
   predictive score is trustworthy.
3. Attach the published USGS rating as a reference, where the site has one.
4. Score every curve: in-sample fit, plus the Bayesian comparison (PSIS-LOO, WAIC and
   Pareto-k).
5. Cross-validate, if requested, using leave-one-out or the holdout sweep, chosen per
   site by sample size.
6. Write the interactive map, the results tables and, optionally, the static figures
   and the exported posteriors.

The MAGL network is the built-in site source (`magl_rating_curves`), because that
is the network this package was developed against. `run` also accepts an explicit
list of samples, so any set of sites can go through the same pipeline:

    report = batch.run(samples=[my_sample_a, my_sample_b], models="all")
