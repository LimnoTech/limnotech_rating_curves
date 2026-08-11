# limnotech_rating_curves.exports

Writing a fitted rating to disk, and reading it back.

A saved rating is two files that travel together:

```
===============================  ================================================
``<site>__<model>.rating.json``  the manifest: what was fitted, how it was
                                 sampled, the measurements, the curve, the scores
``<site>__<model>.nc``           the posterior draws, as ArviZ NetCDF
===============================  ================================================
```

The manifest can be read and diffed without PyMC installed, and it carries the fitted
curve as a table, so `load_rating` returns an object that predicts immediately.
The posterior is the part that cannot be reconstructed: any new score, any different
credible level and any re-plot on a new stage grid is computed from the draws. It is
therefore written on every save, and `load_rating` reports when it is absent.

    lrc.save_rating(rating, "out/")                        # -> both paths
    again = lrc.load_rating("out/04176356__power_law.rating.json")
    again.predict(5.0)                                     # from the stored curve
    again.posterior()                                      # the draws, on demand

A manifest is not a way to rebuild the fitted model object. A ratingcurve posterior can
be reloaded into a live model by the ratingcurve package itself; a bdrc fit cannot,
because its curve comes from a closed-form kriging step that is not stored.
`SavedRating` therefore predicts by interpolating the stored curve, which is
what the dashboard and the map do with a fitted rating in any case, and it reports that
this is what it does.

## Moving a fit to another machine
A directory of these pairs is the entire interface. Fit on one machine, copy the
directory, read it on another:

    rating-curves export output/fitted_curves --to ratings_to_ship --csv
    # copy ratings_to_ship/ across
    rating-curves inspect ratings_to_ship/workbook__SBR-09__quadratic.rating.json

The manifest records a SHA-256 of the posterior written beside it
(`posterior_sha256`), so a truncated or mismatched copy is caught by
`SavedRating.verify` rather than read as the fit's own draws. Reading a manifest
requires only the standard library, numpy and pandas. Reading a posterior requires
ArviZ, and only when `SavedRating.posterior` is called.
