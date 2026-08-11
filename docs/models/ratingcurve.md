# limnotech_rating_curves.models.ratingcurve

Fitting backend: the USGS `ratingcurve` package (power law and spline).

    https://github.com/DOI-USGS/ratingcurve

`ratingcurve` estimates a rating as a Bayesian model, so every fit carries posterior
uncertainty: the fitted table gives a mean, a median and a geometric standard error
rather than a single interpolated value. This module wraps the package twice over.
`BayesianRating` is a uniform handle on one of its models, and `fit` turns
a sample into this package's own
`FitResult`.

Two of the package's estimator families fit reliably in this environment and are the
ones exposed here:

```
===============  ===============================================================
``power_law``    multi-segment power law, Heaviside parameterization. One
                 segment is the plain :math:`Q = C(h-e)^{\beta}`; two or more
                 make it a segmented rating with a breakpoint between segments.
``spline``       natural cubic spline in log space, knot count set by ``knots``.
===============  ===============================================================
```

The package's experimental parameterizations (`broken`, `iso`, `lecoz`,
`reitan`, `smoothbroken`) are reachable through `BayesianRating` but are not
registered in the model catalog: on pymc 5.28 and numpy 2.4 they fail with dimension,
NaN or attribute errors.

## The zero-flow prior
The power-law family accepts a prior on its breakpoints, the first of which is the
stage of zero flow. That is the parameter a small sample identifies least well, so
`breakpoint_prior` centers it on an estimate from
`limnotech_rating_curves.data.zero_flow`, using Johnson's three-point method by
default or a value supplied by the caller. The spline has no such parameter and ignores
it.
