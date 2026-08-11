# limnotech_rating_curves.models.catalog

The catalog of rating models: one entry per model, behind one interface.

Two different packages fit the ratings in this library: the USGS `ratingcurve` power
law and spline, and the `bdrc` Bayesian generalized power law, a native PyMC port
with no R dependency. Everything downstream, including the estimators, the
cross-validation, the map and the metrics table, has to treat them identically, so each
model is registered here as a `ModelEntry` exposing the same four operations:

```
===========================  ==================================================
``fit(sample, ...)``         fit on all the data       -> ``FitResult``
``fit_fold(train, test, …)`` fit one CV fold           -> ``FoldCurve`` or None
``save_posterior(...)``      export the posterior      -> path or None
``bayes_metrics(fit)``       PSIS-LOO / WAIC scores    -> dict
===========================  ==================================================
```

Each entry also carries the display fields (`key`, `label`, `short_label`,
`color`) that are the single source of truth for how a model appears in a legend, a
table, a plot and the map.

## Selecting models
`select` resolves a caller's argument to a list of entries in a stable order. It
accepts model keys and these group names:

```
===============  ===============================================================
``all``          every model in the catalog
``bdrc``         the four bdrc variants (also ``bdrc:all``)
``ratingcurve``  the ratingcurve power laws and spline (also ``rc``, ``rc:all``)
``default``      the curated comparison set (also ``None``)
===============  ===============================================================
```

## The models

```
==================  ============================================================
``power_law``       :math:`Q = C(h-e)^{\beta}`, one segment.
``power_law_2seg``  Segmented power law, two segments: a rating whose control
                    changes with stage, such as channel to overbank.
``power_law_3seg``  Three segments. Requires many measurements to be justified.
``spline``          Natural cubic spline in log space. Flexible and
                    atheoretical: suited to interpolation and not to
                    extrapolation, since nothing constrains it beyond the
                    measurements.
``bdrc_gplm0``      bdrc generalized power law: the exponent varies smoothly
                    with stage, as a Gaussian process on :math:`\beta(h)`, with
                    constant error variance. The default bdrc variant.
``bdrc_gplm``       The same, with error variance that varies with stage.
``bdrc_plm0``       bdrc plain power law, constant exponent and variance:
                    comparable to ``power_law`` but under bdrc's priors.
``bdrc_plm``        bdrc plain power law with stage-varying variance.
``linear``          Least-squares straight line in stage. The form used by 28 of
                    the 35 completed MAGL workbooks.
``quadratic``       Least-squares quadratic in stage, equivalent to an Excel
                    order-2 chart trendline. Not Bayesian, so it has no ELPD,
                    and it carries an effective stage range because a parabola
                    turns around. Four workbooks use it.
``exponential``     :math:`Q = A e^{Bh}`, fitted as OLS of :math:`\log Q` on
                    stage, which is what Excel's ``trendlineType="exp"``
                    computes. Three workbooks use it.
==================  ============================================================
```

## Roles
The three least-squares forms are in the catalog in order to reproduce what a field
spreadsheet's chart trendline computes, so that curve can be plotted on the same axes
as the Bayesian ratings. That is a different claim from recommending the form as a
model, and a report that does not distinguish the two is misleading. Each entry
therefore carries a `role`:

```
======================  ======================================================
``"model"``             a rating this package offers on its own account
``"spreadsheet_form"``  a least-squares form the field workbooks use, refitted
                        here
======================  ======================================================
```

Neither role is the literal workbook curve with its typed coefficients. That is not a
model but an external curve attached to a site, and it lives on
`SiteRating.extra_curves` (see
`limnotech_rating_curves.core.ExternalCurve`).
