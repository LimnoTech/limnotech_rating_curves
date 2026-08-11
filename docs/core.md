# limnotech_rating_curves.core

The data a rating is fitted to, and the records a fit produces.

`Sample` carries the observations: paired stage and discharge measurements,
their source, and the datum the stage is on. Every function in the package that fits,
scores, plots or maps a rating takes a `Sample`, and `Sample.of`
constructs one from the common input shapes:

    Sample.of(dataframe)                     # columns named stage / discharge
    Sample.of(dataframe, stage="wse_ft")     # or named explicitly
    Sample.of(stage_ft, discharge_cfs)       # two arrays
    Sample.of(sample)                        # already a Sample: returned unchanged

The remaining types are the records a fit produces: `FitResult` (one model
fitted to one sample), `FoldCurve` (one cross-validation fold),
`Metrics` (the scores) and `SiteRating` (every model fitted at one
monitoring site, which is what the map draws).

Units are feet for stage and cubic feet per second for discharge throughout. This is a
constraint rather than an assumption: the USGS field measurements, the published
ratings and the MAGL survey are all in feet, and
`limnotech_rating_curves.data.datum` converts other inputs on the way in.
