# limnotech_rating_curves.data.datum

Converting reported stage onto the vertical reference a rating requires.

A rating relates discharge to gage height: the height of the water surface above a
fixed local reference low enough that gage height stays positive. Sensors rarely
report that quantity. They report either a water-surface elevation on a survey datum
such as NAVD88, or a distance down to the water surface from a mounting point.
Fitting a rating to the wrong one produces a curve that looks acceptable and is
wrong, because a constant added to stage is absorbed into the fitted power-law offset
and shifts the estimated exponent.

Every entry point that accepts data therefore accepts `stage_datum=`, which this
module interprets. Each form below converts a stage column to gage height:

```
===============================  =================================================
``stage_datum=None``             already gage height; no conversion
``stage_datum=612.34``           subtract this elevation (e.g. a USGS gage datum)
``stage_datum="lowest"``         reference slightly below the stage at which flow
                                 would reach zero (see
                                 :data:`LOWEST_MARGIN_FRACTION`)
``stage_datum=series``           a timeseries of reference elevations (NAVD88 ft),
                                 matched to each measurement's timestamp
``stage_datum=array``            one reference elevation per measurement
``stage_datum=StageDatum(...)``  any of the above, named and labelled
===============================  =================================================
```

## Examples
A USGS gage publishes `alt_va`, the NAVD88 elevation of gage height zero, so a
NAVD88 water-surface elevation becomes gage height by subtraction:

>>> to_gage_height([614.1, 615.2], 612.34).stage_ft
array([1.76, 2.86])

A sensor with no surveyed tie can be referenced to its own record:

>>> to_gage_height([614.1, 615.2], "lowest").stage_ft
array([0.11, 1.21])

The second form yields a local gage height. It supports fitting and prediction at that
sensor but is not comparable to another site's stage, which is why every conversion
records what it did in `ConvertedStage.note`. Note that the lowest measurement
does not land on 0 ft: a margin is left below it for the stage of zero flow, which the
rating models require to be positive and to stay off the boundary of its range.
`LOWEST_MARGIN_FRACTION` documents that margin, and why it is measured from the
estimated zero flow rather than from the lowest measurement whenever discharge is
available.

Distance-to-surface readings are converted by `distance_to_stage`. Sensors that
are physically raised or lowered are handled by
`stepwise_reference_elevation`.
