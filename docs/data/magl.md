# limnotech_rating_curves.data.magl

Rating curves for the LimnoTech MAGL sensor network.

This module is specific to MAGL. The rest of the package is general and does not
depend on it.

## Why MAGL needs its own module
The sensors do not measure stage. A MAGL water-level sensor is mounted above the water
and reports the distance down to the surface. Stage is the water-surface elevation on
NAVD88:

    reference elevation(t) = control point + recorded sensor moves up to t
    water-surface elevation(t) = reference elevation(t) - distance(t)

The reference is time-varying because raising or lowering a sensor shifts the distance
reading by the amount of the move; the reference must step with it or stage is
discontinuous across the move. Control-point elevations and the timestamped moves come
from two survey CSVs.

The distance record comes from two sources. A spreadsheet pickle covers the historical
window in feet, and the pagaia database covers the ongoing record in meters.
`distance_record_ft` merges them, spreadsheet first.

A second, smaller vertical correction applies. The surveyed control point is not the
point the radar measures from. The constant per-station gap between them is what
separates the control-point datum from the sensor datum that the spreadsheet's own
"Water Elevation (NAVD88)" column is on. `control_point_offset` measures it.

The discharge measurements are manual and few: three to seven per sensor, gauged by
wading, ADCP or FlowTracker, in two spreadsheet exports. That sample size is why
`cross_validate` here defaults to leave-one-out. A 90% holdout would train on
three points, on which most models fail to fit.

## Usage
    import limnotech_rating_curves as lrc
    from limnotech_rating_curves.data import magl

    magl.list_sensors()                       # sensors with discharge measurements
    sample = magl.sensor_sample("SBR-09")     # MAGL discharge against MAGL stage
    comparison = sample.compare(models="all")
    comparison.metrics

    magl.cross_validate(sample)               # leave-one-out

## The three samples this module can build
`sensor_sample`
    MAGL discharge against MAGL stage: the rating used at a MAGL sensor. Stage is a
    local gage height by default, referenced to the sensor's own lowest reading.
`colocated_sample`
    USGS field discharge against the co-located MAGL sensor's stage, tied to the
    gage's datum. This tests whether MAGL stage reproduces a rating the USGS already
    maintains, and is the most direct available check on the MAGL stage derivation.
`gage_sample`
    USGS discharge against USGS stage at the nearby gage. No MAGL data is involved; it
    is the reference case, re-exported here for convenience.
