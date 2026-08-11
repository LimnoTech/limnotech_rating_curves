# limnotech_rating_curves.data.pagaia

Fitting a rating to a pagaia station.

`fb-pagaia` is LimnoTech's client for the Freeboard ODM2 database. It represents each
monitoring station as an object that knows its coordinates, the variables it records,
and how to fetch them, so a station can be passed to this package directly rather than
as separate arrays.

**It is an optional extra**, since it reads a LimnoTech database over the VPN and is of
no use without an account there. Install it with `pip install "limnotech-rating-curves
[pagaia]"`; everything else in the package works without it. This module itself imports
without it, because the functions that take a station work on what the station object
does rather than on its type - only the loaders that *build* a station need the client,
and they call `require()`, which raises an `ImportError` naming the extra. `installed()`
answers the same question without raising, for code that has a second source to fall
back on.

The session and the station are constructed by the caller, so that the environment
connected to and the station loaded are both explicit:

    from fb_pagaia import core, defaults, utils
    import limnotech_rating_curves as lrc

    api = utils.start_session()
    station = core.Station.from_name(name="SBR-09_WL", api=api)

    sample = lrc.station_sample(station, discharge=field_measurements)
    rating = lrc.fit_rating(sample)

## What the station provides, and what the caller supplies
A water-level station records stage or distance-to-surface on its own schedule. It does
not record discharge. A rating therefore needs two further inputs:

1. The discharge measurements, passed as `discharge`: a DataFrame with a time column
   and a discharge column, or a Series indexed by time.
2. The vertical meaning of the station's readings, passed as `stage_datum=`. If the
   station reports gage height, no argument is needed. If it reports a water-surface
   elevation, pass the elevation of gage height zero, or `"lowest"` where there is no
   surveyed tie. If it reports distance down to the water, pass
   `reference_elevation` so the reading can be converted to an elevation first.

`station_sample` performs the matching: for each discharge measurement it takes
the station's nearest reading within a tolerance and reports the time difference, so a
pair matched across a two-hour gap on a rising limb is visible in the record rather
than accepted silently.

MAGL stations are a special case, because their vertical reference comes from a survey
table and steps when a sensor is moved. They are handled by
`limnotech_rating_curves.data.magl`.
