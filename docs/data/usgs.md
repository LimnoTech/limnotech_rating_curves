# limnotech_rating_curves.data.usgs

USGS data: field measurements, gage datums, and published rating curves.

A USGS gage provides three distinct products, which are not interchangeable.

Field measurements (`measurements`, `gage_sample`) are gauged
stage-discharge pairs: a hydrographer measured the discharge and read the gage. These
are the observed sample a rating is fitted to, and they are what makes USGS gages a
usable test of any rating method, since a gage typically carries dozens of independent
measurements over years.

The published rating (`published_rating`) is the curve the USGS maintains for
that gage, retrieved from the Water Data STAC API. It is a fitted product,
shift-corrected and maintained by hand, not a set of measurements. It is the reference
to compare a fit against and must not be treated as a sample to fit to.

The continuous record (`continuous_record`) is the gage's own 15-minute
gage height paired with its 15-minute discharge. It is a fourth thing, and the easiest
one to misuse: that discharge was computed *from* that stage through the published
rating, so the pairs are the rating's own output rather than independent observations of
it. Fitting to them fits the rating to itself. What they are good for is asking whether
the site was operated under one rating or several, and when it moved - a stage bin that
holds a wide range of discharges, organised by year, is a rating that has been rebuilt.
Fetched and cached a calendar year at a time, so a twenty-year span can be built up over
several calls.

`rating_deviation` puts the field measurements against the published curve, both as a
percent difference in discharge and as the stage shift that would put the curve through
each measurement. Near the bottom of a rating the percent difference is meaningless -
a foot of stage there is orders of magnitude of discharge - which is why the shift, in
feet, is the number the USGS itself keeps and the one to read.

The gage datum (`gage_datum`) is the elevation of gage height zero and the
vertical datum that elevation is on. It is required to convert a surveyed
water-surface elevation into that gage's gage height; see
`limnotech_rating_curves.data.datum`.

All three read public USGS web services (`dataretrieval` for measurements, the site
file for datums, the STAC ratings collection for published curves) through the cache
in `limnotech_rating_curves.support.cache`.

Gage numbers are strings. A leading zero is significant, so `"04176356"` is not
`4176356`, and site identifiers are never converted to integers anywhere in this
package.
