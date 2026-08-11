# limnotech_rating_curves.support.cache

Pickle-through cache for the slow fetches (USGS web services, pagaia).

A rating fit is inexpensive next to the network round trips that assemble its data, so
every fetch goes through `cached`. The cache key is an md5 of the call's
arguments, so a repeated request with the same arguments is served from a file.

    stage = cached("magl_stage", (station, start, end), lambda: fetch(...))

Nothing here is rating-specific. It is a two-function utility, so that the data-source
modules do not each implement their own caching.
