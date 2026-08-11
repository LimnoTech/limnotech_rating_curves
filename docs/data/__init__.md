# limnotech_rating_curves.data.__init__

Measurement sources, and the vertical reference each one reports.

Each loader returns a `Sample` ready to fit:

    from limnotech_rating_curves.data import gage_sample, sensor_sample

    gage_sample("04176356")                    # USGS field measurements
    sensor_sample("SBR-09")                    # a MAGL sensor
    station_sample(station, discharge)         # a loaded pagaia station
    colocated_sample("SBR-09")                 # MAGL stage against a gage's discharge

Also in this subpackage: `datum`, which puts stage
on the correct vertical reference, and
`zero_flow`, which estimates the stage at which
discharge would reach zero.

`dataretrieval`, `fb_pagaia` and `openpyxl` are imported inside the functions
that use them, so importing this subpackage remains inexpensive.
