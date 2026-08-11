# limnotech_rating_curves.data.magl_workbook

Reading the rating that each MAGL field workbook already contains.

Every `flow@<sensor>.xlsx` in `magl_curve_data` holds a rating that field staff
fitted in Excel: a chart trendline whose coefficients were typed into the summary
sheet and are used to fill that sheet's Estimated Discharge column. That curve is the
incumbent this package's models are compared against, so it has to be readable as a
curve rather than as a printed equation.

Three properties of the workbooks complicate reading it.

The equation is not on the stage axis. Twenty workbooks feed the raw radar distance to
water, in millimetres, into the equation; three feed the NAVD88 water-surface
elevation. The two become comparable only once the distance-based ones are mapped
through the workbook's own datum.

The datum is measured rather than assumed. The `Rating Curve` sheet plots a NAVD88
elevation and the summary sheet holds the distance for the same visit, so the
elevation tying the two axes together is recoverable per visit:
`elevation = stage + distance / 304.8`. `datum_tie` computes it and reports
the spread across visits, because a workbook that used two radar positions has no
single constant tie and should not be plotted as though it did.

The sign conventions vary. `=$L$3*K5-$M$3`, `=$M$3*B5+$N$3`,
`=($M$3*(C5^2))-(C5*$N$3)+$O$3` and a hand-typed `=90793*EXP((-0.006*B5))` all
occur. Rather than pattern-match each variant, `stored_equation` translates the
workbook's own formula into an expression in one variable and evaluates it, so the
resulting curve is arithmetically the one the spreadsheet computes. It is checked
against the workbook's cached Estimated Discharge values before being returned.

## Usage
    from limnotech_rating_curves.data import magl_workbook as workbook

    workbook.status()                        # every workbook, one row each
    rating = workbook.stored_equation("SBR-09")
    rating.discharge_at_stage([797.6, 799.0])
