# limnotech_rating_curves.models.polynomial

Least-squares polynomial ratings: the spreadsheet curve, in code.

A field spreadsheet's rating is usually an Excel chart trendline: `Q` fitted as a
polynomial in stage by ordinary least squares, with the equation and R-squared shown on
the plot. It differs from everything else in this package, having no priors, no
posterior and no stage of zero flow, and it is included so that a spreadsheet curve can
be reproduced exactly and then compared against the Bayesian ratings on the same axes.

    rating = lrc.fit_rating(measurements, model="quadratic")
    rating.equation()          # 'Q = 6.2949 h^2 - 10024.7 h + 3990000'
    rating.coefficients        # highest power first, as Excel prints them
    rating.effective_range     # the stage band over which the curve may be used

The fit is on raw stage rather than on `h - e`, because that is what Excel's
`trendlineType="poly"` does and matching it is the purpose. One consequence: with
stage as an elevation near 800 ft the design matrix is badly conditioned, since `h^2`
is around 640000, so the coefficients are large, opposed in sign and individually
meaningless, and only their combination is interpretable. numpy's least squares handles
the conditioning more carefully than Excel's normal equations, so expect agreement in
the leading figures rather than in the last digit.

## The effective range
A power law rises without bound; a parabola turns around. Outside a limited band a
quadratic rating either falls as stage rises or becomes negative, and neither is usable
as a rating. Every fit therefore carries an `effective_range`, the stages where
the fitted curve is both increasing and positive, intersected with the measured range,
and `predict` returns NaN outside it rather than a value that is confidently
wrong.
