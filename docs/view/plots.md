# limnotech_rating_curves.view.plots

Static rating figures, as matplotlib PNGs.

The interactive map is the primary view of a batch of results. These figures are for a
report or a notebook, and for the single-site case where a map is unnecessary.

Discharge is always on a log axis. A rating spans orders of magnitude, so on a linear
axis the low-flow measurements collapse toward the origin and the part of the curve
that is hardest to estimate becomes unreadable. Stage is drawn on whichever axis the
sample is on, named in the axis label, so a figure cannot be read against the wrong
datum.

Model colors come from `limnotech_rating_curves.models.catalog`, the same colors
the map uses, so a model has one color everywhere.

## The figures

Each one is reachable as a method on the object it draws, which is how a notebook should
call it; the functions here take that object as their first argument.

| function | method | what it shows |
|---|---|---|
| `plot_site` | — | one site: measurements, every fitted curve, the reference |
| `plot_log_log` | — | the curve against effective head, where a power law is a straight line |
| `plot_residual_ratio` | `RatingSet.plot_residuals` | predicted / observed at every measurement, against stage |
| `plot_fit_check` | `RatingModel.plot_check` | the two panels above, side by side |
| `plot_ranking` | `RatingSet.plot_ranking` | `elpd_diff` per model with error bars, and the 2-standard-error band |
| `plot_record` | `Sample.plot_record` | a continuous stage record with the measurements on it |
| `plot_fold` | — | one cross-validation fold: trained-on solid, held-out hollow |
| `plot_high_flow_error` | — | held-out predicted / observed against stage |
| `plot_rating_cloud` | — | a whole continuous record as a stage-discharge cloud, coloured by year |
| `plot_rating_deviation` | — | field measurements against a published rating: percent difference and shift, over time |

The residual-ratio figure is the companion to every curve plot. Overlaid curves hide a
bias at high flow, because the eye cannot read a factor of two off a log axis at the top
of the range; as a ratio, a factor of two reads the same at any flow and 1.0 is right.

`plot_record` takes its window from `Sample.time_range()` - the measured span padded by
`settings.GRID_PAD_FRACTION` at each end, the same padding `Sample.stage_grid` uses on
the stage axis - so no dates have to be looked up by hand.
