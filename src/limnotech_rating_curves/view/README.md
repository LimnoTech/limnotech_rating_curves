# view

Figures. Most are reachable as methods (`rating.plot()`,
`rating_set.plot_ranking()`, `cv.plot()`); the functions here are what those call,
and the map has no method form.

```python
from limnotech_rating_curves import view

view.plot_site(site)
view.build_map(sites, "ratings.html", cv_data=payload)
```

## `plots.py` — matplotlib

| Name | Shows |
| --- | --- |
| `plot_site(site)` | One site: measurements, every fitted curve, the reference curve |
| `plot_log_log(rating)` | A rating on log-log axes against effective head |
| `plot_residual_ratio(rating)` | Predicted / observed discharge at every measurement, against stage |
| `plot_fit_check(rating)` | The two panels a rating is judged on: shape and residuals |
| `plot_ranking(rating_set)` | Model ranking by predictive score, with uncertainty on each difference |
| `plot_fold(fold)` | One cross-validation fold: its curve and what it did and did not see |
| `plot_high_flow_error(cross_validation)` | Held-out predicted / observed against stage |
| `plot_record(sample, series)` | A continuous stage record with the fitted measurements on it |
| `plot_rating_cloud(record)` | Every paired reading in a continuous record as a stage-discharge cloud |
| `plot_rating_deviation(deviation)` | How far each field measurement sits from the published rating, over time |
| `save_site_figure(site, directory)`, `save_fold_figures(site, folds, directory)` | Write them to disk |
| `draw_curve(...)`, `limit_discharge_axis(...)` | The shared drawing primitives; curves outside the measured range are marked as extrapolation |
| `write_figures_readme(directory)` | Explain the figure layout beside the output |

## `mapview.py` — the interactive Plotly map

| Name | Purpose |
| --- | --- |
| `build_map(sites, output_html=None, cv_data=None, ...)` | Assemble and write the map |
| `build_figure(sites, ...)` | The same figure without writing anything |
| `layout_boxes(has_cv)` | The named rectangles of the inspector panel, in paper coordinates |
| `layout_overlaps(has_cv)` | Panel rectangles that intersect — empty is the correct answer |

Hovering a site swaps the right-hand panel to that site's fits, metrics table and
cross-validation folds.
