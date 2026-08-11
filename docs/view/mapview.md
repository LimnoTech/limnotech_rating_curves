# limnotech_rating_curves.view.mapview

The interactive map: every site, with an inspector for the site under the cursor.

This is the primary way to review a set of fitted ratings, because the question is
usually not what one curve looks like but which model performs best at each site and
whether that answer changes between sites. One map answers that more directly than one
figure per site.

Layout. Sites are drawn on a topographic basemap on the left, with the legend over its
bottom-left corner. On the right is a two-pane inspector that redraws for whichever site
the cursor is over: the plot on top, a control row beneath it, and the score table at
the bottom. Axes are frozen per site, so curves stay comparable across panes and across
folds. `layout_boxes` and `layout_overlaps` compute the pane rectangles and
check that none overlap, and `build_map` logs a warning when a change breaks
that.

Fitted curves pane. The site's measurements; every model fitted on all of them, each
with its 95% credible band; every curve the package did not fit; and a table of
per-curve scores. Each column header carries a tooltip describing the metric and how to
read it, taken from
`limnotech_rating_curves.evaluate.metrics.METRIC_GLOSSARY`, since a table of
eight numbers is not usable without them.

Hovering a measurement shows its Pareto-k under each model, which identifies the
measurements the fits depend on most, so a curve supported by a single high-flow point
is visible as such.

Cross-validation pane. Present when the build ran cross-validation. A model selector
and fold arrows step through each (model, fold) refit, showing the fold's curve with
its credible band, the training points drawn solid and the held-out points greyed, and
a table giving that fold's held-out scores, the summary over all folds, and the all-data
PSIS-LOO score. Those are three different quantities and the table distinguishes them:
the fold scores come from refits on subsets, while PSIS-LOO is an approximation
computed once from the full-data fit.

## Line style carries the category, hue carries the model
Colour alone is unreliable on a projector, in print, and for a colour-blind reader, so
line style encodes the category and hue distinguishes models within it:

```
===============================  ====================  ========================
Bayesian ratings (ratingcurve)   solid                 blues, spline in purple
Bayesian ratings (bdrc)          solid                 greens and teals
least-squares polynomial forms   dashed                ochre and orange
least-squares exponential form   dotted                brown
USGS published rating            dashed, black         black
a field spreadsheet's own curve  long-dash-dot, heavy  magenta
===============================  ====================  ========================
```

Styles come from the catalog entries (`entry.color`, `entry.dash`) and from each
`ExternalCurve` rather than from this module, so
the map, the static figures and the reports agree by construction.

Implementation. The output is one self-contained Plotly HTML file. Every pane trace is
pre-built and tagged with a `meta` dict
(`{sample, pane, model, fold, kind, legendkey}`). A state machine in the injected
script decides which traces are visible for the current `{site, pane, model, fold}`
and which legend groups the reader has switched off, then relayouts the axes and the
pane's controls. Legend clicks are intercepted by that script rather than handled by
Plotly, because Plotly's own toggle is overwritten by the next repaint. There is no
framework beyond Plotly and that script.
