# limnotech_rating_curves.models.__init__

The model catalog, and the two packages that fit the models.

Models are normally reached through `limnotech_rating_curves.fit_rating`, which
takes a model key. This subpackage defines those keys and contains the fitting code:

    from limnotech_rating_curves.models import MODELS, select

    [entry.key for entry in MODELS]      # every model, in display order
    select("bdrc")                       # a group: the four bdrc variants

```
==================  ==========================================================
:mod:`catalog`      one :class:`~catalog.ModelEntry` per model: key, label,
                    color, and the four operations used downstream
:mod:`ratingcurve`  the USGS ``ratingcurve`` power law and spline
:mod:`bdrc`         the bdrc generalized power law, a native PyMC port
:mod:`polynomial`   least-squares linear and quadratic, the spreadsheet forms
:mod:`exponential`  least-squares exponential, Excel's ``trendlineType="exp"``
==================  ==========================================================
```

The modules other than `catalog` are implementation. Ordinary use does not
require importing `bdrc` or `ratingcurve` directly.
