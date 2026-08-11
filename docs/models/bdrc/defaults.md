# limnotech_rating_curves.models.bdrc.defaults

bdrc's own constants: the priors from the paper, and the sampling budget.

The prior values are tuned to SI units (stage in m, discharge in m^3/s) in
Hrafnkelsson et al. (2022) and are transcribed from R's `bdrc:::priors`. They
must not be rescaled: `fit_predict` converts
ft/cfs measurements into SI for the fit precisely so these stay valid.
