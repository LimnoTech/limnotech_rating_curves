# limnotech_rating_curves.models.bdrc.criteria

Information criteria: the pointwise log-likelihood, WAIC and DIC.

The pointwise log-likelihood is the quantity the rest of the package depends on, since
it is what `limnotech_rating_curves.evaluate.metrics` scores PSIS-LOO from.
Because bdrc evaluates it in raw log-discharge space, it needs no rescaling to be
comparable with the ratingcurve family's.
