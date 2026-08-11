# limnotech_rating_curves.models.bdrc.posterior

The closed-form half of the fit: latent draws, predictions, and the fit object.

Given the hyperparameters, everything else is exact. The latent vector
`x = (log a, b, beta)` is a Gaussian conditional, and `beta` at an unobserved
stage is a Gaussian-process (kriging) conditional, so this module needs no sampler:
it walks the hyperparameter draws once and assembles the rating curve from them.
