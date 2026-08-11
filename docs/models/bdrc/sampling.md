# limnotech_rating_curves.models.bdrc.sampling

Walking the marginal posterior: NUTS, or the fast variational fit.

Only the 2-10 hyperparameters are sampled. Everything else, the latent
`(log a, b, beta)` and the predictions, is a Gaussian conditional and is drawn in
closed form afterwards, in
`posterior`.
