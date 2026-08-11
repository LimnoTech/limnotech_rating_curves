# limnotech_rating_curves.evaluate.diagnostics

Convergence diagnostics for a fitted rating.

Every rating in this package is fitted by MCMC, and an MCMC fit can fail without
raising: the chains return numbers regardless. Convergence should therefore be checked
before a rating's parameters or its predictive score are read. Three diagnostics
answer that question, and this module produces all three.

R-hat, the Gelman-Rubin statistic (`convergence`, `plot_convergence`).
Four chains start from different points. If they have all found the same distribution,
the variance between chains matches the variance within each one and their ratio is 1.
R-hat is that ratio, rank-normalized and split-chain. Above roughly 1.01 the chains
disagree, meaning at least one has not found the posterior, and nothing computed from
the fit, neither the parameters nor the ELPD, can be relied on until that is resolved.

Effective sample size (`convergence`). MCMC draws are correlated, so 4000 draws
carry less information than 4000 independent ones; ESS is the equivalent independent
count. `ess_bulk` governs how well the centre of the distribution is estimated, and
`ess_tail` governs the 5th and 95th percentiles, which is what a credible interval is
made of. A fit with adequate bulk ESS and poor tail ESS has a reliable mean and an
unreliable interval. Below a few hundred, increase the sampling budget.

Pareto-k (`plot_pareto_k`). This diagnoses the data rather than the sampler: per
measurement, whether PSIS-LOO could reweight the posterior as if that measurement were
absent. A high value means it could not, because the posterior depends too heavily on
that one point, which makes Pareto-k the closest thing in this package to a
per-measurement influence score. In a rating the highest-flow measurement is almost
always the one flagged. See `limnotech_rating_curves.evaluate.metrics`.

    rating = lrc.fit_rating(measurements, model="bdrc_gplm0")
    rating.diagnostics()            # the table
    lrc.diagnostics.plot_convergence(rating)
    lrc.diagnostics.plot_pareto_k(rating)
