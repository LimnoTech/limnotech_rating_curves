import numpy as np

from .compiled import CompiledModel
from ... import settings


def sample_hyperparameters(compiled: CompiledModel, *, method, draws, tune, chains,
                           seed, target_accept, cores, progressbar,
                           advi_n=settings.BDRC_ADVI_STEPS,
                           nuts_sampler=settings.BDRC_NUTS_SAMPLER):
    """Sample the marginal posterior, using NUTS or ADVI.

    Returns (theta_draws, idata) with theta_draws shaped (n_draws, n_theta),
    chains concatenated. Both methods return draws x chains draws in total. For
    ADVI they come from the fitted approximation in one block, so they are
    independent rather than a Markov chain and there is no sample_stats group to
    synthesize; `chains` is only a multiplier there, not four separate runs.

    The draws come back **chain-major**, which is what lets the closed-form
    quantities downstream be reshaped to (chain, draw) and given to ArviZ - see
    :func:`~limnotech_rating_curves.models.bdrc.criteria.attach_log_likelihood`.
    """
    import pymc as pm

    C = compiled.C
    with pm.Model() as model:
        # bdrc's own unconstrained parameterization is sampled directly: zeta =
        # log(h_min - c) already enforces c < h_min, the log-scales enforce
        # positivity, so no PyMC transform is needed and Flat carries the prior
        # through the Potential below. The density comes from the compiled model's
        # shared inputs, so this graph has no dataset shapes baked into it either.
        theta = pm.Flat("theta", shape=len(C.theta_names), initval=C.theta_mode)
        pm.Potential("bdrc_marginal", compiled.logp_graph(theta))
        for i, name in enumerate(C.theta_names):
            pm.Deterministic(name, theta[i])
        if method == "advi":
            approx = pm.fit(n=advi_n, method="advi", model=model,
                            random_seed=seed, progressbar=progressbar)
            # draws x chains, matching what the NUTS branch below returns in
            # total, so the two methods hand back the same-sized posterior and a
            # comparison between them is not also a comparison of Monte Carlo
            # error. Drawing from a fitted approximation is cheap -- measured at
            # ~0.15 ms per draw against a fixed per-fit cost of ~12 s -- so the
            # extra draws cost a fraction of a second.
            idata = approx.sample(draws=draws * chains, random_seed=seed,
                                  return_inferencedata=True)
            hist = np.asarray(getattr(approx, "hist", []), float)
            if hist.size:
                idata.attrs["sampling_method"] = "advi"
                idata.attrs["advi_n"] = int(advi_n)
                idata.attrs["advi_elbo_n"] = int(hist.size)
                idata.attrs["advi_elbo_final"] = float(hist[-1])
        else:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                              random_seed=seed, target_accept=target_accept,
                              progressbar=progressbar, nuts_sampler=nuts_sampler)
    theta_draws = (idata.posterior["theta"]
                   .transpose("chain", "draw", "theta_dim_0").values
                   .reshape(-1, len(C.theta_names)))
    return theta_draws, idata
