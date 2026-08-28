import numpy as np
import pandas as pd

from .compiled import c_upper_bound, compiled_for, finalize_grid
from .defaults import (ADVI_STEPS, CFS_TO_CMS, FT_TO_M, MODELS, NUM_CHAINS,
                       NUM_DRAWS, NUM_TUNE, NUTS_SAMPLER, SEED, target_accept_for)
from .design import components
from .posterior import assemble
from .sampling import sample_hyperparameters


def fit(discharge, stage, model="gplm0", *, method="nuts", c_param=None,
        h_max=None, forcepoint=None, draws=NUM_DRAWS, tune=NUM_TUNE,
        chains=NUM_CHAINS, cores=1, seed=SEED, target_accept=None,
        progressbar=False, advi_n=ADVI_STEPS, nuts_sampler=NUTS_SAMPLER):
    """Fit a bdrc rating curve. SI units: `discharge` in m^3/s, `stage` in m.

    model       one of "plm0", "plm", "gplm0", "gplm"
    method      "nuts" (default) or "advi"
    c_param     known stage of zero flow (m); inferred when None
    h_max       extend the rating curve up to this stage (m)
    forcepoint  boolean mask of measurements the curve should be forced through
    cores       1 keeps the four chains sequential; PyMC's multiprocessing
                deadlocks under some Windows launchers (see the README)
    target_accept NUTS target acceptance rate; None resolves it from the number of
                measurements with `defaults.target_accept_for`, which raises it on
                a short record so the sampler does not diverge there
    nuts_sampler which NUTS implementation walks the marginal: "pymc", "nutpie",
                "numpyro" or "blackjax". All four sample the same posterior, so
                this changes the wall clock and nothing else.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model: {model!r} (expected one of {MODELS})")
    discharge = np.asarray(discharge, float).ravel()
    stage = np.asarray(stage, float).ravel()
    if discharge.size != stage.size:
        raise ValueError("discharge and stage must have the same length")
    if stage.size < 2:
        raise ValueError("at least two paired observations of stage and discharge "
                         "are required to fit a rating curve")
    if np.any(discharge <= 0):
        raise ValueError("all discharge measurements must be strictly greater than "
                         "zero; if you know the stage of zero discharge, use c_param")
    if c_param is not None and stage.min() < c_param:
        raise ValueError("c_param must be lower than the minimum stage in the data")

    if target_accept is None:
        target_accept = target_accept_for(stage.size)

    forcepoint = (np.zeros(stage.size, bool) if forcepoint is None
                  else np.asarray(forcepoint, bool))
    order = np.argsort(stage, kind="stable")
    stage, discharge, forcepoint = stage[order], discharge[order], forcepoint[order]
    y_obs = np.log(discharge)

    c_upper = None if c_param is not None else c_upper_bound(y_obs, stage, forcepoint)
    C = components(model, y_obs, stage, c_param, forcepoint, h_min=c_upper)
    C.c_upper = c_upper
    # Binding a dataset to the cached graph; nothing is compiled unless this is the
    # first (model, known_c) of its kind in this process.
    compiled = compiled_for(C)
    finalize_grid(compiled, h_max)

    theta_draws, idata = sample_hyperparameters(
        compiled, method=method, draws=draws, tune=tune, chains=chains,
        seed=seed, target_accept=target_accept, cores=cores,
        progressbar=progressbar, advi_n=advi_n, nuts_sampler=nuts_sampler)

    return assemble(compiled, theta_draws, idata, seed=seed,
                    run_info={"model": model, "c_param": c_param, "h_max": h_max,
                              "forcepoint": forcepoint, "c_upper": c_upper,
                              "method": method, "advi_n": advi_n,
                              "nuts_sampler": nuts_sampler,
                              "target_accept": target_accept,
                              "draws": draws, "tune": tune, "chains": chains,
                              "seed": seed})


def fit_predict(stage_ft, discharge_cfs, grid_stage_ft, *, model="gplm0",
                method="nuts", c_param_ft=None, with_loglik=False,
                advi_n=ADVI_STEPS, **kwargs):
    """Fit one sample and predict discharge on `grid_stage_ft`, in ft and cfs.

    Returns a DataFrame with columns stage_ft, q_median_cfs, q_lower_cfs,
    q_upper_cfs (the posterior-predictive median and 2.5 / 97.5% bounds) and
    q_mean_cfs, q_posterior_median_cfs, q_posterior_lower_cfs,
    q_posterior_upper_cfs (the same summaries of the rating itself, without gaging
    scatter), keeping only finite, positive predictions. With `with_loglik=True` returns a dict of
    {"curve", "loglik", "native", "fit"}, where `loglik` is the (draws, obs)
    pointwise log-likelihood in raw log-discharge space and `native` carries this
    model's own WAIC / DIC.

    bdrc's priors are tuned to SI, so the fit runs in m and m^3/s and the curve is
    converted back - the same conversion the retired R bridge did."""
    stage = np.asarray(stage_ft, float).ravel() * FT_TO_M
    discharge = np.asarray(discharge_cfs, float).ravel() * CFS_TO_CMS
    grid = np.asarray(grid_stage_ft, float).ravel() * FT_TO_M
    c_param = None if c_param_ft is None else float(c_param_ft) * FT_TO_M

    fitted = fit(discharge, stage, model, method=method, c_param=c_param,
                 h_max=grid.max(), advi_n=advi_n, **kwargs)
    predicted = fitted.predict(grid)
    rating = fitted.predict_posterior(grid)
    curve = pd.DataFrame({
        "stage_ft": predicted["h"] / FT_TO_M,
        "q_median_cfs": predicted["median"] / CFS_TO_CMS,
        "q_lower_cfs": predicted["lower"] / CFS_TO_CMS,
        "q_upper_cfs": predicted["upper"] / CFS_TO_CMS,
        "q_mean_cfs": rating["mean"] / CFS_TO_CMS,
        "q_posterior_median_cfs": rating["median"] / CFS_TO_CMS,
        "q_posterior_lower_cfs": rating["lower"] / CFS_TO_CMS,
        "q_posterior_upper_cfs": rating["upper"] / CFS_TO_CMS})
    curve = curve[np.isfinite(curve["q_median_cfs"]) & (curve["q_median_cfs"] > 0)]
    if not with_loglik:
        return curve
    return {"curve": curve, "loglik": fitted.log_lik_i, "fit": fitted,
            "native": {"WAIC": fitted.WAIC, "DIC": fitted.DIC, "lppd": fitted.lppd,
                       "p_waic": fitted.effective_num_param_WAIC}}
