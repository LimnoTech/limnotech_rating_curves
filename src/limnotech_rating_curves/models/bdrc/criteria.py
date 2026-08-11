import numpy as np
import scipy.linalg as sla

from .compiled import CompiledModel
from .defaults import MAX_VARIANCE, NUGGET
from .posterior import mcmc_summary, normal_logpdf


def pointwise_log_likelihood(discharge, stage, h_rows, rating_curve_mean_posterior,
                             sigma_eps_posterior) -> np.ndarray:
    """(draws, n) pointwise log-likelihood at the observed measurements - R's
    bdrc:::log_lik_i. A Gaussian density on raw log discharge, evaluated at the
    posterior mean curve, which is the space metrics.py compares ELPD in.

    `sigma_eps_posterior` is either one value per draw (constant variance) or a
    (rows, draws) matrix (variance varying with stage), as R stores it. Takes plain
    arrays so it can be driven by an external posterior."""
    rows = np.searchsorted(np.asarray(h_rows, float), np.asarray(stage, float))
    log_mean = np.log(np.asarray(rating_curve_mean_posterior, float)[rows]).T
    sigma_eps_posterior = np.asarray(sigma_eps_posterior, float)
    sd = (sigma_eps_posterior[rows].T if sigma_eps_posterior.ndim == 2
          else sigma_eps_posterior[:, None])
    return normal_logpdf(np.log(np.asarray(discharge, float))[None, :], log_mean, sd)


def waic_from_log_likelihood(log_lik) -> dict:
    """WAIC and its pieces from a (draws, obs) pointwise log-likelihood - R's
    bdrc:::calc_waic, including its log-sum-exp lppd and the unbiased (ddof=1)
    posterior variance R's var() uses for p_waic."""
    log_lik = np.asarray(log_lik, float)
    n_draws = log_lik.shape[0]
    lppd_i = np.logaddexp.reduce(log_lik, axis=0) - np.log(n_draws)
    p_waic_i = np.var(log_lik, axis=0, ddof=1)
    waic_i = -2 * (lppd_i - p_waic_i)
    return {"waic": float(waic_i.sum()), "lppd": float(lppd_i.sum()),
            "p_waic": float(p_waic_i.sum()), "waic_i": waic_i}


def add_information_criteria(fit_obj, compiled: CompiledModel, theta_draws) -> None:
    """WAIC (R's bdrc:::calc_waic) and DIC (R's <model>.calc_Dhat)."""
    C = compiled.C
    fit_obj.log_lik_i = pointwise_log_likelihood(
        fit_obj.discharge, C.stage, fit_obj.h,
        fit_obj.rating_curve_mean_posterior, fit_obj.sigma_eps_posterior)
    log_lik = fit_obj.log_lik_i
    waic = waic_from_log_likelihood(log_lik)
    fit_obj.WAIC_i = waic["waic_i"]
    fit_obj.lppd = waic["lppd"]
    fit_obj.effective_num_param_WAIC = waic["p_waic"]
    fit_obj.WAIC = waic["waic"]

    fit_obj.D_hat = deviance_at_median(compiled, theta_draws)
    median_log_lik = float(mcmc_summary(log_lik.sum(axis=1)[None, :])["median"].iloc[0])
    fit_obj.effective_num_param_DIC = -2 * median_log_lik - fit_obj.D_hat
    fit_obj.DIC = fit_obj.D_hat + 2 * fit_obj.effective_num_param_DIC


def deviance_at_median(compiled: CompiledModel, theta_draws) -> float:
    """D_hat: -2 x log-likelihood at the median hyperparameters and the conditional
    mean of the latent parameters there.

    Note the phantom observation is given zero variance here, where the marginal
    density gives it sig_b^2 - reproducing an inconsistency inside bdrc itself
    (see DIFFERENCES_FROM_R note 6)."""
    C = compiled.C
    theta_median = np.median(theta_draws, axis=0)
    varr, Sig_x, X, _, _ = compiled.design_at(theta_median)
    if np.any(varr > MAX_VARIANCE):
        return np.nan
    Sig_eps = np.diag(np.append(varr, 0.0) if C.varying_exponent else varr)
    M = X @ Sig_x @ X.T + Sig_eps
    M[np.diag_indices_from(M)] += NUGGET
    L = np.linalg.cholesky(M)
    w = sla.solve_triangular(L, C.y - X @ C.mu_x, lower=True)
    x = C.mu_x + Sig_x @ (X.T @ sla.solve_triangular(L.T, w, lower=False))
    y_mean = (X @ x)[:C.n]
    return float(-2 * normal_logpdf(C.y_obs, y_mean, np.sqrt(varr)).sum())


def attach_log_likelihood(fit_obj, idata) -> None:
    """Put the pointwise log-likelihood into the InferenceData so metrics.py can
    score this fit with az.loo / az.waic straight from the posterior.

    The reshape back to (chain, draw) is valid because the hyperparameter draws come
    out of the sampler chain-major and every closed-form quantity is computed in
    that same order."""
    if idata is None:
        return
    import xarray as xr
    n_chains = idata.posterior.sizes["chain"]
    n_draws = idata.posterior.sizes["draw"]
    log_lik = fit_obj.log_lik_i.reshape(n_chains, n_draws, -1)
    idata.add_groups({"log_likelihood": xr.Dataset(
        {"obs": (("chain", "draw", "obs_dim_0"), log_lik)})})
