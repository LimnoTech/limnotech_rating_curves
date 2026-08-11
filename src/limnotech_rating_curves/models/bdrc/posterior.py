from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.linalg as sla

from .compiled import CompiledModel
from .defaults import NUGGET, N_SPLINE_BASIS
from .design import matern52_correlation


def normal_logpdf(x, mean, sd):
    return -0.5 * (np.log(2 * np.pi) + 2 * np.log(sd) + ((x - mean) / sd) ** 2)


def latent_draw(theta, compiled: CompiledModel, rng):
    """One draw of x = (log a, b, beta) from its Gaussian conditional given theta,
    with the observed-stage mean and predictive draws that go with it.

    Uses bdrc's own sampling route (draw from the prior, then correct with the
    Kalman-style update) rather than forming the conditional covariance, which
    keeps the algebra identical to the C++ and stays stable when Sig_x is
    ill-conditioned."""
    C = compiled.C
    varr, Sig_x, X, _, eta = compiled.design_at(theta)
    M = X @ Sig_x @ X.T + np.diag(
        np.append(varr, C.Sig_ab[1, 1]) if C.varying_exponent else varr)
    M[np.diag_indices_from(M)] += NUGGET
    L = np.linalg.cholesky(M)

    prior_draw = C.mu_x + sla.cholesky(Sig_x, lower=True) @ rng.standard_normal(C.mu_x.size)
    noise = rng.standard_normal(C.n) * np.sqrt(varr)
    if C.varying_exponent:
        noise = np.append(noise, rng.standard_normal() * np.sqrt(C.Sig_ab[1, 1]))
    residual = X @ prior_draw - C.y + noise
    W = sla.solve_triangular(L, X @ Sig_x, lower=True)
    x = prior_draw - W.T @ sla.solve_triangular(L, residual, lower=True)

    y_mean = (X @ x)[:C.n]
    y_pred = y_mean + rng.standard_normal(C.n) * np.sqrt(varr)
    return x, y_mean, y_pred, varr, eta


def unobserved_draw(theta, x, eta, compiled: CompiledModel, rng):
    """Predictions at the unobserved stages for one draw: (y_mean, y_pred, varr).

    For gplm0/gplm this first krigs beta onto the grid from the Gaussian-process
    prior, conditional on beta at the observed stages. Grid stages at or below c
    get -inf log discharge (zero flow), as in R."""
    C = compiled.C
    take = dict(zip(C.theta_names, np.asarray(theta, float)))
    n_grid = C.h_u.size

    if C.varying_variance:
        z = np.array([take[f"z_{i}"] for i in range(1, 6)])
        eta_grid = C.P @ np.concatenate(([take["eta_1"]],
                                         np.exp(take["log_sigma_eta"]) * z))
        varr = np.exp(C.B_u @ eta_grid)
    else:
        varr = np.full(n_grid, np.exp(take["log_sigma_eps2"]))

    beta_grid = None
    if C.varying_exponent:
        variance = np.exp(2 * take["log_sigma_beta"])
        joint = variance * matern52_correlation(C.dist_all,
                                                np.exp(take["log_phi_beta"]))
        joint[np.diag_indices_from(joint)] += NUGGET
        n_obs = C.n_unique
        obs_obs = joint[:n_obs, :n_obs]
        grid_grid = joint[n_obs:, n_obs:]
        obs_grid = joint[:n_obs, n_obs:]
        grid_obs = joint[n_obs:, :n_obs]
        solved = np.linalg.solve(obs_obs, np.column_stack((x[2:], obs_grid)))
        mean = grid_obs @ solved[:, 0]
        covariance = grid_grid - grid_obs @ solved[:, 1:]
        beta_grid = mean + stable_cholesky(covariance) @ rng.standard_normal(n_grid)

    y_mean = np.full(n_grid, -np.inf)
    y_pred = np.full(n_grid, -np.inf)
    if C.known_c:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_stage = np.log(C.h_u - C.c_param)
        exponent = x[1] + beta_grid if C.varying_exponent else x[1]
        y_mean = x[0] + exponent * log_stage
        if C.varying_exponent:
            # R forces zero discharge at the first grid stage, which is c itself
            y_mean[0] = -np.inf
        y_pred = y_mean + rng.standard_normal(n_grid) * np.sqrt(varr)
    else:
        c = C.h_min - np.exp(take["zeta"])
        above = C.h_u > c
        if above.any():
            log_stage = np.log(C.h_u[above] - c)
            exponent = x[1] + beta_grid[above] if C.varying_exponent else x[1]
            y_mean[above] = x[0] + exponent * log_stage
            y_pred[above] = (y_mean[above]
                             + rng.standard_normal(above.sum()) * np.sqrt(varr[above]))
    y_mean[~np.isfinite(y_mean)] = -np.inf
    y_pred[~np.isfinite(y_pred)] = -np.inf
    return y_mean, y_pred, varr, beta_grid


def stable_cholesky(matrix):
    """Lower Cholesky factor, retrying with a growing jitter. The kriging Schur
    complement is positive definite in exact arithmetic but can lose definiteness
    on a dense grid; R's arma::chol simply errors there
    (see DIFFERENCES_FROM_R note 13)."""
    scale = float(np.mean(np.diag(matrix))) or 1.0
    for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        try:
            return sla.cholesky(matrix + jitter * scale * np.eye(matrix.shape[0]),
                                lower=True)
        except sla.LinAlgError:
            continue
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


# ==============================================================================
# summaries (R: bdrc:::get_MCMC_summary_cpp)
# ==============================================================================

def mcmc_summary(draws_by_row, h=None) -> pd.DataFrame:
    """bdrc's 2.5 / 50 / 97.5% summary of a (rows, draws) matrix.

    Deliberately uses bdrc's order-statistic rule -- sort, then take element
    floor(q*(size-1)) -- not an interpolated percentile, so the numbers match R
    (see DIFFERENCES_FROM_R note 9)."""
    draws_by_row = np.atleast_2d(np.asarray(draws_by_row, float))
    size = draws_by_row.shape[1]
    ordered = np.sort(draws_by_row, axis=1)
    pick = lambda q: ordered[:, int(np.floor(q * (size - 1)))]
    out = pd.DataFrame({"lower": pick(0.025), "median": pick(0.5), "upper": pick(0.975)})
    if h is not None:
        out.insert(0, "h", np.asarray(h, float))
    return out


# ==============================================================================
# the fit object
# ==============================================================================

@dataclass
class BdrcFit:
    """A fitted bdrc rating curve. Field names follow the R model object so the two
    can be compared directly.

    Posterior matrices are (rows, draws) with rows in ascending stage - the union of
    the unique observed stages and the prediction grid - matching R's
    `rating_curve_posterior` layout."""
    model: str
    stage: np.ndarray                 # observed stages, ascending (m)
    discharge: np.ndarray             # observed discharge, same order (m^3/s)
    h: np.ndarray                     # the rating curve's stage rows (m)

    a_posterior: np.ndarray
    b_posterior: np.ndarray
    c_posterior: np.ndarray | None
    sigma_eps_posterior: np.ndarray   # scalar per draw (plm0/gplm0) or (rows, draws)
    sigma_beta_posterior: np.ndarray | None = None
    phi_beta_posterior: np.ndarray | None = None
    sigma_eta_posterior: np.ndarray | None = None
    eta_posterior: np.ndarray | None = None      # (6, draws), on R's eta scale
    beta_posterior: np.ndarray | None = None
    f_posterior: np.ndarray | None = None

    rating_curve_posterior: np.ndarray = None        # posterior predictive discharge
    rating_curve_mean_posterior: np.ndarray = None   # posterior mean discharge
    rating_curve: pd.DataFrame = None
    rating_curve_mean: pd.DataFrame = None
    beta_summary: pd.DataFrame = None
    f_summary: pd.DataFrame = None
    sigma_eps_summary: pd.DataFrame = None
    param_summary: pd.DataFrame = None

    posterior_log_likelihood: np.ndarray = None
    log_lik_i: np.ndarray = None      # (draws, n) pointwise, raw log-discharge space
    lppd: float = np.nan
    WAIC: float = np.nan
    WAIC_i: np.ndarray = None
    effective_num_param_WAIC: float = np.nan
    D_hat: float = np.nan
    DIC: float = np.nan
    effective_num_param_DIC: float = np.nan

    idata: object = None              # ArviZ InferenceData (posterior + log_likelihood)
    run_info: dict = field(default_factory=dict)

    def predict(self, newdata=None) -> pd.DataFrame:
        """Discharge quantiles at `newdata` (stage in m), by linear interpolation of
        the fitted curve - R's predict.plm0 / predict.gplm0. Stages outside the
        curve return 0, and stages above its maximum are an error (raise `h_max` at
        fit time to extrapolate)."""
        if newdata is None:
            newdata = self.h
        newdata = np.asarray(newdata, float)
        if np.isnan(newdata).any():
            raise ValueError("newdata must not contain NaN")
        if (newdata > self.h.max()).any():
            raise ValueError("newdata must lie within the fitted stage range; use "
                             "h_max at fit time to extrapolate to higher stages")
        out = pd.DataFrame({"h": newdata})
        for column in ("lower", "median", "upper"):
            interpolated = np.interp(newdata, self.h, self.rating_curve[column],
                                     left=np.nan, right=np.nan)
            out[column] = np.nan_to_num(interpolated, nan=0.0)
        return out

    def __repr__(self):
        n_draws = self.b_posterior.size
        return (f"BdrcFit({self.model}, n={self.stage.size}, draws={n_draws}, "
                f"WAIC={self.WAIC:.3f})")


def assemble(compiled: CompiledModel, theta_draws, idata, *, seed, run_info):
    """Turn hyperparameter draws into the full fit object: latent draws, the rating
    curve and its summaries, the pointwise log-likelihood, WAIC and DIC."""
    from .criteria import add_information_criteria, attach_log_likelihood

    C = compiled.C
    rng = np.random.default_rng(seed)
    n_draws = theta_draws.shape[0]

    latent = np.empty((n_draws, C.mu_x.size))
    y_mean_obs = np.empty((n_draws, C.n))
    y_pred_obs = np.empty((n_draws, C.n))
    varr_obs = np.empty((n_draws, C.n))
    y_mean_grid = np.empty((n_draws, C.h_u.size))
    y_pred_grid = np.empty((n_draws, C.h_u.size))
    varr_grid = np.empty((n_draws, C.h_u.size))
    beta_grid = (np.empty((n_draws, C.h_u.size)) if C.varying_exponent else None)
    log_lik = np.empty(n_draws)

    for i, theta in enumerate(theta_draws):
        x, y_mean, y_pred, varr, eta = latent_draw(theta, compiled, rng)
        latent[i], y_mean_obs[i], y_pred_obs[i], varr_obs[i] = x, y_mean, y_pred, varr
        log_lik[i] = normal_logpdf(C.y_obs, y_mean, np.sqrt(varr)).sum()
        grid_mean, grid_pred, grid_varr, grid_beta = unobserved_draw(
            theta, x, eta, compiled, rng)
        y_mean_grid[i], y_pred_grid[i], varr_grid[i] = grid_mean, grid_pred, grid_varr
        if C.varying_exponent:
            beta_grid[i] = grid_beta

    # rows of the rating curve: unique stages over observations + grid, ascending
    stage_all = np.concatenate((C.stage, C.h_u))
    h_rows, first_row = np.unique(stage_all, return_index=True)
    stack = lambda obs, grid: np.concatenate((obs, grid), axis=1)[:, first_row].T

    fit_obj = BdrcFit(
        model=C.model, stage=C.stage, discharge=np.exp(C.y_obs), h=h_rows,
        a_posterior=np.exp(latent[:, 0]), b_posterior=latent[:, 1],
        c_posterior=(None if C.known_c
                     else C.h_min - np.exp(theta_draws[:, 0])),
        sigma_eps_posterior=None,
        rating_curve_posterior=np.exp(stack(y_pred_obs, y_pred_grid)),
        rating_curve_mean_posterior=np.exp(stack(y_mean_obs, y_mean_grid)),
        posterior_log_likelihood=log_lik,
        idata=idata, run_info=run_info)

    take = {name: theta_draws[:, i] for i, name in enumerate(C.theta_names)}
    if C.varying_variance:
        # R reports eta as P %*% (eta_1, sigma_eta * z), i.e. the cumulative sums
        z_draws = np.array([take[f"z_{i}"] for i in range(1, 6)])
        fit_obj.sigma_eta_posterior = np.exp(take["log_sigma_eta"])
        fit_obj.eta_posterior = C.P @ np.vstack(
            (take["eta_1"], fit_obj.sigma_eta_posterior * z_draws))
        fit_obj.sigma_eps_posterior = np.sqrt(stack(varr_obs, varr_grid))
        fit_obj.sigma_eps_summary = mcmc_summary(fit_obj.sigma_eps_posterior, h=h_rows)
    else:
        fit_obj.sigma_eps_posterior = np.sqrt(np.exp(take["log_sigma_eps2"]))
    if C.varying_exponent:
        fit_obj.sigma_beta_posterior = np.exp(take["log_sigma_beta"])
        fit_obj.phi_beta_posterior = np.exp(take["log_phi_beta"])
        beta_obs = latent[:, 2:] @ C.A.T          # per observation, then to rows
        fit_obj.beta_posterior = stack(beta_obs, beta_grid)
        fit_obj.f_posterior = fit_obj.beta_posterior + fit_obj.b_posterior[None, :]
        fit_obj.beta_summary = mcmc_summary(fit_obj.beta_posterior, h=h_rows)
        fit_obj.f_summary = mcmc_summary(fit_obj.f_posterior, h=h_rows)

    fit_obj.rating_curve = mcmc_summary(fit_obj.rating_curve_posterior, h=h_rows)
    fit_obj.rating_curve_mean = mcmc_summary(fit_obj.rating_curve_mean_posterior,
                                             h=h_rows)
    fit_obj.param_summary = param_summary(fit_obj, C, theta_draws, idata)

    add_information_criteria(fit_obj, compiled, theta_draws)
    attach_log_likelihood(fit_obj, idata)
    return fit_obj


def param_summary(fit_obj, C, theta_draws, idata) -> pd.DataFrame:
    """R's param_summary: a, b, then the hyperparameters on their natural scale,
    with r_hat and effective sample size."""
    columns = {"a": fit_obj.a_posterior, "b": fit_obj.b_posterior}
    if not C.known_c:
        columns["c"] = fit_obj.c_posterior
    if C.varying_variance:
        columns["sigma_eta"] = fit_obj.sigma_eta_posterior
        for i in range(N_SPLINE_BASIS):
            columns[f"eta_{i + 1}"] = fit_obj.eta_posterior[i]
    else:
        columns["sigma_eps"] = fit_obj.sigma_eps_posterior
    if C.varying_exponent:
        columns["sigma_beta"] = fit_obj.sigma_beta_posterior
        columns["phi_beta"] = fit_obj.phi_beta_posterior
    # R's order is a, b, c, then the model's own hyperparameters
    order = ["a", "b"] + (["c"] if "c" in columns else [])
    if C.model == "plm0":
        order += ["sigma_eps"]
    elif C.model == "gplm0":
        order += ["sigma_eps", "sigma_beta", "phi_beta"]
    elif C.model == "plm":
        order += ["sigma_eta"] + [f"eta_{i}" for i in range(1, 7)]
    else:
        order += ["sigma_beta", "phi_beta", "sigma_eta"] + [f"eta_{i}" for i in range(1, 7)]

    summary = mcmc_summary(np.vstack([columns[name] for name in order]))
    summary.index = order
    diagnostics = sampler_diagnostics(idata, C)
    summary["eff_n_samples"] = [diagnostics["ess"].get(name, np.nan) for name in order]
    summary["r_hat"] = [diagnostics["r_hat"].get(name, np.nan) for name in order]
    return summary


def sampler_diagnostics(idata, C) -> dict:
    """r_hat and effective sample size for the sampled hyperparameters, from ArviZ.

    R computes these with its own split-chain variogram estimator over
    (a, b, theta); here they come from ArviZ's rank-normalized diagnostics over the
    sampled theta only, since a and b are drawn in closed form rather than by MCMC
    (see DIFFERENCES_FROM_R note 3)."""
    import arviz as az
    ess, r_hat = {}, {}
    try:
        summary = az.summary(idata, var_names=list(C.theta_names), round_to=None)
    except Exception:  # noqa: BLE001
        return {"ess": ess, "r_hat": r_hat}
    natural = {"zeta": "c", "log_sigma_eps2": "sigma_eps",
               "log_sigma_beta": "sigma_beta", "log_phi_beta": "phi_beta",
               "log_sigma_eta": "sigma_eta"}
    for name in C.theta_names:
        if name not in summary.index:
            continue
        # every transform is monotone, so the diagnostics carry over unchanged
        key = natural.get(name, name)
        ess[key] = float(summary.loc[name, "ess_bulk"])
        r_hat[key] = float(summary.loc[name, "r_hat"])
    return {"ess": ess, "r_hat": r_hat}
