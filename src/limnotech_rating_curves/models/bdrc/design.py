from dataclasses import dataclass

import numpy as np

from .defaults import (MAX_STAGE_GAP_CM, MU_A, MU_B, N_SPLINE_BASIS,
                       prior_covariance, theta_names, varying_exponent,
                       varying_variance)


def b_splines(z):
    """The six cubic B-spline basis functions bdrc uses for log variance, evaluated
    at `z` in [0, 1]. Returns (len(z), 6).

    A literal transcription of R's bdrc:::B_splines: order-4 splines on two equally
    spaced interior knots, written out term by term. Kept as an explicit
    transcription rather than a scipy.interpolate.BSpline call so the basis is
    provably the same one R uses (see DIFFERENCES_FROM_R note 11)."""
    z = np.asarray(z, float).ravel()
    n_interior = 2                              # kx
    dx = 1.0 / (n_interior + 1)
    order = 4                                   # M
    knots = dx * np.arange(n_interior + 2)      # epsilon-knots

    # tau, the padded knot vector, indexed 1-based as in the R source
    tau_vec = np.empty(n_interior + 2 * order)
    tau_vec[:order] = knots[0]
    tau_vec[order:n_interior + order] = knots[1:n_interior + 1]
    tau_vec[n_interior + order:] = knots[n_interior + 1]
    tau = lambda i: tau_vec[i - 1]

    kx, M = n_interior, order
    inside = lambda lo, hi, closed=False: (
        (tau(lo) <= z) * ((z <= tau(hi)) if closed else (z < tau(hi)))).astype(float)
    basis = np.zeros((N_SPLINE_BASIS, z.size))

    basis[0] = (1 / dx**3) * (tau(M+1) - z)**3 * inside(M, M+1)

    basis[1] = ((1 / dx**3) * (z - tau(2)) * (tau(M+1) - z)**2 * inside(M, M+1)
                + (1/2 / dx**3) * (tau(M+2) - z) * (z - tau(3)) * (tau(M+1) - z) * inside(M, M+1)
                + (1/4 / dx**3) * (tau(M+2) - z)**2 * (z - tau(M)) * inside(M, M+1)
                + (1/4 / dx**3) * (tau(M+2) - z)**3 * inside(M+1, M+2))

    basis[2] = ((1/2 / dx**3) * (z - tau(3))**2 * (tau(M+1) - z) * inside(M, M+1)
                + (1/4 / dx**3) * (z - tau(3)) * (tau(M+2) - z) * (z - tau(M)) * inside(M, M+1)
                + (1/4 / dx**3) * (z - tau(3)) * (tau(M+2) - z)**2 * inside(M+1, M+2)
                + (1/6 / dx**3) * (tau(M+3) - z) * (z - tau(M))**2 * inside(M, M+1)
                + (1/6 / dx**3) * (tau(M+3) - z) * (z - tau(M)) * (tau(M+2) - z) * inside(M+1, M+2)
                + (1/6 / dx**3) * (tau(M+3) - z)**2 * (z - tau(M+1)) * inside(M+1, M+2)
                + (1/6 / dx**3) * (tau(M+3) - z)**3 * inside(M+2, M+3))

    basis[kx+1] = (-(1/6 / dx**3) * (tau(kx+2) - z)**3 * inside(kx+2, kx+3)
                   - (1/6 / dx**3) * (tau(kx+2) - z)**2 * (z - tau(kx+4)) * inside(kx+3, kx+4)
                   - (1/6 / dx**3) * (tau(kx+2) - z) * (z - tau(kx+5)) * (tau(kx+3) - z) * inside(kx+3, kx+4)
                   - (1/6 / dx**3) * (tau(kx+2) - z) * (z - tau(kx+5))**2 * inside(kx+4, kx+5)
                   - (1/4 / dx**3) * (z - tau(kx+6)) * (tau(kx+3) - z)**2 * inside(kx+3, kx+4)
                   - (1/4 / dx**3) * (z - tau(kx+6)) * (tau(kx+3) - z) * (z - tau(kx+5)) * inside(kx+4, kx+5)
                   - (1/2 / dx**3) * (z - tau(kx+6))**2 * (tau(kx+4) - z) * inside(kx+4, kx+5))

    basis[kx+2] = (-(1/4 / dx**3) * (tau(kx+3) - z)**3 * inside(kx+3, kx+4)
                   - (1/4 / dx**3) * (tau(kx+3) - z)**2 * (z - tau(kx+5)) * inside(kx+4, kx+5)
                   - (1/2 / dx**3) * (tau(kx+3) - z) * (z - tau(kx+6)) * (tau(kx+4) - z) * inside(kx+4, kx+5)
                   - (1 / dx**3) * (z - tau(kx+7)) * (tau(kx+4) - z)**2 * inside(kx+4, kx+5))

    basis[kx+3] = -(1 / dx**3) * (tau(kx+4) - z)**3 * inside(kx+4, kx+5, closed=True)

    return basis.T


def unique_stage_matrix(stage):
    """The n x n_unique indicator mapping each (ascending) observation to its unique
    stage - R's bdrc:::create_A_cpp."""
    stage = np.asarray(stage, float)
    h_unique = np.unique(stage)
    A = np.zeros((stage.size, h_unique.size))
    A[np.arange(stage.size), np.searchsorted(h_unique, stage)] = 1.0
    return A


def distance_matrix(x):
    x = np.asarray(x, float)
    return np.abs(x[:, None] - x[None, :])


def matern52_correlation(dist, range_param):
    """Matern correlation with smoothness nu = 5/2 and range `range_param` (phi_beta)."""
    root5 = np.sqrt(5.0)
    scaled = root5 * dist / range_param
    return (1.0 + scaled + 5.0 * dist**2 / (3.0 * range_param**2)) * np.exp(-scaled)


def _r_seq_by(start, stop, step):
    """R's seq(from, to, by=) - including its 1e-10 length fuzz and the final
    pmin(x, to) clamp. Replicated exactly because the prediction grid below the
    data depends on it."""
    count = int((stop - start) / step + 1e-10)
    values = start + np.arange(count + 1) * step
    return np.minimum(values, stop) if step > 0 else np.maximum(values, stop)


def prediction_stages(stage_sorted, h_min_data, h_max_data, h_min_pred, h_max_pred):
    """The unobserved stages bdrc adds to the rating curve - R's bdrc:::h_unobserved.

    Three pieces, concatenated in R's order (not sorted): a 5 cm ladder from
    `h_min_pred` (the stage of zero flow) up to the lowest observation, fill points
    inside any observed gap wider than 5 cm, and an extension above the data up to
    `h_max_pred`. The interior fill works in centimetres, as R does."""
    stage_cm = 100.0 * np.asarray(stage_sorted, float)
    # gap to the next observation, with R's dummy 1000 cm sentinel after the last
    next_cm = np.append(stage_cm[1:], 1000.0)
    gap_cm = np.abs(stage_cm - next_cm)
    wide = gap_cm > MAX_STAGE_GAP_CM
    # the sentinel gap is always wide and is dropped, as in R
    lower_cm, span_cm, upper_cm = stage_cm[wide][:-1], gap_cm[wide][:-1], next_cm[wide][:-1]

    interior = []
    for lo, span, hi in zip(lower_cm, span_cm, upper_cm):
        n_points = 2 + int(np.ceil(span / MAX_STAGE_GAP_CM))
        interior.append(np.linspace(lo, hi, n_points)[1:-1])
    interior = 0.01 * np.concatenate(interior) if interior else np.empty(0)

    below = _r_seq_by(h_min_pred, h_min_data, 0.05)
    below = below[below != h_min_data]
    n_above = 2 + int(np.ceil(20.0 * (h_max_pred - h_max_data)))
    above = np.linspace(h_max_data, h_max_pred, n_above)
    above = above[above != h_max_data]
    return np.concatenate((below, interior, above))


# ==============================================================================
# model components (R: bdrc:::get_model_components)
# ==============================================================================

@dataclass
class Components:
    """Everything about one model + dataset that does not depend on theta.

    The numeric side of a fit. The pytensor graphs in
    :mod:`~limnotech_rating_curves.models.bdrc.graphs` read the same arrays out of
    shared variables instead, so a graph can be compiled once and re-bound to any
    dataset; :func:`~limnotech_rating_curves.models.bdrc.compiled.compiled_for`
    is what copies one of these into them.
    """
    model: str
    y_obs: np.ndarray          # log discharge, ascending by stage
    stage: np.ndarray          # stage, ascending
    y: np.ndarray              # y_obs, plus the phantom observation for gplm0/gplm
    n: int
    c_param: float | None      # known stage of zero flow, or None
    h_min: float               # the origin l = log(h - h_min + exp(zeta)) is measured from
    h_min_data: float
    h_max_data: float
    epsilon: np.ndarray        # per-observation variance weight (forcepoint)
    mu_x: np.ndarray
    Sig_ab: np.ndarray
    theta_names: tuple
    # gplm0 / gplm only
    h_unique: np.ndarray = None
    n_unique: int = 0
    A: np.ndarray = None
    dist: np.ndarray = None
    Z: np.ndarray = None
    # plm / gplm only
    P: np.ndarray = None
    B: np.ndarray = None
    # prediction grid, filled in after the mode is found
    h_u: np.ndarray = None
    B_u: np.ndarray = None
    dist_all: np.ndarray = None
    theta_mode: np.ndarray = None
    # curvature at the mode, filled in only if a caller asks for it
    hessian: np.ndarray = None
    c_upper: float | None = None

    @property
    def varying_exponent(self):
        return varying_exponent(self.model)

    @property
    def varying_variance(self):
        return varying_variance(self.model)

    @property
    def known_c(self):
        return self.c_param is not None


def components(model, y_obs, stage, c_param, forcepoint, h_min=None) -> Components:
    """Assemble the theta-independent pieces for one model and dataset."""
    n = stage.size
    epsilon = np.ones(n)
    epsilon[forcepoint] = 1.0 / n

    C = Components(
        model=model, y_obs=y_obs, stage=stage, y=y_obs.copy(), n=n, c_param=c_param,
        h_min=stage.min() if h_min is None else h_min,
        h_min_data=stage.min(), h_max_data=stage.max(), epsilon=epsilon,
        mu_x=np.array([MU_A, MU_B]), Sig_ab=prior_covariance(model),
        theta_names=theta_names(model, c_param is not None))

    if C.varying_variance:
        C.P = np.tril(np.ones((N_SPLINE_BASIS, N_SPLINE_BASIS)))
        centred = stage - stage.min()
        C.B = b_splines(centred / centred[-1])
    if C.varying_exponent:
        # a phantom observation b ~ N(mu_b, sig_b^2), appended as an extra data row
        C.y = np.append(y_obs, MU_B)
        C.h_unique = np.unique(stage)
        C.n_unique = C.h_unique.size
        C.A = unique_stage_matrix(stage)
        C.dist = distance_matrix(C.h_unique)
        C.mu_x = np.concatenate(([MU_A, MU_B], np.zeros(C.n_unique)))
        C.Z = np.concatenate(([[0.0, 1.0]], np.zeros((1, C.n_unique))), axis=1)
    return C
