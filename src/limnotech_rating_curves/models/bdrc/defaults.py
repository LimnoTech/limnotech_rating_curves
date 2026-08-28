"""Model identity and the small functions that read the bdrc settings.

The numbers themselves - sampler budgets, priors, the spline basis size - live in
:mod:`limnotech_rating_curves.settings` under the ``BDRC_`` prefix, with every one
overridable from the environment. This module imports them under the short names R's
``bdrc`` and the paper use, so the graphs and design matrices below read the way the
paper writes them.
"""

import numpy as np

from ... import settings
from ...settings import (BDRC_LAMBDA_C as LAMBDA_C,
                         BDRC_LAMBDA_ETA_1 as LAMBDA_ETA_1,
                         BDRC_LAMBDA_PB as LAMBDA_PB,
                         BDRC_LAMBDA_SB as LAMBDA_SB,
                         BDRC_LAMBDA_SE as LAMBDA_SE,
                         BDRC_LAMBDA_SETA as LAMBDA_SETA,
                         BDRC_MATERN_SMOOTHNESS as MATERN_SMOOTHNESS,
                         BDRC_MAX_STAGE_GAP_CM as MAX_STAGE_GAP_CM,
                         BDRC_MAX_VARIANCE as MAX_VARIANCE,
                         BDRC_MU_A as MU_A,
                         BDRC_MU_B as MU_B,
                         BDRC_NUGGET as NUGGET,
                         BDRC_N_SPLINE_BASIS as N_SPLINE_BASIS,
                         BDRC_P_AB as P_AB,
                         BDRC_SIG_A as SIG_A,
                         BDRC_SIG_B_CONST_EXPONENT as SIG_B_CONST_EXPONENT,
                         BDRC_SIG_B_VARYING_EXPONENT as SIG_B_VARYING_EXPONENT)

#: The four model variants, in the order R lists them.
MODELS = ("plm0", "plm", "gplm0", "gplm")


def target_accept_for(n: int, default: float = None) -> float:
    """The target acceptance rate to sample `n` measurements with.

    ``settings.BDRC_SMALL_SAMPLE_TARGET_ACCEPT`` for a short record, `default`
    otherwise. A short record's marginal posterior has a thin ridge that NUTS
    diverges on at a normal step size.
    """
    default = settings.BDRC_TARGET_ACCEPT if default is None else default
    return (settings.BDRC_SMALL_SAMPLE_TARGET_ACCEPT
            if int(n) <= settings.BDRC_SMALL_SAMPLE_N else default)


def varying_exponent(model: str) -> bool:
    """Does this model let the exponent vary with stage - is it a *g*plm?"""
    return model in ("gplm0", "gplm")


def varying_variance(model: str) -> bool:
    """Does this model let the error variance vary with stage?"""
    return model in ("plm", "gplm")


def prior_covariance(model: str) -> np.ndarray:
    """``Sig_ab``, the 2x2 Gaussian prior covariance of ``(log a, b)``.

    A property of the model alone, not of the data: ``b`` gets a wide prior when it
    is the whole exponent and a very tight one when ``beta(h)`` carries the
    curvature (see note 8 in
    :data:`~limnotech_rating_curves.models.bdrc.DIFFERENCES_FROM_R`).
    """
    sig_b = (SIG_B_VARYING_EXPONENT if varying_exponent(model)
             else SIG_B_CONST_EXPONENT)
    return np.array([[SIG_A ** 2, P_AB * SIG_A * sig_b],
                     [P_AB * SIG_A * sig_b, sig_b ** 2]])


def theta_names(model: str, known_c: bool) -> tuple:
    """The sampled hyperparameters, in R's order.

    With the stage of zero flow known, ``zeta = log(h_min - c)`` is not sampled and
    drops out of the front of the vector.
    """
    if model == "plm0":
        names = ("zeta", "log_sigma_eps2")
    elif model == "gplm0":
        names = ("zeta", "log_sigma_eps2", "log_sigma_beta", "log_phi_beta")
    elif model == "plm":
        names = ("zeta", "log_sigma_eta", "eta_1") + tuple(f"z_{i}" for i in range(1, 6))
    elif model == "gplm":
        names = (("zeta", "log_sigma_beta", "log_phi_beta", "log_sigma_eta", "eta_1")
                 + tuple(f"z_{i}" for i in range(1, 6)))
    else:
        raise ValueError(f"unknown model: {model!r} (expected one of {MODELS})")
    return names[1:] if known_c else names
