import numpy as np

#: The four model variants, in the order R lists them.
MODELS = ("plm0", "plm", "gplm0", "gplm")

FT_TO_M = 0.3048
CFS_TO_CMS = 0.028316846592

# --- sampling ------------------------------------------------------------------
# R's bdrc runs 4 chains x 20000 Metropolis iterations with a 2000 burn-in thinned
# by 5 (3601 draws per chain); NUTS needs far fewer.
NUM_CHAINS = 4
NUM_DRAWS = 1000
NUM_TUNE = 1000
ADVI_STEPS = 10_000

#: Which NUTS implementation walks the marginal posterior. Any of PyMC's backends
#: works unchanged: the marginal is an ordinary pytensor graph wrapped in a
#: ``pm.Potential``, so there is nothing sampler-specific about it. Compiling it is
#: worth more here than for a typical model, because the marginal is only 2-10
#: dimensional and cheap per evaluation -- so per-step interpreter overhead, not
#: arithmetic, is what dominates. Overridden per call; see
#: ``settings.NUTS_SAMPLER`` for the package-wide default.
NUTS_SAMPLER = "nutpie"

#: NUTS target acceptance rate on a sample of ordinary size.
TARGET_ACCEPT = 0.9

#: A sample of this many measurements or fewer is a *short record*: the marginal
#: posterior of a rating fitted to a handful of points has a thin ridge that NUTS
#: diverges on at a normal step size, so it is sampled at
#: :data:`SMALL_SAMPLE_TARGET_ACCEPT`, which forces a smaller step. Mirrors
#: ``settings.SMALL_SAMPLE_N`` / ``settings.SMALL_SAMPLE_TARGET_ACCEPT``, the
#: package-wide values, kept here because this backend carries its own defaults.
SMALL_SAMPLE_N = 5
SMALL_SAMPLE_TARGET_ACCEPT = 0.999

SEED = 42


def target_accept_for(n: int, default: float = TARGET_ACCEPT) -> float:
    """The target acceptance rate to sample `n` measurements with.

    :data:`SMALL_SAMPLE_TARGET_ACCEPT` for a short record, `default` otherwise.
    """
    return SMALL_SAMPLE_TARGET_ACCEPT if int(n) <= SMALL_SAMPLE_N else default

# --- priors (R: bdrc:::priors) -------------------------------------------------
# Hyperprior rate parameters and the Gaussian prior on (log a, b).
MU_A = 3.0
MU_B = 1.835
SIG_A = 3.0
P_AB = 0.0
NUGGET = 1e-8
LAMBDA_C = 2.0             # (h_min - c) ~ Exponential(2)
LAMBDA_SE = 28.78          # sigma_eps ~ Exponential(28.78)      (plm0, gplm0)
LAMBDA_SB = 5.405          # sigma_beta ~ Exponential(5.405)     (gplm0, gplm)
LAMBDA_PB = 3.988          # phi_beta                            (gplm0, gplm)
LAMBDA_ETA_1 = 28.78       # eta_1                               (plm, gplm)
LAMBDA_SETA = 8.62         # sigma_eta ~ Exponential(8.62)       (plm, gplm)
SIG_B_CONST_EXPONENT = 0.426   # prior sd of b when f(h) = b        (plm0, plm)
SIG_B_VARYING_EXPONENT = 0.01  # prior sd of b when f(h) = b+beta(h) (gplm0, gplm)

MAX_VARIANCE = 100.0       # R rejects any theta whose error variance exceeds this
MATERN_SMOOTHNESS = 2.5    # nu = 5/2, fixed in the paper
N_SPLINE_BASIS = 6         # cubic B-splines, 2 interior knots (plm, gplm)
MAX_STAGE_GAP_CM = 5.0     # prediction grid: fill any observed gap wider than 5 cm


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
