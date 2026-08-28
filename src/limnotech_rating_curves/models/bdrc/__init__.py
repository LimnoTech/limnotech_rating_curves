import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# MKL 2026 + numpy 2.4 on this machine segfault in MKL's threaded BLAS path;
# sequential threading avoids it and costs nothing at these sizes.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from . import compiled, criteria, defaults, design, graphs, posterior, sampling
from .compiled import (CompiledModel, c_upper_bound, cached_models,
                       clear_cache, compiled_for, curvature_at,
                       posterior_mode)
from .criteria import pointwise_log_likelihood, waic_from_log_likelihood
from .defaults import MODELS, target_accept_for, theta_names
from .design import (Components, b_splines, components, prediction_stages,
                     unique_stage_matrix)
from .differences import DIFFERENCES_FROM_R
from .fitting import fit, fit_predict
from .graphs import SharedDataset, marginal_logp_graph
from .posterior import BdrcFit, mcmc_summary

__all__ = [
    # fitting
    "fit", "fit_predict", "BdrcFit", "MODELS",
    # the compile cache
    "compiled_for", "CompiledModel", "SharedDataset", "cached_models", "clear_cache",
    # pieces worth reaching for directly
    "b_splines", "unique_stage_matrix", "prediction_stages", "mcmc_summary",
    "pointwise_log_likelihood", "waic_from_log_likelihood", "marginal_logp_graph",
    "posterior_mode", "curvature_at", "c_upper_bound", "components",
    "Components", "theta_names",
    # constants
    "target_accept_for",
    "DIFFERENCES_FROM_R",
    # modules
    "defaults", "design", "graphs", "compiled", "sampling", "posterior", "criteria",
]
