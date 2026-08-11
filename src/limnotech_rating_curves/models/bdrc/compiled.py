import logging

import numpy as np
import scipy.optimize as sopt

from . import graphs
from .design import components, prediction_stages

log = logging.getLogger(__name__)

#: One entry per (model, known_c). Keyed that way because those two facts are the
#: only things a graph's structure depends on.
_CACHE: dict = {}


class CompiledModel:
    """The compiled pytensor functions for one (model, known_c) pair.

    Attributes
    ----------
    data : ~limnotech_rating_curves.models.bdrc.graphs.SharedDataset
        The shared variables the functions read from.
    C : ~limnotech_rating_curves.models.bdrc.design.Components or None
        The dataset currently bound, or None before the first :meth:`bind`.
    """

    def __init__(self, model: str, known_c: bool):
        import pytensor
        import pytensor.tensor as pt

        self.data = graphs.SharedDataset(model, known_c)
        self.C = None
        self._hessian = None

        theta = pt.dvector("theta")
        self._theta = theta
        logp = graphs.marginal_logp_graph(theta, self.data)
        log_stage, varr, Sig_x, X, eta = graphs.design_graph(theta, self.data)
        options = dict(on_unused_input="ignore")
        log.debug("compiling the bdrc marginal for %s (c %s)", model,
                  "known" if known_c else "inferred")
        self.logp = pytensor.function([theta], logp, **options)
        self.logp_grad = pytensor.function(
            [theta], [logp, pt.grad(logp, theta)], **options)
        self.design = pytensor.function(
            [theta], [varr, Sig_x, X, log_stage] + ([eta] if eta is not None else []),
            **options)

    @property
    def model(self):
        return self.data.model

    @property
    def hessian(self):
        """The compiled Hessian of the marginal density, built on first use.

        Only :func:`c_upper_bound` and :func:`posterior_mode` ever evaluate it, and
        the Hessian graph is a large share of the compile time, so a fit that never
        asks for it never pays for it.
        """
        if self._hessian is None:
            import pytensor
            import pytensor.tensor as pt
            log.debug("compiling the bdrc Hessian for %s", self.model)
            self._hessian = pytensor.function(
                [self._theta],
                pt.hessian(graphs.marginal_logp_graph(self._theta, self.data),
                           self._theta),
                on_unused_input="ignore")
        return self._hessian

    def logp_graph(self, theta):
        """A fresh symbolic copy of the marginal density, on the same shared inputs.

        This is what the sampler wraps in a ``pm.Potential``: PyMC builds its own
        function from it, and because the graph carries no dataset shapes, the one
        it builds is structurally identical from fit to fit.
        """
        return graphs.marginal_logp_graph(theta, self.data)

    def bind(self, C) -> None:
        """Point the shared variables at a dataset, and keep it for the numpy side."""
        self.data.bind(C)
        self.C = C

    def design_at(self, theta):
        """(varr, Sig_x, X, log_stage, eta) as numpy arrays."""
        out = self.design(np.asarray(theta, float))
        varr, Sig_x, X, log_stage = out[:4]
        return varr, Sig_x, X, log_stage, (out[4] if len(out) > 4 else None)

    def __repr__(self):
        n = "unbound" if self.C is None else f"n={self.C.n}"
        return (f"CompiledModel({self.model}, c "
                f"{'known' if self.data.known_c else 'inferred'}, {n})")


def compiled_for(C) -> CompiledModel:
    """The compiled model for this dataset, from the cache, bound and ready.

    Parameters
    ----------
    C : ~limnotech_rating_curves.models.bdrc.design.Components

    Returns
    -------
    CompiledModel
        Compiled on the first call for this ``(model, known_c)`` in this process and
        reused afterwards. The returned object is **shared, not a copy**: binding a
        new dataset invalidates any earlier binding, which is why a fit binds once
        and then runs to completion before the next fit starts.
    """
    key = (C.model, C.known_c)
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = _CACHE[key] = CompiledModel(C.model, C.known_c)
    compiled.bind(C)
    return compiled


def clear_cache() -> int:
    """Drop every compiled graph, freeing what they hold. Returns how many went.

    Compilation is per process and the graphs are small, so this exists for tests
    and for long-lived processes that want the memory back, not for normal use.
    """
    count = len(_CACHE)
    _CACHE.clear()
    return count


def cached_models() -> tuple:
    """Which ``(model, known_c)`` graphs this process has compiled so far."""
    return tuple(sorted(_CACHE))


# ==============================================================================
# posterior mode (R: optim inside get_model_components) and the c upper bound
# ==============================================================================

def posterior_mode(compiled: CompiledModel) -> np.ndarray:
    """The marginal posterior mode, from theta = 0 by L-BFGS-B.

    R optimizes the same objective from the same start; the mode seeds the sampler
    and fixes the prediction grid's lower end. The convergence tolerances are
    tightened past scipy's defaults because the mode sets the grid, and a loose mode
    would shift every predicted stage.

    Only the mode. The curvature there is a separate call - see :func:`curvature_at`
    - because compiling a Hessian is a large share of a graph's compile time and
    exactly one caller needs it.
    """
    n_theta = len(compiled.C.theta_names)

    def objective(theta):
        value, gradient = compiled.logp_grad(theta)
        return -float(value), -np.asarray(gradient, float)

    result = sopt.minimize(objective, np.zeros(n_theta), jac=True, method="L-BFGS-B",
                           options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20_000})
    return np.asarray(result.x, float)


def curvature_at(compiled: CompiledModel, theta) -> np.ndarray:
    """Minus the Hessian of the marginal density at `theta` - the precision matrix.

    Compiles the Hessian graph on first use for this model, which is why it is not
    folded into :func:`posterior_mode`: :func:`c_upper_bound` is the only caller, so
    on a run that never needs the bound the Hessian is never built.
    """
    return -np.asarray(compiled.hessian(np.asarray(theta, float)), float)


def c_upper_bound(y_obs, stage, forcepoint):
    """When the data has no measurements near zero flow, bdrc bounds c from above
    rather than letting it drift: fit plm0, take the mode and sd of
    log(h_min - c), and if the 2.5% end of that interval still implies more than a
    2 m drop below the lowest stage, cap c there. Returns None when not triggered.
    R raises a warning here; this returns the bound and lets the caller warn.

    The plm0 graph this needs is the cached one, so on any run that has already
    fitted a plm0 (or has done this once before) it costs no compilation at all."""
    plm0 = components("plm0", y_obs, stage, None, forcepoint)
    compiled = compiled_for(plm0)
    mode = posterior_mode(compiled)
    log_drop_sd = np.sqrt(np.diag(np.linalg.inv(curvature_at(compiled, mode))))[0]
    lower_drop = np.exp(mode[0] - 1.96 * log_drop_sd)
    return (stage.min() - lower_drop) if lower_drop > 2 else None


def finalize_grid(compiled: CompiledModel, h_max) -> None:
    """Attach the prediction grid to the components, once the mode is known."""
    from .design import b_splines, distance_matrix

    C = compiled.C
    # the mode only: C.hessian stays None because nothing downstream reads it, and
    # building it would compile a Hessian graph on every fit for nothing
    C.theta_mode = posterior_mode(compiled)
    h_min_pred = C.c_param if C.known_c else C.h_min - np.exp(C.theta_mode[0])
    h_max_pred = C.h_max_data if h_max is None else float(h_max)
    if h_max_pred < C.h_max_data:
        raise ValueError("h_max must be at least the largest observed stage, "
                         f"which is {C.h_max_data} m")
    C.h_u = prediction_stages(C.stage, C.h_min_data, C.h_max_data,
                              h_min_pred, h_max_pred)
    if C.varying_variance:
        span = C.h_max_data - C.h_min_data
        standardized = np.clip((C.h_u - C.h_min_data) / span, 0.0, 1.0)
        C.B_u = b_splines(standardized)
    if C.varying_exponent:
        C.dist_all = distance_matrix(np.concatenate((C.h_unique, C.h_u)))
