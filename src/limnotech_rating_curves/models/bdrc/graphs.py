import numpy as np

from .defaults import (LAMBDA_C, LAMBDA_ETA_1, LAMBDA_PB, LAMBDA_SB, LAMBDA_SE,
                       LAMBDA_SETA, MAX_VARIANCE, MU_A, MU_B, N_SPLINE_BASIS,
                       NUGGET, prior_covariance, theta_names, varying_exponent,
                       varying_variance)

#: R's ``P``: lower-triangular ones, which turns the spline weights into their
#: cumulative sums. A constant of the model, so it stays inlined - and it stays a
#: matrix product rather than a ``cumsum`` so the arithmetic is bit-for-bit what R
#: does.
_CUMULATIVE_SUM = np.tril(np.ones((N_SPLINE_BASIS, N_SPLINE_BASIS)))


class SharedDataset:
    """The dataset a compiled bdrc graph is bound to, as pytensor shared variables.

    One of these belongs to each compiled graph. Its arrays start as placeholders of
    the right dimensionality and no fixed size - ``shape=(None,) * ndim`` is what
    keeps the compiled function shape-agnostic - and :meth:`bind` swaps in a real
    dataset without touching the graph.

    Parameters
    ----------
    model : {'plm0', 'plm', 'gplm0', 'gplm'}
        Which variant, which fixes the shape of the algebra (does the exponent vary
        with stage, does the variance) and the prior covariance of ``(log a, b)``.
    known_c : bool
        Whether the stage of zero flow is supplied rather than estimated. It changes
        the hyperparameter vector, so it is part of a graph's identity.
    """

    def __init__(self, model: str, known_c: bool):
        import pytensor

        self.model = model
        self.known_c = known_c
        self.theta_names = theta_names(model, known_c)
        self.Sig_ab = prior_covariance(model)      # a property of the model alone

        def free(value):
            """A shared array whose size stays symbolic."""
            value = np.asarray(value, float)
            return pytensor.shared(value, shape=(None,) * value.ndim)

        # Placeholder sizes are arbitrary but must exceed 1 in every dimension: a
        # length-1 axis would be marked broadcastable and then refuse a real dataset.
        self.stage = free(np.zeros(3))
        self.y = free(np.zeros(3))
        self.epsilon = free(np.ones(3))
        self.c_param = pytensor.shared(np.array(0.0))
        self.h_min = pytensor.shared(np.array(0.0))
        self.A = free(np.zeros((3, 2))) if self.varying_exponent else None
        self.dist = free(np.zeros((2, 2))) if self.varying_exponent else None
        self.B = free(np.zeros((3, 6))) if self.varying_variance else None

    @property
    def varying_exponent(self):
        return varying_exponent(self.model)

    @property
    def varying_variance(self):
        return varying_variance(self.model)

    def bind(self, C) -> None:
        """Point this graph at a dataset.

        Parameters
        ----------
        C : ~limnotech_rating_curves.models.bdrc.design.Components
            The dataset to fit. Must be the same model and the same known-or-inferred
            ``c`` as this graph was built for, which
            :func:`~limnotech_rating_curves.models.bdrc.compiled.compiled_for`
            guarantees by construction.
        """
        if C.model != self.model or C.known_c != self.known_c:
            raise ValueError(
                f"this graph was compiled for {self.model} with c "
                f"{'known' if self.known_c else 'inferred'}, so it cannot be bound "
                f"to {C.model} with c {'known' if C.known_c else 'inferred'}")
        self.stage.set_value(np.asarray(C.stage, float))
        self.y.set_value(np.asarray(C.y, float))
        self.epsilon.set_value(np.asarray(C.epsilon, float))
        self.c_param.set_value(np.array(0.0 if C.c_param is None else float(C.c_param)))
        self.h_min.set_value(np.array(float(C.h_min)))
        if self.varying_exponent:
            self.A.set_value(np.asarray(C.A, float))
            self.dist.set_value(np.asarray(C.dist, float))
        if self.varying_variance:
            self.B.set_value(np.asarray(C.B, float))


def _matern52_correlation_graph(dist, range_param):
    import pytensor.tensor as pt
    root5 = np.sqrt(5.0)
    scaled = root5 * dist / range_param
    return (1.0 + scaled + 5.0 * dist**2 / (3.0 * range_param**2)) * pt.exp(-scaled)


def _prior_mean(data: SharedDataset):
    """``mu_x``, the prior mean of the latent vector, at whatever size the data is.

    ``(mu_a, mu_b)`` for the plain power laws, extended with a zero mean for
    ``beta`` at each unique stage when the exponent varies."""
    import pytensor.tensor as pt
    head = np.array([MU_A, MU_B])
    if not data.varying_exponent:
        return pt.as_tensor_variable(head)
    return pt.concatenate([head, pt.zeros((data.dist.shape[0],))])


def design_graph(theta, data: SharedDataset):
    """Symbolic (l, varr, Sig_x, X, eta) at one hyperparameter vector."""
    import pytensor.tensor as pt

    take = dict(zip(data.theta_names, [theta[i] for i in range(len(data.theta_names))]))
    if data.known_c:
        log_stage = pt.log(data.stage - data.c_param)
    else:
        log_stage = pt.log(data.stage - data.h_min + pt.exp(take["zeta"]))

    n = data.stage.shape[0]

    if data.varying_variance:
        z = pt.stack([take[f"z_{i}"] for i in range(1, 6)])
        eta = _CUMULATIVE_SUM @ pt.concatenate(
            [[take["eta_1"]], pt.exp(take["log_sigma_eta"]) * z])
        varr = data.epsilon * pt.exp(data.B @ eta)
    else:
        eta = None
        varr = data.epsilon * pt.exp(take["log_sigma_eps2"])

    if data.varying_exponent:
        n_unique = data.dist.shape[0]
        correlation = (_matern52_correlation_graph(data.dist,
                                                  pt.exp(take["log_phi_beta"]))
                       + NUGGET * pt.eye(n_unique))
        Sig_x = pt.set_subtensor(
            pt.set_subtensor(pt.zeros((n_unique + 2, n_unique + 2))[:2, :2],
                             data.Sig_ab)[2:, 2:],
            pt.exp(2 * take["log_sigma_beta"]) * correlation)
        # the phantom observation row Z = (0, 1, 0, ..., 0)
        phantom = pt.concatenate([np.array([[0.0, 1.0]]), pt.zeros((1, n_unique))],
                                 axis=1)
        X = pt.concatenate([
            pt.concatenate([pt.ones((n, 1)), log_stage[:, None],
                            log_stage[:, None] * data.A], axis=1),
            phantom], axis=0)
    else:
        Sig_x = pt.as_tensor_variable(data.Sig_ab)
        X = pt.concatenate([pt.ones((n, 1)), log_stage[:, None]], axis=1)
    return log_stage, varr, Sig_x, X, eta


def _observation_covariance(varr, data: SharedDataset):
    """Sig_eps: the observation variances, plus the phantom row's own variance."""
    import pytensor.tensor as pt
    if not data.varying_exponent:
        return pt.diag(varr)
    return pt.diag(pt.concatenate([varr, [data.Sig_ab[1, 1]]]))


def log_prior_graph(theta, data: SharedDataset):
    """log p(theta) on the sampled (unconstrained) scale, Jacobians included - the
    sum of R's bdrc:::pri terms for this model. Normalizing constants are dropped,
    exactly as R drops them."""
    import pytensor.tensor as pt
    take = dict(zip(data.theta_names, [theta[i] for i in range(len(data.theta_names))]))
    terms = []
    if not data.known_c:
        terms.append(take["zeta"] - pt.exp(take["zeta"]) * LAMBDA_C)
    if data.varying_variance:
        terms += [
            0.5 * take["eta_1"] - pt.exp(0.5 * take["eta_1"]) * LAMBDA_ETA_1,
            -0.5 * sum(take[f"z_{i}"]**2 for i in range(1, 6)),
            take["log_sigma_eta"] - pt.exp(take["log_sigma_eta"]) * LAMBDA_SETA,
        ]
    else:
        terms.append(0.5 * take["log_sigma_eps2"]
                     - pt.exp(0.5 * take["log_sigma_eps2"]) * LAMBDA_SE)
    if data.varying_exponent:
        terms += [
            take["log_sigma_beta"] - pt.exp(take["log_sigma_beta"]) * LAMBDA_SB,
            -0.5 * take["log_phi_beta"]
            - LAMBDA_PB * np.sqrt(0.5) * pt.exp(-0.5 * take["log_phi_beta"]),
        ]
    return sum(terms)


def marginal_logp_graph(theta, data: SharedDataset):
    """log p(theta | y) up to a constant: the Gaussian marginal likelihood with the
    latent level integrated out, plus the hyperpriors."""
    import pytensor.tensor as pt
    _, varr, Sig_x, X, _ = design_graph(theta, data)
    M = X @ Sig_x @ X.T + _observation_covariance(varr, data)
    M = M + NUGGET * pt.eye(data.y.shape[0])
    L = pt.linalg.cholesky(M, on_error="nan")
    w = pt.linalg.solve_triangular(L, data.y - X @ _prior_mean(data), lower=True)
    quadratic = -0.5 * pt.dot(w, w) - pt.sum(pt.log(pt.diag(L)))
    # R hard-rejects any theta whose error variance exceeds MAX_VARIANCE; the same
    # guard here would be non-differentiable, so it is expressed as -inf instead
    # (see DIFFERENCES_FROM_R note 12).
    ok = pt.all(varr <= MAX_VARIANCE)
    return pt.switch(ok, quadratic + log_prior_graph(theta, data), -np.inf)
