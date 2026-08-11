# limnotech_rating_curves.models.bdrc.__init__

Bayesian discharge rating curves: a native PyMC port of the R `bdrc` package.

Fits the power-law and generalized power-law rating curves of Hrafnkelsson et al.
(2022), in the same four variants the R package exposes:

    plm0   Q = a(h-c)^b            constant exponent, constant variance
    plm    Q = a(h-c)^b            constant exponent, variance varying with stage
    gplm0  Q = a(h-c)^f(h)         exponent f(h) = b + beta(h), constant variance
    gplm   Q = a(h-c)^f(h)         exponent f(h) = b + beta(h), variance varying

On a log scale the model is linear-Gaussian in the latent vector
x = (log a, b, beta(h)), conditional on the hyperparameters theta:

    y = log(Q) = X(theta) x + eps,   eps ~ N(0, diag(varr(theta)))
    x ~ N(mu_x, Sig_x(theta))

which is what makes the port tractable. The latent level integrates out
analytically, leaving a marginal posterior over only 2-10 hyperparameters:

    log p(theta | y) = log N(y; X mu_x, X Sig_x X' + Sig_eps) + log p(theta)

R's bdrc walks that marginal with Metropolis, 4 chains x 20000 iterations
thinned by 5. Here the same marginal is handed to NUTS, which has its gradient
from pytensor. Everything downstream is closed form and needs no sampler at all:
the latent x given theta is a Gaussian conditional, and beta at unobserved stages
is a Gaussian-process (kriging) conditional. So the port is one pytensor
log-density plus numpy post-processing.

Value equivalence with R. The marginal log-density, priors, design matrices,
prediction grid, posterior summaries, pointwise log-likelihood, WAIC and DIC are
transcribed from the R/C++ source rather than re-derived. Places where this port
deliberately or unavoidably differs from R are collected in
`DIFFERENCES_FROM_R`, including bdrc's own internal inconsistencies that are
reproduced on purpose so the numbers match.

Units. bdrc's priors are tuned to SI (stage m, discharge m^3/s), so the SI
interface is `fit`. `fit_predict` is the ft/cfs entry point the rest of
this package calls: it converts to SI for the fit and back to cfs for the curve.

    from limnotech_rating_curves.models import bdrc

    fitted = bdrc.fit(discharge_cms, stage_m, model="gplm0")
    fitted.predict(np.linspace(8.0, 10.0, 50))    # h, lower, median, upper
    fitted.param_summary                          # a, b, c, sigma_eps, ...

## Where the code lives

```
==============  ================================================================
:mod:`defaults`   the priors from the paper, and the sampling budget
:mod:`design`     the theta-independent pieces: B-splines, the unique-stage
                  indicator, the prediction grid, :class:`~design.Components`
:mod:`graphs`     the marginal log-density as one pytensor graph over *shared*
                  dataset inputs, which is what makes it reusable
:mod:`compiled`   the compile cache: eight functions serve every fit in a
                  process, plus the posterior mode and the bound on ``c``
:mod:`sampling`   NUTS or ADVI over the hyperparameters
:mod:`posterior`  the closed-form latent and predictive draws, and
                  :class:`~posterior.BdrcFit`
:mod:`criteria`   pointwise log-likelihood, WAIC, DIC
:mod:`fitting`    :func:`fit` and :func:`fit_predict`
==============  ================================================================
```

Compilation cost. A fit is dominated by compiling the density, not by the arithmetic, so the
graph is built over shared variables with symbolic shapes and cached per
`(model, c known or inferred)`. The first bdrc fit in a process pays the compile;
every later fit reuses it, at any site, any sample size and any
cross-validation fold.
See `PERFORMANCE.md`.

References
    Hrafnkelsson, B., Sigurdarson, H., Rognvaldsson, S., Jansson, A. O., Vias, R. D.,
    and Gardarsson, S. M. (2022). Generalization of the power-law rating curve using
    hydrodynamic theory and Bayesian hierarchical modeling. Environmetrics 33(2):e2711.
