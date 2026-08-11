DIFFERENCES_FROM_R = """
Every way this module's numbers can differ from the R bdrc package (v2.0.1), and
why. `validate_against_r.py` checks each claim below; the figures quoted are what
it measures. Anything that does not depend on the sampler agrees to machine
precision -- in particular the marginal log-density itself, for all four models
with c known and inferred, matches R to 1e-13 relative or better.

Different by design
-------------------
1. Sampler. R walks the marginal posterior of theta with Metropolis: 4 chains
   x 20000 iterations, 2000 burn-in, thinned by 5, giving 14404 draws at ~30%
   acceptance.
   This module hands the identical marginal density to NUTS: 4 x 1000 draws after
   1000 tuning, no thinning. Same target, so posteriors agree to Monte Carlo error,
   but no individual draw corresponds to an R draw and any quantile moves in its
   last couple of significant figures from fit to fit.
   Consequence: acceptance_rate has no analogue and is not reported.

2. Draw-level randomness. The latent x, the kriged beta on the grid, and the
   predictive noise come from numpy's Generator, not R's RNG, so seeds do not
   correspond. The algorithm is bdrc's own (draw from the prior, then apply the
   Kalman-style correction) -- only the stream differs.

3. r_hat and effective sample size. R uses its own split-chain variogram estimator
   over (a, b, theta). This module uses ArviZ's rank-normalized r_hat and bulk ESS
   over the sampled theta only, because a and b here are exact conditional draws
   rather than MCMC output. Same quantities, different estimator; the transforms
   between the sampled and reported scales are monotone, so the diagnostics carry
   over unchanged. NUTS values are far higher for the same wall clock.

4. Model comparison. R's tournament() ranks models by WAIC, DIC, or posterior model
   probability from a harmonic-mean marginal-likelihood estimator. WAIC and DIC are
   ported to match R; the harmonic-mean estimator is not, since this package scores
   every rating with PSIS-LOO through metrics.py, which is the better estimator and
   the one the comparison table already uses.

5. Plotting. R's ggplot2 S3 methods are not ported; plots.py covers this.

Reproduced on purpose, though arguably bugs in R
------------------------------------------------
6. D_hat gives the phantom observation zero variance where the marginal density
   gives it sig_b^2 -- R's gplm0.calc_Dhat builds `diag(c(varr, 0))` while
   gplm0.cpp builds `diagmat(join_vert(varr, sig_b_sq))`. Kept, so DIC matches R.
   Affects gplm0 and gplm only.

7. sigma_eps_posterior folds in the forcepoint weight for plm/gplm (it is
   sqrt(epsilon * exp(B eta))) but not for plm0/gplm0 (there it is
   sqrt(exp(theta)), with epsilon dropped). So when forcepoint is set, the
   pointwise log-likelihood, WAIC and DIC use a different variance at the forced
   points than the likelihood the sampler saw -- and differently between the two
   model families. Both behaviours are kept as R has them. With no forcepoint
   (epsilon all 1) the distinction vanishes, which is the normal case here.

8. In gplm0 and gplm the prior on b is imposed twice: through Sig_ab, whose (2,2)
   entry is sig_b^2, and again as a phantom observation row Z = (0,1,0,...) with
   value mu_b and variance sig_b^2. With sig_b = 0.01 for these two models the
   effect is that b is pinned at 1.835 +/- ~0.014 and all curvature is carried by
   beta(h) -- b is not meaningfully estimated in gplm0/gplm. Reproduced as-is.

9. mcmc_summary uses bdrc's order-statistic rule -- sort, then take element
   floor(q*(size-1)) -- not an interpolated percentile. Kept so summaries match;
   np.percentile would interpolate and give slightly different bounds.

10. R computes log_lik_i as dnorm(log(q), log(exp(y_mean)), ...), round-tripping the
    posterior mean through exp and then log. This module stores
    rating_curve_mean_posterior in discharge units and takes its log the same way,
    so the same ~1e-16 round-trip is present rather than using y_mean directly.

Unavoidable, and measured to be immaterial
------------------------------------------
11. B-spline basis: transcribed term by term from R's B_splines rather than built
    with scipy.interpolate. Matches R to 7e-16, at the data and on the prediction
    grid, for both plm and gplm. The transcription is kept so the basis is provably
    the same one rather than merely a defensible cubic B-spline basis.

12. R hard-rejects any theta whose error variance exceeds 100 by returning the
    finite sentinel p = -1e9; a gradient sampler cannot use a finite sentinel there,
    so this returns -inf. The validator checks the two agree on *which* probes are
    rejected. NUTS never proposes into that region under these priors.

13. Kriging beta onto the grid needs a Cholesky factor of a Schur complement that is
    positive definite in exact arithmetic but can lose definiteness on a dense grid.
    R calls arma::chol and errors out; _stable_cholesky retries with a growing
    jitter and falls back to an eigendecomposition. Only reached where R would have
    failed outright.

14. Posterior mode. R uses optim(method="L-BFGS-B", hessian=TRUE) at its default
    tolerances with a finite-difference Hessian; this module uses scipy's L-BFGS-B
    with the exact pytensor gradient, tightened tolerances, and an exact pytensor
    Hessian. Measured: this module's mode is at a log-density greater than or equal
    to R's on every model and dataset (by up to 1.5e-7), so where the coordinates
    differ -- up to 3.4e-4 for gplm, whose ridge is flat -- it is R that stopped
    early. The Hessian enters only through the c_upper test (does the data lack
    near-zero-flow measurements); its first diagonal element agrees to 3e-6
    relative, nowhere near enough to flip that threshold.

15. Prediction grid. R's seq(from, to, by=0.05) fuzz and final clamp are replicated
    exactly in _r_seq_by, and seq(..., length.out=) matches np.linspace exactly
    (both pin both endpoints). With c known the grid is therefore bit-identical
    (5e-15 m). With c inferred the grid's lower end is the mode, so it inherits
    note 14: measured up to 1.6e-6 m, i.e. under two micrometres of stage.
    One structural difference: the interior gap fill and the below-data ladder drop
    their endpoints by index here, where R uses setdiff on exact float equality --
    same result, except that R would additionally drop an interior point that
    happened to equal an endpoint exactly.

16. WAIC and DIC carry more sampler noise than anything else reported. Their
    lppd part matches R closely (0.3-0.5% on the reference fits), but p_waic is a
    posterior *variance* of the pointwise log-likelihood and DIC's effective
    parameter count is a difference of two medians -- both are high-variance
    functionals that depend on which draws you got. Measured against R: WAIC gaps of
    0.24 (gplm0) and 0.34 (plm0), DIC gaps of 0.16 and 0.27, all on the deviance
    scale where differences under ~2 are conventionally not meaningful. Compare them
    absolutely, never relative to their own value -- WAIC near zero (plm0 sits at
    about -3) would make an ordinary wobble look like a 12% error.

17. Extreme hyperparameters. At theta far outside the posterior the covariance
    X Sig_x X' + Sig_eps becomes ill-conditioned (cond ~ 2.6e10 at the worst probe
    tested), and three algebraically identical numpy routes for the same log-density
    disagree by ~1.4 absolute out of 1.75e7. Agreement with R there is ~1e-7
    relative rather than 1e-13. This bounds nothing that matters -- the region is
    unreachable by the sampler -- but the validator scores it separately rather
    than hiding it in an averaged tolerance.
"""
