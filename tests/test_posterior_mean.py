"""``posterior_mean`` must report the rating, not the rating inflated by the scatter.

The models fit in log space and report cfs. A measurement error that is symmetric in
logs is not symmetric in cfs, so averaging the posterior predictive - which is what
``predict`` does - returns the rating multiplied by ``exp(sigma ** 2 / 2)`` rather than
the rating. ``posterior_mean`` averages the rating itself.

The decisive check is :func:`test_median_matches_ratingcurves_own_draws`. A
symmetric error leaves the median where it is, so the median of our reconstructed
rating draws must equal the median of ``predict_posterior``. That validates the
reconstruction of the mean function against ratingcurve's own output rather than
against our arithmetic.

See ``posterior_clarification.md``.
"""

import numpy as np
import pandas as pd
import pytest

import limnotech_rating_curves as lrc

#: Enough measurements, and a tight enough scatter, that the fit is well determined and
#: the two estimators differ by a knowable amount rather than by noise.
N_MEASUREMENTS = 30
TRUE_SCATTER = 0.06


def _measurements(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stage = np.sort(rng.uniform(2.0, 9.0, N_MEASUREMENTS))
    discharge = (180 * (stage - 1.2) ** 1.6
                 * np.exp(rng.normal(0, TRUE_SCATTER, N_MEASUREMENTS)))
    return pd.DataFrame({
        "stage_ft": stage, "discharge_cfs": discharge,
        "time": pd.date_range("2020-01-01", periods=N_MEASUREMENTS, freq="11D")})


def _fit(model):
    frame = _measurements()
    return model.fit(frame, stage="stage_ft", discharge="discharge_cfs", time="time",
                     method="nuts", seed=42)


@pytest.fixture(scope="module")
def power_law():
    return _fit(lrc.PowerLaw(segments=1))


@pytest.fixture(scope="module")
def spline():
    return _fit(lrc.Spline())


@pytest.fixture(scope="module")
def stages():
    return np.linspace(2.5, 8.5, 7)


@pytest.mark.slow
@pytest.mark.parametrize("family", ["power_law", "spline"])
def test_median_matches_ratingcurves_own_draws(family, stages, request):
    """The reconstructed mean function, checked against ratingcurve's own sampling.

    Adding a symmetric log-space error changes the mean but not the median, so these
    two sets of draws must share a median. Any error in reconstructing ``mu`` - a
    mistransposed axis, a missed standardization, the wrong ``ho`` - moves the median
    and fails here.
    """
    rating = request.getfixturevalue(family).result.rating

    ours = np.asarray(rating.posterior_draws(stages), float)
    theirs = np.asarray(
        rating.model.predict_posterior(stages, extend_idata=False), float)

    assert ours.shape == theirs.shape, "draws should match predict_posterior's shape"

    relative = np.abs(np.median(ours, axis=1) - np.median(theirs, axis=1))
    relative /= np.median(theirs, axis=1)

    # Their median carries the sampling noise ours does not, so the two agree only to
    # that noise. The standard error of a median is 1.2533 * spread / sqrt(draws),
    # with the spread being sigma in raw log space; four of those is the bar.
    scale = float(rating.model.q_transform.std_)
    sigma = float(rating.idata.posterior["sigma"].values.mean() * scale)
    tolerance = 4 * 1.2533 * sigma / np.sqrt(ours.shape[1])
    assert relative.max() < tolerance, (
        f"median of the rating draws differs from ratingcurve's by "
        f"{relative.max():.2%}, past the {tolerance:.2%} their sampling noise "
        f"explains; the mean function is not being reconstructed correctly")


@pytest.mark.slow
def test_predict_exceeds_posterior_mean_by_the_scatter_factor(power_law, stages):
    """``predict / posterior_mean`` is the inflation the scatter causes, and only that.

    The margin here is genuinely thin, and the tolerance has to say so. ``TRUE_SCATTER``
    is 0.06 by design - a tight enough scatter that the fit is well determined - which
    puts the inflation at ``exp(0.06 ** 2 / 2)``, about 0.2%. ``predict`` resamples the
    posterior predictive on every call, so its own Monte Carlo error is
    ``sigma / sqrt(draws)``, about 0.09%. The effect is only twice the noise, so
    per-stage ``ratio > 1`` is a coin flip and must not be asserted: it is the *mean*
    ratio that has to clear 1 by more than that noise.
    """
    rating = power_law.result.rating

    # sigma is sampled in the standardized log space the model fits in
    scale = float(rating.model.q_transform.std_)
    sigma = rating.idata.posterior["sigma"].values.ravel() * scale
    expected = float(np.mean(np.exp(sigma ** 2 / 2)))

    draws = rating.idata.posterior.sizes["chain"] * rating.idata.posterior.sizes["draw"]
    monte_carlo = float(sigma.mean()) / np.sqrt(draws)

    ratio = rating.predict(stages) / rating.posterior_mean(stages)

    assert ratio.mean() > 1 + monte_carlo, (
        f"the posterior predictive mean averages {ratio.mean():.6f} of the rating, "
        f"not the >{1 + monte_carlo:.6f} the scatter requires")
    assert np.all(ratio > 1 - 3 * monte_carlo), (
        f"a stage came back at {ratio.min():.6f} of the rating, further below it than "
        f"resampling noise explains")
    assert np.allclose(ratio, expected, atol=3 * monte_carlo), (
        f"ratio {ratio} does not match exp(sigma**2/2) = {expected:.6f}")


@pytest.mark.slow
def test_posterior_mean_is_the_mean_of_the_draws(power_law, stages):
    rating = power_law.result.rating
    assert np.allclose(rating.posterior_mean(stages),
                       np.asarray(rating.posterior_draws(stages), float).mean(axis=1))


@pytest.mark.slow
def test_no_flow_below_the_stage_of_zero_flow(power_law):
    """Below every draw's breakpoint there is no flow, as ratingcurve reports.

    Measured against the lowest ``hs`` *draw*, not the posterior mean of ``hs`` that
    ``equation`` reports: half the draws sit below that mean and do carry flow there.
    """
    rating = power_law.result.rating
    breakpoints = rating.idata.posterior["hs"].values.reshape(-1, 1)
    below = np.array([breakpoints.min() - 0.5, breakpoints.min() - 0.01])

    assert np.all(np.asarray(rating.posterior_draws(below), float) == 0)
    assert np.all(rating.posterior_mean(below) == 0)


@pytest.mark.slow
def test_partial_flow_agrees_with_ratingcurve_between_the_breakpoint_draws(power_law):
    """Where only some draws carry flow, our draws and theirs still agree on the mean."""
    rating = power_law.result.rating
    zero_flow = float(np.asarray(rating.equation()["hs"], float).ravel()[0])
    straddling = np.array([zero_flow - 0.05, zero_flow + 0.05])

    ours = np.asarray(rating.posterior_draws(straddling), float)
    theirs = np.asarray(
        rating.model.predict_posterior(straddling, extend_idata=False), float)

    assert np.count_nonzero(ours) > 0, "expected some draws to carry flow here"
    # the fraction of draws with no flow is a property of the hs draws alone
    assert np.allclose((ours == 0).mean(axis=1), (theirs == 0).mean(axis=1), atol=0.02)


@pytest.mark.slow
def test_exposed_on_the_public_model(power_law, stages):
    """``posterior_mean`` reaches through the RatingModel wrapper, scalars included."""
    assert np.allclose(power_law.posterior_mean(stages),
                       power_law.result.rating.posterior_mean(stages))

    scalar = power_law.posterior_mean(float(stages[0]))
    assert isinstance(scalar, float)
    assert np.isclose(scalar, power_law.posterior_mean(stages)[0])


@pytest.mark.slow
def test_predict_is_unchanged(power_law, stages):
    """Nothing was clobbered: ``predict`` still averages the posterior predictive.

    Not an exact comparison. Each call to ``predict`` or ``predict_posterior`` runs
    ``pm.sample_posterior_predictive`` again and draws fresh noise, so two calls differ
    by the Monte Carlo error of a mean over ``draws`` samples - about
    ``sigma / sqrt(draws)`` relative. That resampling is itself the point: the estimator
    is not a deterministic function of the posterior, while ``posterior_mean`` is.
    """
    rating = power_law.result.rating
    theirs = np.asarray(
        rating.model.predict_posterior(stages, extend_idata=False), float)

    scale = float(rating.model.q_transform.std_)
    sigma = float(rating.idata.posterior["sigma"].values.mean() * scale)
    tolerance = 5 * sigma / np.sqrt(theirs.shape[1])

    assert np.allclose(rating.predict(stages), theirs.mean(axis=1), rtol=tolerance), (
        "predict no longer averages the posterior predictive")
    # and it is still the inflated quantity, not the rating. On the mean over stages,
    # not per stage: at this sigma the inflation is about twice the resampling noise,
    # so an individual stage can land either side of the rating by chance - the same
    # reason test_predict_exceeds_posterior_mean_by_the_scatter_factor averages first.
    inflation = (rating.predict(stages) / rating.posterior_mean(stages)).mean()
    assert inflation > 1 + sigma / np.sqrt(theirs.shape[1]), (
        f"predict averages {inflation:.6f} of the rating, so it is no longer the "
        f"quantity the scatter inflates")


def test_families_without_a_mean_function_say_so():
    """A family that cannot answer refuses by name rather than quietly interpolating."""
    model = lrc.Bdrc("gplm0")
    with pytest.raises(RuntimeError, match="not fitted"):
        model.posterior_mean(5.0)


@pytest.mark.slow
def test_the_table_reports_both_quantities_separately(power_law, stages):
    """One table, two intervals: the rating's, and the next gaging's.

    The predictive interval is the wider of the two by construction - it is the same
    posterior with the gaging scatter added - and a table where that ordering fails is
    reporting one of them wrongly.
    """
    table = power_law.result.rating.table(stage=stages)

    assert np.allclose(table["discharge_cfs"],
                       power_law.posterior_mean(table["stage_ft"].to_numpy(float)))
    # the two medians coincide: a symmetric log-space error leaves a median alone
    assert np.allclose(table["discharge_posterior_median_cfs"],
                       table["discharge_predictive_median_cfs"], rtol=0.05)
    assert np.all(table["posterior_lower"] >= table["lower"])
    assert np.all(table["posterior_upper"] <= table["upper"])
    assert np.all(table["posterior_lower"] <= table["discharge_cfs"])
    assert np.all(table["discharge_cfs"] <= table["posterior_upper"])
