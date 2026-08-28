"""Predicting at new stages must not destroy the fit's predictive score.

ratingcurve's ``predict`` pushes new stage data onto the PyMC model with
``pm.set_data`` and leaves it there. Computing the pointwise log-likelihood on demand
afterwards then tries to score the fitted observations against whatever stages were
predicted last, and dies on the shape mismatch:

    ValueError: Input dimension mismatch: (input[3].shape[0] = 8676,
                                           input[6].shape[0] = 244)

This went unnoticed because ``ratingcurve.fit`` ends with ``predict(stage)`` on the
fitting stages, so the shapes happened to line up for anyone who only ever scored a
fit before predicting anywhere else. Predicting at a *different* number of stages -
which is what comparing a curve against another gage's measurements does - broke it.

The fix is to attach the log-likelihood at fit time, while the model still holds its
fitting data.
"""

import numpy as np
import pandas as pd
import pytest

import limnotech_rating_curves as lrc


@pytest.fixture(scope="module")
def fitted():
    """A small NUTS fit. Module-scoped: sampling is the slow part."""
    rng = np.random.default_rng(0)
    stage = np.sort(rng.uniform(2.0, 9.0, 30))
    discharge = 180 * (stage - 1.2) ** 1.6 * np.exp(rng.normal(0, 0.06, 30))
    frame = pd.DataFrame({
        "stage_ft": stage, "discharge_cfs": discharge,
        "time": pd.date_range("2020-01-01", periods=30, freq="11D")})
    model = lrc.PowerLaw(segments=1).fit(
        frame, stage="stage_ft", discharge="discharge_cfs", time="time",
        method="nuts", seed=42)
    return model, stage


@pytest.mark.slow
def test_elpd_survives_predicting_at_a_different_number_of_stages(fitted):
    model, _ = fitted

    # a different length from the 30 the model was fitted on - this is what broke it
    model.predict(np.linspace(3.0, 8.0, 137))

    assert np.isfinite(model.metrics.elpd_loo), (
        "elpd_loo went unavailable after predicting at a different number of stages")


@pytest.mark.slow
def test_logo_survives_predicting_at_a_different_number_of_stages(fitted):
    model, _ = fitted

    model.predict(np.linspace(3.0, 8.0, 137))
    scores = model.logo(block="1D")

    assert np.isfinite(scores["elpd_loo"]), scores.get("note")
    assert scores["n_observations"] == 30


@pytest.mark.slow
def test_curve_over_an_arbitrary_grid_also_leaves_scoring_intact(fitted):
    model, _ = fitted

    # curve() goes through table(), the other route that sets model data
    model.curve(np.linspace(2.5, 9.5, 411))

    assert np.isfinite(model.metrics.elpd_loo)


@pytest.mark.slow
def test_predictions_themselves_are_still_correct_after_scoring(fitted):
    """Scoring must not disturb the model the predictions come from.

    Note that ``predict`` is *not* deterministic - it draws from the posterior
    predictive, so two consecutive calls differ by a few tenths of a percent
    (measured: 3.4e-3 relative on this fit). So the check is that scoring leaves
    predictions within that same sampling noise, not that they are identical. A model
    left in a mutated state would be wrong by orders of magnitude, not by 0.3%.
    """
    model, stage = fitted

    first = model.predict(stage)
    second = model.predict(stage)
    noise = float(np.max(np.abs(second - first) / first))

    model.metrics  # noqa: B018 - computing the score must not disturb the model
    model.logo(block="1D")
    after = model.predict(stage)
    drift = float(np.max(np.abs(after - first) / first))

    assert drift < max(10 * noise, 0.05), (
        f"predictions moved by {drift:.2e} after scoring, against {noise:.2e} of "
        "sampling noise between two untouched calls")
