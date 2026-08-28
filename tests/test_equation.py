"""The denormalized power-law equation must reproduce the rating table.

Mirrors ``test_equation`` in ratingcurve's own suite (commit 6df17a3), on the same
``green channel`` dataset, because that is what makes ``equation()`` worth exposing:
the parameters come back out of the standardized space the model fits in, so
evaluating the documented form by hand reproduces the curve the model draws.

    ln(q) = a + sum(b[i] * ln(max(h - hs[i], 0) + ho[i])),  ho[0] = 0, ho[i>0] = 1

The 15% tolerance is upstream's, and it is not slack for a sloppy implementation:
these are posterior *mean* parameters, so the curve through them tracks the table's
median column, and at a breakpoint it puts one sharp kink where the posterior draws
put theirs at different stages and average into a rounded one.
"""

import numpy as np
import pandas as pd
import pytest

import limnotech_rating_curves as lrc

#: Upstream's own tolerance, for the reason given in the module docstring.
TOLERANCE = 0.15


@pytest.fixture(scope="module")
def measurements():
    """ratingcurve's bundled ``green channel`` gaugings, on this package's column names."""
    from ratingcurve import data
    frame = data.load("green channel")
    return pd.DataFrame({"stage_ft": frame["stage"].to_numpy(float),
                         "discharge_cfs": frame["q"].to_numpy(float)})


@pytest.fixture(scope="module")
def fits(measurements):
    """One- and two-segment fits. Module-scoped: sampling is the slow part."""
    fitted = {}
    for segments in (1, 2):
        fitted[segments] = lrc.PowerLaw(segments=segments).fit(
            measurements, stage="stage_ft", discharge="discharge_cfs",
            method="nuts", seed=42)
    return fitted


def _evaluate(params: dict, stage) -> np.ndarray:
    """The documented equation, evaluated by hand from the returned parameters."""
    stage = np.asarray(stage, float)
    offsets = np.ones(len(params["b"]))
    offsets[0] = 0.0
    log_q = np.full(stage.shape, float(params["a"]))
    for exponent, breakpoint_, offset in zip(params["b"], params["hs"], offsets):
        log_q = log_q + exponent * np.log(
            np.clip(stage - breakpoint_, 0, np.inf) + offset)
    return np.exp(log_q)


@pytest.mark.slow
@pytest.mark.parametrize("segments", [1, 2])
def test_equation_returns_the_documented_parameters(fits, segments):
    params = fits[segments].equation()

    # upstream's three keys, unchanged, plus the notation they belong to
    assert set(params) == {"a", "b", "hs", "ho", "symbolic", "numeric"}
    assert np.ndim(params["a"]) == 0
    assert len(params["b"]) == segments
    assert len(params["hs"]) == segments
    assert np.isfinite(params["a"])
    assert np.all(np.isfinite(params["b"]))
    assert np.all(np.isfinite(params["hs"]))


@pytest.mark.slow
@pytest.mark.parametrize("segments", [1, 2])
def test_the_equation_reproduces_the_rating_table(fits, segments):
    model = fits[segments]
    params = model.equation()
    table = model.curve()
    stage = table["stage_ft"].to_numpy(float)

    by_hand = _evaluate(params, stage)
    tabulated = table["discharge_predictive_median_cfs"].to_numpy(float)

    # A power law has no discharge below its stage of zero flow, and the padded grid
    # runs below the measurements, so those rows are not comparable either way.
    usable = np.isfinite(by_hand) & (by_hand > 0) & (tabulated > 0)
    assert usable.sum() > 0.5 * len(stage), \
        f"the equation covered only {usable.sum()} of {len(stage)} tabulated stages"
    np.testing.assert_allclose(by_hand[usable], tabulated[usable], rtol=TOLERANCE)


@pytest.mark.slow
def test_the_equation_is_much_better_than_the_tolerance_away_from_a_kink():
    """The 15% allowance is for the neighbourhood of a breakpoint, not for everywhere.

    A one-segment fit has no kink, so it should agree far more closely than upstream's
    tolerance - otherwise the loose bound would be hiding a real error.
    """
    from ratingcurve import data
    frame = data.load("green channel")
    model = lrc.PowerLaw(segments=1).fit(
        pd.DataFrame({"stage_ft": frame["stage"].to_numpy(float),
                      "discharge_cfs": frame["q"].to_numpy(float)}),
        stage="stage_ft", discharge="discharge_cfs", method="nuts", seed=42)

    table = model.curve()
    stage = table["stage_ft"].to_numpy(float)
    by_hand = _evaluate(model.equation(), stage)
    tabulated = table["discharge_predictive_median_cfs"].to_numpy(float)
    usable = np.isfinite(by_hand) & (by_hand > 0) & (tabulated > 0)

    worst = float(np.max(np.abs(by_hand[usable] - tabulated[usable])
                         / tabulated[usable]))
    assert worst < 0.02, f"worst relative disagreement {worst:.3%} on a kink-free fit"


@pytest.mark.slow
def test_a_spline_has_no_such_parameters(measurements):
    """A spline is a basis matrix and its weights, so it must not invent an equation."""
    model = lrc.Spline().fit(measurements, stage="stage_ft",
                             discharge="discharge_cfs", method="nuts", seed=42)

    assert model.result.rating.equation() == {}
    assert not hasattr(model, "equation"), \
        "equation() is a PowerLaw method - a spline must not advertise one"


def test_equation_before_fitting_is_refused():
    with pytest.raises(RuntimeError, match="not fitted"):
        lrc.PowerLaw(segments=1).equation()
