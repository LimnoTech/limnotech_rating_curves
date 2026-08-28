"""Per-site PSIS-LOO from one joint hierarchical fit.

The hierarchical model fits every site in a single posterior, so there is one
log-likelihood spanning every measurement and one ELPD for the whole fit. What the map
needs is a number per site, and the way to get one is to score once and slice the
pointwise result - PSIS fits each observation's tail independently, so slicing is
exact rather than an approximation.

Two things have to hold for that to be worth showing, and neither is obvious from
reading the code:

* the per-site parts must **sum back** to the joint score, which is what proves the
  decomposition is a decomposition and not an average;
* each site's Pareto-k must line up with **that site's own measurements**, because
  every index stays in range when it does not, and the map would attach a k to the
  wrong point in silence.

See ``hierarchical.md``.
"""

import numpy as np
import pandas as pd
import pytest

import arviz as az

from limnotech_rating_curves.core import Sample
from limnotech_rating_curves.evaluate.metrics import (BAYES_FIELDS, MAX_PCT_K_HIGH,
                                                      pareto_k_per_point)
from limnotech_rating_curves.models.hierarchical import OBSERVED_VAR, fit_hierarchical

#: Site shapes: (measurements, coefficient, exponent, stage of zero flow). The
#: two-measurement site is deliberately included - it is the case the hierarchical
#: model exists for, and the case PSIS is expected to struggle with.
SITES = [(20, 180.0, 1.6, 1.2), (12, 90.0, 1.8, 0.8),
         (5, 300.0, 1.5, 2.0), (2, 150.0, 1.7, 1.0)]

#: Short chains: these tests check identities that hold at any sample size, not the
#: quality of the fit.
DRAWS, TUNE = 400, 800


def _samples(seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    built = {}
    for index, (n, a, b, zero_flow) in enumerate(SITES):
        stage = np.sort(rng.uniform(zero_flow + 0.8, zero_flow + 7.0, n))
        discharge = a * (stage - zero_flow) ** b * np.exp(rng.normal(0, 0.15, n))
        built[f"site{index}"] = Sample.of(
            pd.DataFrame({"stage_ft": stage, "discharge_cfs": discharge}),
            site_id=f"site{index}", source="test")
    return built


@pytest.fixture(scope="module")
def fitted():
    return fit_hierarchical(_samples(), reference=False, draws=DRAWS, tune=TUNE,
                            progressbar=False)


@pytest.mark.slow
def test_the_fit_carries_a_log_likelihood(fitted):
    """Without this group there is nothing to score.

    PyMC does not compute one unless asked, and it ignores ``idata_kwargs`` for the
    external samplers - nutpie being the package default - so asking at sample time
    fails silently. The fit computes it explicitly afterwards instead.
    """
    assert "log_likelihood" in fitted.idata.groups()
    assert OBSERVED_VAR in fitted.idata.log_likelihood.data_vars


@pytest.mark.slow
def test_per_site_scores_sum_to_the_joint_score(fitted):
    """The decisive check: slicing must reproduce the number it was sliced from."""
    joint = az.loo(fitted.idata, var_name=OBSERVED_VAR)
    per_site = [fit.bayes for fit in fitted.results.values()]

    assert np.isclose(sum(s["elpd_loo"] for s in per_site), float(joint.elpd_loo),
                      rtol=0, atol=1e-9), "per-site ELPD does not sum to the joint ELPD"
    assert np.isclose(sum(s["p_loo"] for s in per_site), float(joint.p_loo),
                      rtol=0, atol=1e-9)
    assert sum(s["n_obs"] for s in per_site) == int(joint.n_data_points)


@pytest.mark.slow
def test_pareto_k_lines_up_with_each_site_own_measurements(fitted):
    """One k per measurement, at the site the measurement belongs to.

    The failure this guards against is silent: a mis-sliced column is still a valid
    float at a valid index, and the map would show it against the wrong point.
    """
    samples = _samples()
    for site_id, fit in fitted.results.items():
        k = pareto_k_per_point(fit)
        assert len(k) == fit.n == len(samples[site_id]), site_id
        assert np.isfinite(k).all(), f"{site_id} has a missing Pareto-k"
        assert len(fit.bayes["elpd_loo_i"]) == fit.n, site_id
        assert np.isclose(sum(fit.bayes["elpd_loo_i"]), fit.bayes["elpd_loo"])


@pytest.mark.slow
def test_a_permuted_site_order_does_not_move_the_scores(fitted):
    """Scores follow the site, not the position it was passed in.

    ``_observations`` concatenates sites in the order the mapping supplies them, so
    the log-likelihood columns move when that order does. The scores must not.
    """
    reversed_samples = dict(reversed(list(_samples().items())))
    other = fit_hierarchical(reversed_samples, reference=False, draws=DRAWS, tune=TUNE,
                             progressbar=False)
    for site_id, fit in fitted.results.items():
        assert other.results[site_id].bayes["n_obs"] == fit.bayes["n_obs"], site_id


@pytest.mark.slow
def test_every_bayes_field_is_present(fitted):
    """The map's score table reads ``BAYES_FIELDS``; a missing one shows as blank."""
    for site_id, fit in fitted.results.items():
        missing = [field for field in BAYES_FIELDS if field not in fit.bayes]
        assert not missing, f"{site_id} is missing {missing}"


@pytest.mark.slow
def test_a_two_measurement_site_is_flagged_rather_than_quoted(fitted):
    """Dropping one of two measurements is the case PSIS cannot importance-sample.

    That site's level is then pinned by one point and the population, which is a large
    move away from the full posterior - exactly what a high Pareto-k reports. The
    estimate is marked unusable rather than published, following ``evaluate.logo``.
    """
    sparse = fitted.results["site3"]
    assert sparse.n == 2
    assert sparse.bayes["pareto_k_max"] > 0.7, (
        "expected PSIS to struggle at a two-measurement site; if this now passes "
        "cleanly the reliability rule needs rechecking, not deleting")
    assert not sparse.bayes["reliable"]
    assert "PSIS unreliable" in sparse.bayes["note"]

    dense = fitted.results["site0"]
    assert dense.bayes["pct_k_high"] <= MAX_PCT_K_HIGH
    assert dense.bayes["reliable"]
