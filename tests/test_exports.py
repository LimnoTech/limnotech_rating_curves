import json

import numpy as np
import pandas as pd
import pytest

import limnotech_rating_curves as lrc
from limnotech_rating_curves.export import exports
from limnotech_rating_curves.site import SiteRating
from limnotech_rating_curves.models import catalog


@pytest.fixture
def measurements() -> pd.DataFrame:
    """A synthetic rating: Q = 12 (h - 1.5)^1.9, with minimal noise"""
    stage = np.linspace(2.0, 9.0, 12)
    discharge = 12.0 * (stage - 1.5) ** 1.9
    rng = np.random.default_rng(0)
    discharge = discharge * rng.normal(1.0, 0.02, stage.size)
    return pd.DataFrame({"stage_ft": stage, "discharge_cfs": discharge})


def _round_trip(rating, directory):
    """Save a fitted rating and read it back."""
    written = exports.save_rating(rating, directory, sample_id="TEST-01",
                                  require_posterior=False)
    return exports.load_rating(written["manifest"]), written


@pytest.mark.parametrize("model", ["linear", "quadratic", "exponential"])
def test_least_squares_round_trip_predicts_identically(measurements, tmp_path, model):
    """A least-squares form has a closed form, so the reload must be exact."""
    rating = lrc.fit_rating(measurements, model=model)
    saved, written = _round_trip(rating, tmp_path)

    assert written["posterior"] is None          # these families have no draws
    assert saved.site_id == "TEST-01"
    assert saved.name == model
    assert saved.family in ("polynomial", "exponential")

    low, high = saved.stage_range
    stages = np.linspace(low, high, 25)
    np.testing.assert_allclose(saved.predict(stages), rating.predict(stages),
                               rtol=1e-9, atol=1e-9)
    assert saved.predict(float(stages[5])) == pytest.approx(
        float(rating.predict(float(stages[5]))), rel=1e-9)


def test_metrics_and_curve_survive_the_round_trip(measurements, tmp_path):
    rating = lrc.fit_rating(measurements, model="quadratic")
    saved, _ = _round_trip(rating, tmp_path)

    assert saved.metrics["n"] == len(measurements)
    assert saved.metrics["nse"] == pytest.approx(rating.metrics.nse, rel=1e-12)
    assert saved.metrics["r2_log"] == pytest.approx(rating.metrics.r2_log, rel=1e-12)
    assert len(saved.curve) == len(rating.curve())
    assert set(saved.curve.columns) >= {"stage_ft", "discharge_cfs", "lower", "upper"}
    assert len(saved.measurements) == len(measurements)


def test_manifest_is_readable_json_and_version_guarded(measurements, tmp_path):
    rating = lrc.fit_rating(measurements, model="quadratic")
    _, written = _round_trip(rating, tmp_path)
    manifest = json.loads(open(written["manifest"], encoding="utf-8").read())

    assert manifest["format"] == "limnotech-rating-curve"
    assert manifest["format_version"] == exports.MANIFEST_VERSION
    assert manifest["site"]["id"] == "TEST-01"

    manifest["format_version"] = exports.MANIFEST_VERSION + 1
    ahead = tmp_path / f"ahead{exports.MANIFEST_SUFFIX}"
    ahead.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="format version"):
        exports.load_rating(ahead)


def test_file_names_use_the_canonical_model_key(measurements, tmp_path):
    """A legacy key must not produce a file the catalog cannot look up."""
    assert exports.canonical_model_key("PowerLawRating_1seg") == "power_law"
    assert exports.canonical_model_key("power_law") == "power_law"
    assert exports.canonical_model_key("not_a_model") == "not_a_model"

    rating = lrc.fit_rating(measurements, model="quadratic")
    _, written = _round_trip(rating, tmp_path)
    assert written["manifest"].endswith(f"TEST-01__quadratic{exports.MANIFEST_SUFFIX}")


def test_load_rating_explains_a_missing_or_wrong_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no rating manifest"):
        exports.load_rating(tmp_path / "nothing.rating.json")
    with pytest.raises(IsADirectoryError, match="load_ratings"):
        exports.load_rating(tmp_path)

    posterior = tmp_path / f"site__power_law{exports.POSTERIOR_SUFFIX}"
    posterior.write_bytes(b"not really netcdf")
    with pytest.raises(ValueError, match="posterior half of the pair"):
        exports.load_rating(posterior)

    not_a_manifest = tmp_path / f"junk{exports.MANIFEST_SUFFIX}"
    not_a_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a rating manifest"):
        exports.load_rating(not_a_manifest)


def test_a_missing_posterior_is_reported_not_a_stack_trace(measurements, tmp_path):
    """The half that cannot be reconstructed must fail loudly and by name."""
    rating = lrc.fit_rating(measurements, model="quadratic")
    saved, _ = _round_trip(rating, tmp_path)

    # the quadratic legitimately has none, and says so
    assert saved.expects_posterior is False
    assert saved.verify()["ok"] is True
    with pytest.raises(FileNotFoundError, match="no posterior"):
        saved.posterior()

    # a manifest that claims one, whose file is gone
    manifest = dict(saved.manifest)
    manifest["model"] = {**manifest["model"], "has_posterior": True}
    manifest["posterior_file"] = "vanished.nc"
    path = tmp_path / f"claims{exports.MANIFEST_SUFFIX}"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    orphaned = exports.load_rating(path)
    assert orphaned.verify()["ok"] is False
    assert "pair has been split" in orphaned.verify()["problem"]
    with pytest.raises(FileNotFoundError, match="travel together"):
        orphaned.posterior()


def test_a_swapped_posterior_is_caught_by_the_checksum(measurements, tmp_path):
    rating = lrc.fit_rating(measurements, model="quadratic")
    saved, _ = _round_trip(rating, tmp_path)

    decoy = tmp_path / "decoy.nc"
    decoy.write_bytes(b"the wrong draws entirely")
    manifest = dict(saved.manifest)
    manifest["model"] = {**manifest["model"], "has_posterior": True}
    manifest["posterior_file"] = decoy.name
    manifest["posterior_sha256"] = "0" * 64
    path = tmp_path / f"swapped{exports.MANIFEST_SUFFIX}"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    checked = exports.load_rating(path).verify()
    assert checked["ok"] is False
    assert checked["checksum_matches"] is False
    with pytest.raises(ValueError, match="SHA-256"):
        exports.load_rating(path).posterior()


def test_save_site_round_trips_a_whole_site(measurements, tmp_path):
    """The bug this format existed to have and did not: a multi-site run's directory
    must load back."""
    sample = lrc.Sample.of(measurements, site_id="SBR-99", source="magl")
    entries = [catalog.get(key) for key in ("linear", "quadratic", "exponential")]
    site = SiteRating(sample_id="magl:SBR-99", source="magl", label="SBR-99",
                      sample=sample,
                      fits=[entry.fit(sample) for entry in entries])
    assert all(fit.ok for fit in site.fits)

    written = exports.save_site(site, tmp_path, require_posterior=False)
    assert len(written) == 3

    loaded = exports.load_ratings(tmp_path)
    assert set(loaded) == {("magl:SBR-99", key)
                           for key in ("linear", "quadratic", "exponential")}
    for (_, key), saved in loaded.items():
        live = next(fit for fit in site.fits if fit.key == key)
        low, high = saved.stage_range
        stages = np.linspace(low, high, 15)
        np.testing.assert_allclose(
            saved.predict(stages), live.rating.predict(stages), rtol=1e-9, atol=1e-9)


def test_export_directory_writes_the_human_tables(measurements, tmp_path):
    sample = lrc.Sample.of(measurements, site_id="SBR-99")
    site = SiteRating(sample_id="magl:SBR-99", source="magl", label="SBR-99",
                      sample=sample,
                      fits=[catalog.get("quadratic").fit(sample)])
    source = tmp_path / "run"
    exports.save_site(site, source, require_posterior=False)

    destination = tmp_path / "ship"
    outcome = exports.export_directory(source, destination)
    assert outcome["ratings"] == 1
    assert not outcome["problems"]
    assert (destination / "ratings_index.csv").exists()
    assert (destination / "rating_curves.csv").exists()
    assert len(list(destination.glob(f"*{exports.MANIFEST_SUFFIX}"))) == 1

    curves = pd.read_csv(destination / "rating_curves.csv")
    assert {"site", "model", "stage_ft", "discharge_cfs"} <= set(curves)
    index = pd.read_csv(destination / "ratings_index.csv")
    assert index["intact"].all()


def test_orphan_posteriors_are_named_and_not_silently_ignored(tmp_path):
    (tmp_path / "magl_magl__LC-09__power_law.nc").write_bytes(b"draws")
    assert exports.load_ratings(tmp_path) == {}
    orphans = exports.orphan_posteriors(tmp_path)
    assert [path.name for path in orphans] == ["magl_magl__LC-09__power_law.nc"]


@pytest.mark.slow
def test_posterior_round_trip_with_real_draws(measurements, tmp_path):
    """A fit that does have a posterior: the .nc must be written, checksummed, and
    read back as InferenceData."""
    rating = lrc.fit_rating(measurements, model="power_law", method="advi",
                            advi_iters=2000)
    saved, written = _round_trip(rating, tmp_path)
    assert written["posterior"] is not None
    assert saved.expects_posterior is True

    checked = saved.verify()
    assert checked["ok"] and checked["checksum_matches"] is True

    idata = saved.posterior()
    assert "posterior" in idata.groups()

    # A posterior fit is reloaded by interpolating the stored curve, so the round-trip
    # identity to assert is against the *stored table*: at its own stage nodes,
    # predict must return exactly what was written.
    curve = saved.curve
    grid = curve["stage_ft"].to_numpy(float)
    np.testing.assert_allclose(saved.predict(grid),
                               curve["discharge_cfs"].to_numpy(float),
                               rtol=1e-12, atol=0)

    # Against the live model, compared with the quantity the stored curve actually
    # holds. ``discharge_cfs`` is the posterior mean of the rating, so the live
    # counterpart is ``posterior_mean`` - deterministic, and equal to within the error
    # of interpolating a fine grid, which is what this asserts.
    #
    # Not ``predict``: that is the posterior *predictive* mean, larger by
    # ``exp(sigma ** 2 / 2)`` - 5.7% on this fit - and redrawn on every call, so on
    # this tiny ADVI fit it lands anywhere from 0.53 to 0.98 of the rating. Comparing
    # against it needed a 30% tolerance that hid the systematic offset and still
    # failed on the noise. See ``posterior_clarification.md``.
    midpoints = (grid[:-1] + grid[1:]) / 2.0
    np.testing.assert_allclose(saved.predict(midpoints),
                               rating.posterior_mean(midpoints), rtol=5e-3)

    # and the offset itself is real and one-directional, which is the whole reason the
    # stored curve is the rating rather than the predictive mean
    assert np.mean(rating.predict(midpoints) / saved.predict(midpoints)) > 1.0
