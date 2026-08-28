"""A hierarchical site's marginal posterior must survive a round trip on its own.

The hierarchy shapes the *values* of a site's draws - each site's parameters are pulled
toward its cluster's population - but it does not enter the form of any site's curve.
``SitePosterior.log_discharge`` reads ``level``, ``exponent`` and ``zero_flow``, and
``scatter`` adds ``sigma`` and ``gamma``. Nothing else in the joint posterior appears in
either. So one site's file is a complete description of that site's rating, and these
tests hold that claim to bit-exactness rather than to a tolerance.

What a marginal file cannot do is answer a joint question - the population parameters
and the between-site correlations are not in it. See ``posterior_clarification.md``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from limnotech_rating_curves.core import Sample, SiteRating
from limnotech_rating_curves.models.hierarchical import SitePosterior, fit_hierarchical

#: Three sites of deliberately unequal size: the many-measurement site determines the
#: population, and the four-point site is the one that has to borrow from it.
SITE_SIZES = {"site_A": 14, "site_B": 6, "site_C": 4}


def _samples(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    samples = {}
    for name, n in SITE_SIZES.items():
        stage = np.sort(rng.uniform(2.0, 8.0, n))
        discharge = (150 * (stage - 1.1) ** 1.9
                     * np.exp(rng.normal(0, 0.12, n)))
        samples[name] = Sample.of(
            pd.DataFrame({"stage_ft": stage, "discharge_cfs": discharge}),
            site_id=name, source="test")
    return samples


@pytest.fixture(scope="module")
def joint():
    """One joint fit, short-sampled: these tests are about plumbing, not convergence."""
    samples = _samples()
    return samples, fit_hierarchical(samples, reference=False, draws=200, tune=500,
                                     progressbar=False)


@pytest.fixture(scope="module")
def stages():
    return np.linspace(2.5, 7.5, 11)


@pytest.mark.slow
def test_a_site_posterior_round_trips_exactly(joint, stages, tmp_path_factory):
    """Save, reload, and every quantity a caller can ask for is identical.

    Bit-exact, not close: the file carries the draws themselves, so anything less
    would mean a lossy write - float32, a dropped scalar - that a tolerance test would
    hide.
    """
    _, fit = joint
    directory = tmp_path_factory.mktemp("site_posterior")
    for name in SITE_SIZES:
        live = fit.results[name].rating
        back = SitePosterior.load(live.save(directory / f"{name}.nc"))

        assert back.site_id == live.site_id
        assert back.centring == live.centring
        assert back.gamma_support == live.gamma_support
        for quantity in ("posterior_draws", "posterior_predictive", "scatter",
                         "posterior_mean", "median"):
            assert np.array_equal(getattr(back, quantity)(stages),
                                  getattr(live, quantity)(stages)), quantity
        assert back.table(stages).equals(live.table(stages))


@pytest.mark.slow
def test_the_curve_needs_no_population_parameters(joint, stages):
    """The rating reads only per-site draws, which is why a site is separable.

    Pinned deliberately: if ``log_discharge`` ever started reading ``gamma`` or a
    population parameter, a per-site export would silently stop being sufficient and
    this is the test that should fail.
    """
    _, fit = joint
    live = fit.results["site_B"].rating

    stripped = SitePosterior(
        site_id=live.site_id, level=live.level, exponent=live.exponent,
        zero_flow=live.zero_flow, sigma=live.sigma, centring=live.centring,
        gamma=np.full_like(live.gamma, np.nan), gamma_support=live.gamma_support)

    # the rating survives gamma being destroyed; only the scatter depends on it
    assert np.array_equal(stripped.posterior_draws(stages),
                          live.posterior_draws(stages))
    assert np.array_equal(stripped.posterior_mean(stages),
                          live.posterior_mean(stages))
    assert np.all(np.isnan(stripped.scatter(stages)))


@pytest.mark.slow
def test_load_refuses_a_file_that_is_not_a_site_posterior(joint, tmp_path):
    """A joint fit's idata is a different thing, and saying so beats a KeyError later.

    Note what saving the joint posterior takes: nutpie writes a dict into the
    InferenceData's attributes and NetCDF stores only strings, numbers and arrays, so
    ``coerced_netcdf_attrs`` has to JSON-encode them first. A bare
    ``fit.idata.to_netcdf(...)`` raises ``TypeError`` on a hierarchical fit.
    """
    from limnotech_rating_curves.models.ratingcurve import coerced_netcdf_attrs

    _, fit = joint
    path = tmp_path / "joint.nc"
    with coerced_netcdf_attrs(fit.idata):
        fit.idata.to_netcdf(str(path))

    with pytest.raises(ValueError, match="not a hierarchical site posterior"):
        SitePosterior.load(path)


@pytest.mark.slow
def test_the_exported_pair_rebuilds_a_usable_rating_table(joint, tmp_path):
    """The zip's contents, used the way a reader would: load_rating then table()."""
    from limnotech_rating_curves import exports

    samples, fit = joint
    written = exports.save_fit(fit.results["site_B"], samples["site_B"], tmp_path,
                              sample_id="site_B")
    saved = exports.load_rating(written["manifest"])

    assert saved.verify()["ok"]
    assert saved.family == "hierarchical"
    table = saved.table(stage_min=3.0, stage_max=6.0, step=0.5)
    assert len(table) == 7
    assert table["discharge_cfs"].is_monotonic_increasing
    assert np.all(table["lower"] < table["discharge_cfs"])
    assert np.all(table["discharge_cfs"] < table["upper"])
    # and the draws came along, so the band can be recomputed at another level
    assert "posterior" in saved.posterior().groups()


@pytest.mark.slow
def test_the_map_embeds_a_loadable_export_bundle(joint, tmp_path):
    """The button's payload is the same pair of files a scripted export writes."""
    import base64
    import io
    import zipfile

    from limnotech_rating_curves import exports
    from limnotech_rating_curves.view.mapview import build_figure

    samples, fit = joint
    coords = [(42.3, -83.7), (42.4, -83.6), (42.5, -83.5)]
    sites = [SiteRating(sample_id=name, source="test", label=name,
                        sample=samples[name], fits=[fit.results[name]],
                        coords=coords[index])
             for index, name in enumerate(SITE_SIZES)]

    figure, script = build_figure(sites, exports=True)

    assert "download" in [menu.name for menu in figure.layout.updatemenus]
    payload = script.split('var EXPORT_ZIP = "', 1)[1].split('"', 1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        names = sorted(archive.namelist())
        archive.extractall(tmp_path)

    # one folder per sensor, so a manifest and the posterior it names stay together
    assert names == sorted([f"{name}/{name}__hierarchical{suffix}"
                            for name in SITE_SIZES
                            for suffix in (".nc", ".rating.json")])
    saved = exports.load_rating(
        tmp_path / "site_C" / "site_C__hierarchical.rating.json")
    assert saved.verify()["ok"]
    assert not saved.table(stage_min=3.0, stage_max=4.0, step=0.5).empty
    # and a folder on its own is a directory load_ratings understands
    assert len(exports.load_ratings(tmp_path / "site_A")) == 1


@pytest.mark.slow
def test_the_sidecar_zip_is_the_same_file_the_button_hands_back(joint, tmp_path):
    """Two routes to one archive, byte for byte.

    The point of building the bytes once and using them twice: if the sidecar were
    zipped separately from the embedded copy, the two would drift - different mtimes
    in the zip headers at least, a different set of fits at worst - and "the file on
    disk is what the button gives you" would stop being true.
    """
    import base64
    import hashlib

    from limnotech_rating_curves.view.mapview import build_map

    samples, fit = joint
    coords = [(42.3, -83.7), (42.4, -83.6), (42.5, -83.5)]
    sites = [SiteRating(sample_id=name, source="test", label=name,
                        sample=samples[name], fits=[fit.results[name]],
                        coords=coords[index])
             for index, name in enumerate(SITE_SIZES)]

    written = Path(build_map(sites, output_html=tmp_path / "somemap.html",
                             exports=True))
    sidecar = written.parent / "somemap_fits.zip"
    assert sidecar.exists(), "the sidecar is named after the map it sits beside"

    page = written.read_text(encoding="utf-8")
    embedded = base64.b64decode(page.split('var EXPORT_ZIP = "', 1)[1]
                                    .split('"', 1)[0])
    assert hashlib.sha256(embedded).hexdigest() ==            hashlib.sha256(sidecar.read_bytes()).hexdigest()
    # and the button offers the sidecar's own name, so a reader can tell they are one
    assert f'var EXPORT_NAME = "{sidecar.name}"' in page


def test_no_sidecar_and_no_payload_without_exports(tmp_path, joint):
    """The default writes one file and embeds nothing."""
    from limnotech_rating_curves.view.mapview import build_map

    samples, fit = joint
    sites = [SiteRating(sample_id=name, source="test", label=name,
                        sample=samples[name], fits=[fit.results[name]],
                        coords=(42.3, -83.7))
             for name in SITE_SIZES]

    written = Path(build_map(sites, output_html=tmp_path / "bare.html"))
    assert not (tmp_path / "bare_fits.zip").exists()
    assert 'var EXPORT_ZIP = ""' in Path(written).read_text(encoding="utf-8")


@pytest.mark.parametrize("sample_id, folder", [
    ("magl:SR-04", "magl__SR-04"),
    ("usgs_gage:04176356", "usgs_gage__04176356"),
    ("plain", "plain"),
])
def test_archive_folder_names_are_legal_paths(sample_id, folder):
    """A colon is not a legal path component on Windows, so it cannot reach the zip."""
    from limnotech_rating_curves.view.mapview import _archive_folder

    assert _archive_folder(sample_id) == folder


def test_no_export_button_when_exports_are_not_asked_for():
    """The default must stay cheap: no posteriors written, no megabytes embedded."""
    from limnotech_rating_curves.view.mapview import _controls

    assert [menu["name"] for menu in _controls([], has_cv=False)] == ["pane", "view"]
