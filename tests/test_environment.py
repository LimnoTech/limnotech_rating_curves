import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import limnotech_rating_curves as lrc

# Packages the repository environment carries that this package does not depend on.
REPO_STACK = ("torch", "geopandas", "dash")


def test_blas_is_linked():
    """PyTensor must find a BLAS library.

    An empty ``blas__ldflags`` is the failure mode that motivated the environment
    file: PyTensor falls back to an unoptimized path, NUTS becomes unusable and the
    ELPD out of reach, while every result still looks superficially fine.
    """
    import pytensor

    flags = pytensor.config.blas__ldflags
    assert flags, (
        "PyTensor reports no BLAS library (blas__ldflags is empty), so NUTS and the "
        "ELPD are unusable in this environment. Rebuild it from the repository's root "
        "environment.yml, which pins the MKL-backed builds, and do not pip install "
        "scipy, pymc or pytensor over them."
    )


def test_mkl_matmul_runs():
    """A BLAS matmul must return the right answer without crashing the process.

    MKL 2026 with numpy 2.4 aborts in the threaded BLAS path on Windows unless
    ``MKL_THREADING_LAYER=SEQUENTIAL`` is set before numpy is imported.
    :func:`limnotech_rating_curves.settings.apply_numerical_workarounds` sets it at
    package import, so reaching this assertion at all is part of what is being tested.
    """
    import os

    assert os.environ.get("MKL_THREADING_LAYER") == "SEQUENTIAL", (
        "MKL_THREADING_LAYER is not SEQUENTIAL; a numpy matmul in this environment "
        "can abort the interpreter outright"
    )
    left = np.arange(12.0).reshape(3, 4)
    product = left @ left.T
    assert product.shape == (3, 3)
    assert np.isfinite(product).all()
    np.testing.assert_allclose(product[0, 0], (left[0] ** 2).sum())


def test_a_cxx_compiler_is_available():
    """PyTensor should have a C++ compiler.

    Without one it still produces the same numbers through its NumPy path, but an ADVI
    fit that takes half a minute with the compiler does not finish in a quarter of an
    hour without it. conda-forge's ``gxx`` provides it on every platform.
    """
    import pytensor

    assert pytensor.config.cxx, (
        "PyTensor has no C++ compiler (config.cxx is empty), so every fit that is not "
        "run through nutpie takes the slow NumPy path. Install gxx from conda-forge."
    )


@pytest.mark.parametrize("sampler", lrc.settings.NUTS_SAMPLERS)
def test_every_declared_nuts_sampler_is_importable(sampler):
    """Each entry in ``settings.NUTS_SAMPLERS`` must be installed.

    The setting is what a caller chooses from, so an entry that cannot be imported is
    an unusable option rather than a documented one.
    """
    module = {"pymc": "pymc", "nutpie": "nutpie", "numpyro": "numpyro",
              "blackjax": "blackjax"}[sampler]
    assert importlib.import_module(module) is not None


@pytest.mark.slow
def test_default_sampler_fits_a_rating():
    """The default sampler must fit a rating end to end and produce an ELPD.

    This is the integration check behind the other three: it exercises the compiler,
    BLAS, nutpie's numba compilation and the PSIS-LOO path in one call, on data whose
    answer is known. The measurements come from Q = 12 (h - 1.5) ** 2.1, so a fit that
    works recovers it closely.
    """
    rng = np.random.default_rng(lrc.settings.SEED)
    stage = np.linspace(2.0, 7.0, 16)
    measurements = pd.DataFrame({
        "stage_ft": stage,
        "discharge_cfs": 12.0 * (stage - 1.5) ** 2.1
                         * np.exp(rng.normal(0, 0.05, stage.size)),
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rating = lrc.fit_rating(measurements, method="nuts")

    assert rating.fitted
    assert rating.result.config["nuts_sampler"] == lrc.settings.NUTS_SAMPLER
    assert rating.metrics.r2_log > 0.99
    assert np.isfinite(rating.metrics.elpd_loo), (
        "the fit produced no ELPD, which means the log-likelihood or the PSIS-LOO "
        "path is broken in this environment"
    )
    assert rating.diagnostics()["r_hat"].max() < 1.1


@pytest.mark.parametrize("package", REPO_STACK)
def test_wider_repository_stack_imports(package):
    """torch, geopandas and dash must import when the repository environment is used.

    These are not dependencies of this package, so the test skips where they are
    absent. Where they are present, importing them together with this package is worth
    asserting: each ships its own OpenMP runtime, and loading a second one aborts the
    process unless ``KMP_DUPLICATE_LIB_OK`` is set, which
    :func:`limnotech_rating_curves.settings.apply_numerical_workarounds` does.
    """
    try:
        module = importlib.import_module(package)
    except ImportError:
        pytest.skip(f"{package} is not installed; it belongs to the repository "
                    f"environment, not to this package")
    assert module is not None
