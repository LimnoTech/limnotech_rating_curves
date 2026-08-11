import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

#: The distribution root - the directory holding ``pyproject.toml``, ``README.md``
#: and the generated ``output/``. Two levels above the package itself, because the
#: package lives under ``src/``.
PROJECT_DIR = PACKAGE_DIR.parents[1]

#: The repository this package sits inside, which is where the shared ``data/`` and
#: ``cache/`` directories live - they are used by the wider project, not just here.
REPO_DIR = PACKAGE_DIR.parents[2]

# --- reproducibility ---------------------------------------------------------

#: Default RNG seed. Every fit, every fold split and every posterior draw is
#: seeded from this, so two runs of the same call return the same numbers.
SEED = 42

# --- sampler budgets ---------------------------------------------------------

#: ADVI optimization steps. The package default (200k) buys nothing past ~25k on
#: these sample sizes; below ~20k the fits visibly degrade.
ADVI_ITERS = 25_000

#: Posterior draws taken from a fitted ADVI approximation. Drawing from a fitted
#: normal approximation is nearly free, so keep it high and the posterior-mean
#: curve stays smooth.
ADVI_DRAWS = 8_000

#: NUTS draws per chain, and tuning steps per chain. 4 chains x 1000 draws is a
#: normal MCMC budget and is plenty for PSIS-LOO.
NUTS_DRAWS = 1_000
NUTS_TUNE = 1_000
NUTS_CHAINS = 4

#: Which NUTS implementation walks the posterior. All of these are drop-in: they
#: sample the same model and return the same InferenceData, so the fit is the
#: same statistical object whichever is used and only the wall clock changes.
#:
#: ``"pymc"``      PyMC's own NUTS. Pure Python per leapfrog step.
#: ``"nutpie"``    compiled sampler (Rust core, numba-compiled logp). The default:
#:                 these models are small, so per-step interpreter overhead is
#:                 the binding cost and compiling it away is the one lever that
#:                 helps. Needs no C++ compiler - numba carries its own LLVM.
#: ``"numpyro"``   JAX NUTS. CPU-only on Windows (no CUDA jaxlib for win-64).
#: ``"blackjax"``  JAX NUTS, another implementation.
NUTS_SAMPLER = os.environ.get("LRC_NUTS_SAMPLER", "nutpie")
NUTS_SAMPLERS = ("pymc", "nutpie", "numpyro", "blackjax")

#: Fitting methods a caller may ask for. ``"nuts"`` is full MCMC and is required
#: for a trustworthy ELPD / Pareto-k; ``"advi"`` is the fast variational fit, for
#: quick looks and for cross-validation folds.
METHODS = ("nuts", "advi")

# --- model complexity --------------------------------------------------------

#: Interior knots for the natural-spline rating, capped at n-2 on small samples.
DEFAULT_SPLINE_KNOTS = 6

#: Points on a fitted / fold curve's stage grid.
GRID_POINTS = 200

#: Fraction of the observed stage range a curve grid is padded by on each side,
#: so a plotted rating extends a little beyond the measurements.
GRID_PAD_FRACTION = 0.10

# --- cross-validation --------------------------------------------------------

#: Default holdout fraction and split count for the many-measurement scheme
#: (see :mod:`limnotech_rating_curves.evaluate.crossval`).
CV_HOLDOUT = 0.90
CV_SPLITS = 12

#: Fewest training points a Bayesian power law can be fitted on.
CV_MIN_TRAIN = 3

#: Fewest measurements a sample needs before any fold can be held out.
CV_MIN_POINTS = 4

#: At or above this many measurements the automatic scheme picks the holdout
#: sweep; below it, leave-one-out. Ten-plus measurements make a 90% holdout
#: meaningful; a handful do not.
CV_HOLDOUT_MIN_POINTS = 10

# --- diagnostics -------------------------------------------------------------

#: Pareto-k above this means the PSIS-LOO estimate is unreliable at that point.
PARETO_K_GOOD = 0.7

#: Repair ``elpd_loo`` automatically wherever the package computes it, by refitting
#: the measurements whose Pareto-k exceeds ``PARETO_K_GOOD`` and scoring them exactly
#: (see :mod:`limnotech_rating_curves.evaluate.exact_loo`).
#:
#: Off by default because a refit is a full NUTS chain, roughly twenty seconds on
#: these models, and ``rating.metrics`` is a property callers expect to return
#: promptly. Turning it on makes every ELPD in the package - tables, maps, manifests,
#: batch reports - the repaired one::
#:
#:     lrc.settings.RELOO = True
#:
#: The per-fit call ``lrc.evaluate.reloo(rating)`` does the same thing once, without
#: the flag.
RELOO = False

#: Share of a sample that may be refitted when repairing ``elpd_loo``, rounded to at
#: least one measurement. The cost is otherwise unbounded: a badly conditioned sample
#: can flag half its measurements, which turns a repair into a full exact
#: leave-one-out. Flagged measurements past the budget are left approximate and
#: counted in the result.
RELOO_MAX_FRACTION = 0.25

#: R-hat (Gelman-Rubin) above this means the chains have not mixed.
R_HAT_GOOD = 1.01

#: Bulk-ESS below this many draws per parameter is too few to trust a quantile.
ESS_GOOD = 400

# --- units -------------------------------------------------------------------

MM_PER_FOOT = 304.8
MM_PER_METER = 1000.0
FEET_PER_METER = MM_PER_METER / MM_PER_FOOT
FEET_TO_METERS = 0.3048
CFS_TO_CMS = 0.028316846592
CMS_TO_CFS = 1.0 / CFS_TO_CMS

# --- filesystem --------------------------------------------------------------

DATA_DIR = Path(os.environ.get("LRC_DATA_DIR", REPO_DIR / "data"))
CACHE_DIR = Path(os.environ.get("LRC_CACHE_DIR", REPO_DIR / "cache"))
OUTPUT_DIR = Path(os.environ.get("LRC_OUTPUT_DIR", PROJECT_DIR / "output"))

FIGURE_DIR = OUTPUT_DIR / "figures"
POSTERIOR_DIR = OUTPUT_DIR / "fitted_curves"      # arviz NetCDF exports
MAP_HTML = OUTPUT_DIR / "rating_curves_map.html"


def apply_numerical_workarounds() -> None:
    """Set the environment variables PyMC / MKL need before numpy is imported.

    Two collisions bite on Windows with the current numpy / MKL builds and both
    are fixed by an environment variable that has to be in place *before* numpy
    loads, which is why this runs from the package ``__init__``:

    ``KMP_DUPLICATE_LIB_OK``
        torch, statsmodels and geopandas each ship an OpenMP runtime; loading two
        of them aborts the process unless duplicates are tolerated.
    ``MKL_THREADING_LAYER=SEQUENTIAL``
        MKL 2026 with numpy 2.4 segfaults in the threaded (Intel OpenMP) BLAS
        path on this platform. These models are tiny, so single-threaded BLAS
        costs nothing.

    Both use ``setdefault``, so a value the caller already set wins.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
