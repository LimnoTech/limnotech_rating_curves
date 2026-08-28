"""Every tunable value in the package, in one place, each overridable from the
environment.

Each setting below is read from ``os.environ`` at import time under the name
``LRC_`` + the setting's own name, falling back to the default written here. So

.. code-block:: shell

    LRC_NUTS_DRAWS=4000
    LRC_NOAA_TIMEOUT=60
    LRC_DATA_DIR=/mnt/shared/rating_curve_data

sets :data:`NUTS_DRAWS`, :data:`NOAA_TIMEOUT` and :data:`DATA_DIR`.

Reading happens **once, when this module is first imported**. To configure the
package from a ``.env`` file, load that file before importing the package:

.. code-block:: python

    from dotenv import load_dotenv
    load_dotenv()                        # must come first

    import limnotech_rating_curves as lrc

Loading a ``.env`` afterwards has no effect on values already read. This module
deliberately does not read a ``.env`` file itself: which file to load, and whether it
may override variables already set in the shell, is the caller's decision, not a
library's. ``examples/settings_from_env.py`` and ``.env.example`` show the pattern.

Nothing here depends on the install layout, so a copy installed with
``pip install git+...`` is configured exactly the same way as a source checkout. The
directory settings are all relative to the working directory, never to where the
package is installed. The data files are not shipped with the package, so point
:data:`DATA_DIR` at wherever they actually are.

Assigning to a setting after import also works and takes effect from that point on,
for the settings read at call time rather than captured in a default argument::

    lrc.settings.RELOO = True
"""

import json
import math
import os
from pathlib import Path

#: Prefix on every environment variable this module reads.
ENV_PREFIX = "LRC_"


def _raw(name: str, aliases: tuple = ()) -> str | None:
    """The environment's value for a setting, or None if it is not set.

    Parameters
    ----------
    name : str
        The setting's name. The variable read is :data:`ENV_PREFIX` + `name`.
    aliases : tuple of str, optional
        Older names for the same setting, tried in order after `name`.

    Returns
    -------
    str or None
    """
    for candidate in (name,) + tuple(aliases):
        value = os.environ.get(ENV_PREFIX + candidate)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def _str(name: str, default: str, aliases: tuple = ()) -> str:
    """A string setting."""
    value = _raw(name, aliases)
    return default if value is None else value


def _int(name: str, default: int, aliases: tuple = ()) -> int:
    """An integer setting. Underscores are allowed, as in Python literals."""
    value = _raw(name, aliases)
    return default if value is None else int(value.replace("_", ""))


def _float(name: str, default: float, aliases: tuple = ()) -> float:
    """A float setting."""
    value = _raw(name, aliases)
    return default if value is None else float(value.replace("_", ""))


def _bool(name: str, default: bool, aliases: tuple = ()) -> bool:
    """A boolean setting. ``1/true/yes/on`` are true, ``0/false/no/off`` are false."""
    value = _raw(name, aliases)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{ENV_PREFIX}{name}={value!r} is not a boolean")


def _path(name: str, default, aliases: tuple = ()) -> Path:
    """A filesystem-path setting. Not resolved, so a relative path stays relative."""
    value = _raw(name, aliases)
    return Path(default if value is None else value)


def _strings(name: str, default: tuple, aliases: tuple = ()) -> tuple:
    """A tuple-of-strings setting, written comma-separated."""
    value = _raw(name, aliases)
    if value is None:
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _floats(name: str, default: tuple, aliases: tuple = ()) -> tuple:
    """A tuple-of-floats setting, written comma-separated."""
    value = _raw(name, aliases)
    if value is None:
        return default
    return tuple(float(part) for part in value.split(",") if part.strip())


def _json(name: str, default, aliases: tuple = ()):
    """A setting whose value is a dict or list, written as JSON.

    Used for the few settings - basemap layers, per-source map styling - that are
    structured rather than scalar. Overriding one from the environment means
    supplying the whole structure, not editing part of it.
    """
    value = _raw(name, aliases)
    return default if value is None else json.loads(value)


# --- install layout ----------------------------------------------------------

#: Where this package's own files are. Not a setting - it is a fact about the
#: install, and nothing below is placed relative to it. For an installed copy it
#: points into site-packages, so a path built from it would neither find the data
#: nor be writable.
PACKAGE_DIR = Path(__file__).resolve().parent

# --- filesystem --------------------------------------------------------------
# Every directory here is relative to the working directory, so a copy installed
# with `pip install git+...` behaves exactly like a source checkout.

#: The MAGL network's data files, as against the fetch caches. They do not ship with
#: this package - they are supplied separately - so this is relative to the working
#: directory and never to where the package is installed. Resolved once at import, so
#: the path is stable for the session and prints in full. Point ``LRC_DATA_DIR`` at
#: wherever the files actually are.
DATA_DIR = _path("DATA_DIR", "data").resolve()

#: Where fetched USGS, NOAA and pagaia data is cached between runs. Relative to the
#: working directory, so it never depends on where the package is installed, and
#: resolved once at import so the path is stable for the session and prints in full.
CACHE_DIR = _path("CACHE_DIR", "lrc_cache").resolve()

#: Where this package's own outputs go. Relative to the working directory for the
#: same reason as :data:`CACHE_DIR`: an installed copy must never write into wherever
#: it happens to be installed.
OUTPUT_DIR = _path("OUTPUT_DIR", "output").resolve()
FIGURE_DIR = _path("FIGURE_DIR", OUTPUT_DIR / "figures")
POSTERIOR_DIR = _path("POSTERIOR_DIR", OUTPUT_DIR / "fitted_curves")
MAP_HTML = _path("MAP_HTML", OUTPUT_DIR / "rating_curves_map.html")

#: The MAGL network's own data, and the historical distance-to-surface record.
MAGL_DATA_DIR = _path("MAGL_DATA_DIR", DATA_DIR / "magl")
TIMESERIES_DIR = _path("TIMESERIES_DIR", DATA_DIR / "timeseries")

#: The site registry: a directory of CSVs, one per kind of entry - clusters,
#: co-located pairs, standalone gages, stations read from the pickle. See
#: ``data.magl.REGISTRY_FILES`` for the files and their columns.
MAGL_SITES_DIR = _path("MAGL_SITES_DIR", MAGL_DATA_DIR / "sites",
                       aliases=("MAGL_SITES",))

#: Surveyed control-point elevation per station (NAVD88 feet).
MAGL_CONTROL_POINTS_CSV = _path(
    "MAGL_CONTROL_POINTS_CSV",
    MAGL_DATA_DIR / "magl_station_control_point_elevations.csv")

#: Timestamped changes to those control-point elevations (sensor moves).
MAGL_SENSOR_MOVES_CSV = _path(
    "MAGL_SENSOR_MOVES_CSV",
    MAGL_DATA_DIR / "magl_changes_to_station_control_point_elevations.csv")

#: The historical distance-to-surface record, as a pickle of per-station frames.
MAGL_SPREADSHEET_PICKLE = _path(
    "MAGL_SPREADSHEET_PICKLE", TIMESERIES_DIR / "magl_spreadsheet_timeseries.pkl")

#: Which sensor each station carries. Only the Geolux rows are read.
MAGL_GEOLUX_SENSORS_CSV = _path("MAGL_GEOLUX_SENSORS_CSV",
                                MAGL_DATA_DIR / "geolux_sensors.csv")

#: Where the field workbooks live, one folder per cluster then one per sensor. Not
#: shipped either, so relative to the working directory like :data:`DATA_DIR`.
MAGL_WORKBOOK_ROOT = _path("MAGL_WORKBOOK_ROOT", "magl_curve_data").resolve()

# --- reproducibility ---------------------------------------------------------

#: Default RNG seed. Every fit, every fold split and every posterior draw is
#: seeded from this, so two runs of the same call return the same numbers.
SEED = _int("SEED", 42)

# --- sampler budgets ---------------------------------------------------------

#: ADVI optimization steps. The package default (200k) buys nothing past ~25k on
#: these sample sizes; below ~20k the fits visibly degrade.
ADVI_ITERS = _int("ADVI_ITERS", 25_000)

#: Posterior draws taken from a fitted ADVI approximation. Drawing from a fitted
#: normal approximation is nearly free, so keep it high and the posterior-mean
#: curve stays smooth.
ADVI_DRAWS = _int("ADVI_DRAWS", 8_000)

#: NUTS draws per chain, and tuning steps per chain. 4 chains x 1000 draws is a
#: normal MCMC budget and is plenty for PSIS-LOO.
NUTS_DRAWS = _int("NUTS_DRAWS", 1_000)
NUTS_TUNE = _int("NUTS_TUNE", 1_000)
NUTS_CHAINS = _int("NUTS_CHAINS", 4)

#: NUTS target acceptance rate on a sample of ordinary size. The sampler adapts its
#: step size until it accepts this fraction of proposals; this is ratingcurve's own
#: default and PyMC's recommended value for a well-behaved posterior.
NUTS_TARGET_ACCEPT = _float("NUTS_TARGET_ACCEPT", 0.95)

#: A sample of this many measurements or fewer is a *short record*, and is sampled
#: with :data:`SMALL_SAMPLE_TARGET_ACCEPT` instead.
SMALL_SAMPLE_N = _int("SMALL_SAMPLE_N", 5)

#: Target acceptance rate for a short record. With only a handful of measurements the
#: likelihood barely constrains the breakpoint, so the posterior has a long, thin
#: ridge that NUTS diverges on at a normal step size - the failure ratingcurve's
#: troubleshooting page describes
#: (https://thodson-usgs.github.io/ratingcurve/meta/troubleshooting.html). A target
#: acceptance rate this high forces a small enough step to follow the ridge. It costs
#: wall clock, which is affordable precisely because the sample is tiny.
SMALL_SAMPLE_TARGET_ACCEPT = _float("SMALL_SAMPLE_TARGET_ACCEPT", 0.999)


def target_accept_for(n: int, default: float = None) -> float:
    """The NUTS target acceptance rate to sample `n` measurements with.

    Callers do not have to tune the sampler for a short record: every ``fit``
    in the package resolves its own ``target_accept`` through this function when
    the caller did not name one.

    Parameters
    ----------
    n : int
        Measurements being fitted.
    default : float, optional
        Rate to use when `n` is not a short record. Defaults to
        :data:`NUTS_TARGET_ACCEPT`, read at call time so that assigning to the
        setting after import is honoured.

    Returns
    -------
    float
        :data:`SMALL_SAMPLE_TARGET_ACCEPT` when ``n <= SMALL_SAMPLE_N``, else
        `default`.
    """
    default = NUTS_TARGET_ACCEPT if default is None else default
    return SMALL_SAMPLE_TARGET_ACCEPT if int(n) <= SMALL_SAMPLE_N else default


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
NUTS_SAMPLER = _str("NUTS_SAMPLER", "nutpie")
NUTS_SAMPLERS = _strings("NUTS_SAMPLERS", ("pymc", "nutpie", "numpyro", "blackjax"))

#: Fitting methods a caller may ask for. ``"nuts"`` is full MCMC and is required
#: for a trustworthy ELPD / Pareto-k; ``"advi"`` is the fast variational fit, for
#: quick looks and for cross-validation folds.
METHODS = _strings("METHODS", ("nuts", "advi"))

# --- model complexity --------------------------------------------------------

#: Which models are fitted when a caller names none.
DEFAULT_MODEL_KEYS = _strings("DEFAULT_MODEL_KEYS",
                              ("power_law", "power_law_2seg", "spline",
                               "bdrc_gplm0", "linear", "quadratic",
                               "exponential"))

#: Interior knots for the natural-spline rating, capped at n-2 on small samples.
DEFAULT_SPLINE_KNOTS = _int("DEFAULT_SPLINE_KNOTS", 6)

#: Points on a fitted / fold curve's stage grid.
GRID_POINTS = _int("GRID_POINTS", 200)

#: Fraction of the observed stage range a curve grid is padded by on each side,
#: so a plotted rating extends a little beyond the measurements.
GRID_PAD_FRACTION = _float("GRID_PAD_FRACTION", 0.10)

# --- cross-validation --------------------------------------------------------

#: Default holdout fraction and split count for the many-measurement scheme
#: (see :mod:`limnotech_rating_curves.model_selection.crossval`).
CV_HOLDOUT = _float("CV_HOLDOUT", 0.90)
CV_SPLITS = _int("CV_SPLITS", 12)

#: Fewest training points a Bayesian power law can be fitted on.
CV_MIN_TRAIN = _int("CV_MIN_TRAIN", 3)

#: Fewest measurements a sample needs before any fold can be held out.
CV_MIN_POINTS = _int("CV_MIN_POINTS", 4)

#: At or above this many measurements the automatic scheme picks the holdout
#: sweep; below it, leave-one-out. Ten-plus measurements make a 90% holdout
#: meaningful; a handful do not.
CV_HOLDOUT_MIN_POINTS = _int("CV_HOLDOUT_MIN_POINTS", 10)

# --- diagnostics -------------------------------------------------------------

#: Pareto-k above this means the PSIS-LOO estimate is unreliable at that point.
PARETO_K_GOOD = _float("PARETO_K_GOOD", 0.7)

#: Share (percent) of a group's observations whose Pareto-k may exceed
#: :data:`PARETO_K_GOOD` before that group's ELPD is marked unusable: a few bad
#: points out of many is tolerable, a quarter of them is not a number to quote.
PARETO_MAX_PCT_K_HIGH = _float("PARETO_MAX_PCT_K_HIGH", 5.0)

#: Repair ``elpd_loo`` automatically wherever the package computes it, by refitting
#: the measurements whose Pareto-k exceeds :data:`PARETO_K_GOOD` and scoring them
#: exactly (see :mod:`limnotech_rating_curves.model_selection.exact_loo`).
#:
#: Off by default because a refit is a full NUTS chain, roughly twenty seconds on
#: these models, and ``rating.metrics`` is a property callers expect to return
#: promptly. Turning it on makes every ELPD in the package - tables, maps, manifests,
#: multi-site reports - the repaired one::
#:
#:     lrc.settings.RELOO = True
#:
#: The per-fit call ``lrc.model_selection.reloo(rating)`` does the same thing once,
#: without the flag.
RELOO = _bool("RELOO", False)

#: Share of a sample that may be refitted when repairing ``elpd_loo``, rounded to at
#: least one measurement. The cost is otherwise unbounded: a badly conditioned sample
#: can flag half its measurements, which turns a repair into a full exact
#: leave-one-out. Flagged measurements past the budget are left approximate and
#: counted in the result.
RELOO_MAX_FRACTION = _float("RELOO_MAX_FRACTION", 0.25)

#: R-hat (Gelman-Rubin) above this means the chains have not mixed.
R_HAT_GOOD = _float("R_HAT_GOOD", 1.01)

#: Bulk-ESS below this many draws per parameter is too few to trust a quantile.
ESS_GOOD = _int("ESS_GOOD", 400)

#: Divergences a fit may have and still be called quotable.
MAX_DIVERGENCES = _int("MAX_DIVERGENCES", 0)

# --- units -------------------------------------------------------------------

MM_PER_FOOT = _float("MM_PER_FOOT", 304.8)
MM_PER_METER = _float("MM_PER_METER", 1000.0)
FEET_PER_METER = MM_PER_METER / MM_PER_FOOT
FEET_TO_METERS = _float("FEET_TO_METERS", 0.3048)
CFS_TO_CMS = _float("CFS_TO_CMS", 0.028316846592)
CMS_TO_CFS = 1.0 / CFS_TO_CMS

# --- stage datum and zero flow -----------------------------------------------

#: How much headroom ``stage_datum="lowest"`` leaves below the stage of zero flow, as
#: a fraction of the observed stage range.
#:
#: The margin exists because of a hard constraint, not as a matter of taste. A rating
#: model estimates the **stage of zero flow** below the lowest measurement, and the
#: power-law implementation restricts that parameter to ``[0, min(stage))``. Two ways
#: to get it wrong, and this constant avoids both:
#:
#: * Reference *at* the lowest measurement and the interval collapses to nothing - the
#:   parameter has nowhere to live and the fit dies at initialization.
#: * Reference so that the true zero flow lands at **0**, the interval's lower bound,
#:   and it dies too. The truncated normal's unconstrained transform sends that bound
#:   to negative infinity, so the optimizer walks off to NaN chasing it.
#:
#: The second is why the margin is measured from the *estimated* stage of zero flow
#: rather than from the lowest measurement: any fixed offset below the lowest
#: measurement is exactly wrong for the data whose zero flow happens to sit there.
#: Measuring from the estimate puts the fitted parameter in the interior of its range
#: for any data. See ``StageDatum.lowest_observed``.
LOWEST_MARGIN_FRACTION = _float("LOWEST_MARGIN_FRACTION", 0.10)

#: Rolling-median window (in measurements) used to smooth the h(Q) curve, on samples
#: large enough to smooth.
ZERO_FLOW_SMOOTH_WINDOW = _int("ZERO_FLOW_SMOOTH_WINDOW", 5)

#: Below this many distinct measurements, smoothing is switched off.
#:
#: Smoothing exists to suppress *measurement noise*, and it needs enough points
#: either side of each one to have noise to average out. On a short record a
#: five-point median instead averages across genuine curvature - on eight points it
#: reaches a third of the way across the whole rating - and the flattening it
#: introduces biases the solved offset far more than the noise it removes. So on a
#: short record the raw points are the better estimate of the curve, and
#: ``gage_height_of_discharge`` reads from them directly.
ZERO_FLOW_MIN_POINTS_TO_SMOOTH = _int("ZERO_FLOW_MIN_POINTS_TO_SMOOTH", 15)

#: When Johnson's method cannot be solved, zero flow is placed this fraction of
#: the observed stage range below the lowest measurement.
ZERO_FLOW_FALLBACK_PAD_FRACTION = _float("ZERO_FLOW_FALLBACK_PAD_FRACTION", 0.10)

# --- where a gauging is worth taking -----------------------------------------

#: Posterior draws used when scoring expected information gain.
ACQUISITION_MAX_DRAWS = _int("ACQUISITION_MAX_DRAWS", 1_000)

#: A candidate stage is dropped when more than this fraction of draws put it at or
#: below the stage of zero flow, where the model predicts no discharge.
ACQUISITION_ZERO_FLOW_DRAW_FRACTION = _float(
    "ACQUISITION_ZERO_FLOW_DRAW_FRACTION", 0.5)

# --- USGS --------------------------------------------------------------------

#: USGS parameter codes for discharge (cubic feet per second) and gage height (feet).
USGS_PARAM_DISCHARGE_CFS = _str("USGS_PARAM_DISCHARGE_CFS", "00060")
USGS_PARAM_GAGE_HEIGHT_FT = _str("USGS_PARAM_GAGE_HEIGHT_FT", "00065")

#: Root of the Water Data STAC API, whose ``ratings`` collection holds the
#: published rating files.
USGS_STAC_ROOT = _str("USGS_STAC_ROOT", "https://api.waterdata.usgs.gov/stac/v0")

#: Where one site's expanded site file comes from.
USGS_SITE_FILE_URL = _str(
    "USGS_SITE_FILE_URL",
    "https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={site}"
    "&siteOutput=expanded&siteStatus=all")

#: Where a site's instantaneous-values readings come from.
USGS_INSTANTANEOUS_URL = _str("USGS_INSTANTANEOUS_URL",
                              "https://waterservices.usgs.gov/nwis/iv/")

#: How long to wait on any USGS web service, in seconds.
USGS_TIMEOUT = _int("USGS_TIMEOUT", 30)

#: The three rating files a site may publish.
#:
#: ``exsa``  expanded, shift-adjusted rating - the interpolated ~0.01-ft table you
#:           look an observed gage height up in. The shift is already applied.
#: ``base``  the base rating points the expanded table was built from.
#: ``corr``  stage corrections.
USGS_RATING_FILE_TYPES = _strings("USGS_RATING_FILE_TYPES",
                                  ("exsa", "base", "corr"))

# --- NOAA --------------------------------------------------------------------

#: Root of the National Water Prediction Service API.
NOAA_NWPS_ROOT = _str("NOAA_NWPS_ROOT", "https://api.water.noaa.gov/nwps/v1")

#: NOAA's NWSLI-to-USGS crosswalk. A fixed-width, pipe-delimited listing with three
#: header lines, then one row per gage: ``NWSLI|USGS number|GOES id|HSA|lat|lon|name``.
NOAA_HADS_CROSSWALK_URL = _str(
    "NOAA_HADS_CROSSWALK_URL",
    "https://hads.ncep.noaa.gov/USGS/ALL_USGS-HADS_SITES.txt")

#: Largest stage difference (ft) between the two agencies' live readings that still
#: counts as "the same axis". Wider than gauge noise, far narrower than a real datum
#: offset, which is typically a whole foot or more.
NOAA_DATUM_TOLERANCE_FT = _float("NOAA_DATUM_TOLERANCE_FT", 0.05)

#: How long to wait on either web service, in seconds.
NOAA_TIMEOUT = _int("NOAA_TIMEOUT", 30)

# --- pagaia ------------------------------------------------------------------

#: Which pagaia environment the MAGL stations are read from. Always production - it
#: is the database of record for the MAGL network, and staging is not to be read
#: from.
PAGAIA_ENVIRONMENT = _str("PAGAIA_ENVIRONMENT", "production")

#: Largest acceptable gap between a discharge measurement and the station reading
#: used for it.
PAGAIA_MATCH_TOLERANCE = _str("PAGAIA_MATCH_TOLERANCE", "3h")

#: Geolux readings before this instant are millimeters; at or after it they are
#: meters like every other station's. UTC - pagaia returns UTC timestamps and
#: ``data.datum._as_naive_index`` drops the zone without shifting the clock, so this
#: compares against the index directly. Parsed with ``pandas.Timestamp`` at use.
PAGAIA_GEOLUX_MILLIMETER_END = _str("PAGAIA_GEOLUX_MILLIMETER_END",
                                    "2026-08-19 14:30")

# --- the MAGL network --------------------------------------------------------

#: Largest acceptable gap between a discharge measurement and the stage reading
#: matched to it.
MAGL_STAGE_MATCH_TOLERANCE = _str("MAGL_STAGE_MATCH_TOLERANCE", "3h")

#: A stage reading this far below the lowest gauged stage is a sensor dropout, not
#: real low water, and is discarded before taking a low-water reference.
MAGL_DROPOUT_BAND_FT = _float("MAGL_DROPOUT_BAND_FT", 10.0)

#: MAGL stage begins around here; USGS visits older than this cannot be matched.
MAGL_RECORD_START = _str("MAGL_RECORD_START", "2024-08-01")

#: An offset larger than this is not a plausible radar-to-benchmark gap and means
#: the survey record or the spreadsheet record for that station needs review.
MAGL_OFFSET_SUSPECT_FT = _float("MAGL_OFFSET_SUSPECT_FT", 10.0)

#: Sheet in a ``flow@<sensor>.xlsx`` workbook holding the paired measurements the
#: spreadsheet's own chart trendline is fitted to.
MAGL_RATING_CURVE_SHEET = _str("MAGL_RATING_CURVE_SHEET", "Rating Curve")

#: The sample sources the MAGL site registry can produce.
MAGL_SOURCES = _strings("MAGL_SOURCES", ("usgs_gage", "colocated", "magl"))

# --- the MAGL field workbooks ------------------------------------------------

#: Rows of a summary sheet that hold the banner, the header, the coefficients and
#: the gauging visits. Nothing a rating needs is below this, and stopping here is
#: what keeps the 60,000-row 'water elevation' sheet from ever being parsed.
WORKBOOK_SUMMARY_ROWS = _int("WORKBOOK_SUMMARY_ROWS", 25)

#: Widest spread across visits in the elevation tying the stage axis to the distance
#: axis before that tie is called inconsistent. A radar that was moved mid-record
#: does not have one tie.
WORKBOOK_DATUM_TIE_TOLERANCE_FT = _float("WORKBOOK_DATUM_TIE_TOLERANCE_FT", 0.5)

#: Largest relative disagreement tolerated between the re-evaluated equation and the
#: workbook's own Estimated Discharge column.
WORKBOOK_EQUATION_CHECK_TOLERANCE = _float("WORKBOOK_EQUATION_CHECK_TOLERANCE", 1e-6)

# --- the hierarchical model --------------------------------------------------

#: Discharge that ``sigma_i`` refers to (cfs). Scatter grows as flow falls, so sigma
#: is only comparable between sites when quoted at a common discharge; 1 cfs sits in
#: the range these sensors work in.
HIER_REFERENCE_DISCHARGE_CFS = _float("HIER_REFERENCE_DISCHARGE_CFS", 1.0)

#: Centre and width of the per-site exponent prior, and the range outside which no
#: channel produces an exponent. Manning with width proportional to ``depth ** m``
#: gives ``m + 5/3``: 1.67 for a rectangular section, 2.17 parabolic, 2.67 for a V.
HIER_EXPONENT_PRIOR = _floats("HIER_EXPONENT_PRIOR", (2.0, 0.6))
HIER_EXPONENT_BOUNDS = _floats("HIER_EXPONENT_BOUNDS", (1.0, 4.0))

#: Prior scales for the population of site scatters, and for the flow dependence of
#: scatter. Loose - these are the quantities being estimated.
HIER_SIGMA_MEAN_PRIOR_SCALE = _float("HIER_SIGMA_MEAN_PRIOR_SCALE", 1.0)
HIER_SIGMA_SPREAD_PRIOR_SCALE = _float("HIER_SIGMA_SPREAD_PRIOR_SCALE", 0.5)
HIER_GAMMA_PRIOR_SCALE = _float("HIER_GAMMA_PRIOR_SCALE", 0.5)

#: Percentiles of the fitted discharges that bound where the flow dependence of
#: scatter applies. ``gamma`` is the slope of log scatter on log discharge, estimated
#: from the residuals of the sites in the fit, so it is only supported over the
#: discharges those residuals cover. Held at its edge value outside, rather than
#: extrapolated: a straight line in log space grows without bound as discharge falls,
#: and nothing measured says it should. Taken from the data in each fit, never fixed.
HIER_GAMMA_SUPPORT_PERCENTILES = _floats("HIER_GAMMA_SUPPORT_PERCENTILES",
                                         (5.0, 95.0))

#: Fewest measurements at which a site's own zero flow is estimated rather than held
#: at its prior centre. Below this the site has no stage range to place it with.
HIER_SAMPLE_ZERO_FLOW_FROM = _int("HIER_SAMPLE_ZERO_FLOW_FROM", 3)

#: Bounds on the log-depth prior spread, and the fallback when a site has too few
#: measurements for ``breakpoint_prior`` to place a breakpoint at all.
HIER_LOG_DEPTH_SPREAD_BOUNDS = _floats("HIER_LOG_DEPTH_SPREAD_BOUNDS", (0.1, 1.0))
HIER_FALLBACK_LOG_DEPTH = _floats("HIER_FALLBACK_LOG_DEPTH",
                                  (math.log(0.5), 1.0))

#: How far back USGS reference measurements are taken. A channel changes, so older
#: measurements scatter about a different rating than today's.
HIER_REFERENCE_YEARS = _int("HIER_REFERENCE_YEARS", 5)

# --- the bdrc backend --------------------------------------------------------
# R's bdrc runs 4 chains x 20000 Metropolis iterations with a 2000 burn-in thinned
# by 5 (3601 draws per chain); NUTS needs far fewer. These are separate from the
# package-wide sampler budgets above because this backend samples a marginal
# posterior of its own, at its own cost per step.

BDRC_NUM_CHAINS = _int("BDRC_NUM_CHAINS", 4)
BDRC_NUM_DRAWS = _int("BDRC_NUM_DRAWS", 1000)
BDRC_NUM_TUNE = _int("BDRC_NUM_TUNE", 1000)
BDRC_ADVI_STEPS = _int("BDRC_ADVI_STEPS", 10_000)
BDRC_SEED = _int("BDRC_SEED", 42)

#: Which NUTS implementation walks the marginal posterior. Any of PyMC's backends
#: works unchanged: the marginal is an ordinary pytensor graph wrapped in a
#: ``pm.Potential``, so there is nothing sampler-specific about it. Compiling it is
#: worth more here than for a typical model, because the marginal is only 2-10
#: dimensional and cheap per evaluation - so per-step interpreter overhead, not
#: arithmetic, is what dominates. Overridden per call; see :data:`NUTS_SAMPLER` for
#: the package-wide default.
BDRC_NUTS_SAMPLER = _str("BDRC_NUTS_SAMPLER", "nutpie")

#: Target acceptance rate on a sample of ordinary size, and the short-record pair
#: that mirrors :data:`SMALL_SAMPLE_N` / :data:`SMALL_SAMPLE_TARGET_ACCEPT`. Kept
#: separate because the marginal posterior of a rating fitted to a handful of points
#: has a thin ridge of its own that NUTS diverges on at a normal step size.
BDRC_TARGET_ACCEPT = _float("BDRC_TARGET_ACCEPT", 0.9)
BDRC_SMALL_SAMPLE_N = _int("BDRC_SMALL_SAMPLE_N", 5)
BDRC_SMALL_SAMPLE_TARGET_ACCEPT = _float("BDRC_SMALL_SAMPLE_TARGET_ACCEPT", 0.999)

# Priors, from R's ``bdrc:::priors``: hyperprior rate parameters and the Gaussian
# prior on (log a, b). Changing one of these makes the port no longer agree with R.
BDRC_MU_A = _float("BDRC_MU_A", 3.0)
BDRC_MU_B = _float("BDRC_MU_B", 1.835)
BDRC_SIG_A = _float("BDRC_SIG_A", 3.0)
BDRC_P_AB = _float("BDRC_P_AB", 0.0)
BDRC_NUGGET = _float("BDRC_NUGGET", 1e-8)
BDRC_LAMBDA_C = _float("BDRC_LAMBDA_C", 2.0)            # (h_min - c) ~ Exp(2)
BDRC_LAMBDA_SE = _float("BDRC_LAMBDA_SE", 28.78)        # sigma_eps  (plm0, gplm0)
BDRC_LAMBDA_SB = _float("BDRC_LAMBDA_SB", 5.405)        # sigma_beta (gplm0, gplm)
BDRC_LAMBDA_PB = _float("BDRC_LAMBDA_PB", 3.988)        # phi_beta   (gplm0, gplm)
BDRC_LAMBDA_ETA_1 = _float("BDRC_LAMBDA_ETA_1", 28.78)  # eta_1      (plm, gplm)
BDRC_LAMBDA_SETA = _float("BDRC_LAMBDA_SETA", 8.62)     # sigma_eta  (plm, gplm)

#: Prior sd of ``b`` when ``f(h) = b`` (plm0, plm), and when ``beta(h)`` carries the
#: curvature instead (gplm0, gplm).
BDRC_SIG_B_CONST_EXPONENT = _float("BDRC_SIG_B_CONST_EXPONENT", 0.426)
BDRC_SIG_B_VARYING_EXPONENT = _float("BDRC_SIG_B_VARYING_EXPONENT", 0.01)

#: R rejects any theta whose error variance exceeds this.
BDRC_MAX_VARIANCE = _float("BDRC_MAX_VARIANCE", 100.0)

#: ``nu = 5/2``, fixed in the paper.
BDRC_MATERN_SMOOTHNESS = _float("BDRC_MATERN_SMOOTHNESS", 2.5)

#: Cubic B-splines, 2 interior knots (plm, gplm).
BDRC_N_SPLINE_BASIS = _int("BDRC_N_SPLINE_BASIS", 6)

#: Prediction grid: fill any observed gap wider than 5 cm.
BDRC_MAX_STAGE_GAP_CM = _float("BDRC_MAX_STAGE_GAP_CM", 5.0)

# --- matplotlib figures ------------------------------------------------------

#: How a published reference rating is drawn, everywhere.
PLOT_REFERENCE_STYLE = _json("PLOT_REFERENCE_STYLE",
                             {"color": "black", "linewidth": 2.0,
                              "linestyle": "--"})

#: Color for the part of a curve that sits outside the measured stage range. Every
#: family is tabulated on a grid padded past both ends of the measurements
#: (``core.padded_stage_grid``), so the curves are comparable end to end; this is
#: what marks the part of them that no measurement supports.
PLOT_EXTRAPOLATION_COLOR = _str("PLOT_EXTRAPOLATION_COLOR", "#e08214")

#: Legend text for that part.
PLOT_EXTRAPOLATION_LABEL = _str("PLOT_EXTRAPOLATION_LABEL",
                                "extrapolated beyond the measurements")

# --- the interactive map -----------------------------------------------------

#: Esri topographic basemap with a hydrography overlay. Raster tiles, so the map
#: needs no map-provider token.
MAP_BASEMAP_LAYERS = _json("MAP_BASEMAP_LAYERS", [
    {"below": "traces", "sourcetype": "raster", "type": "raster",
     "sourceattribution": "Esri, USGS, NOAA - World Topographic Map",
     "source": ["https://services.arcgisonline.com/ArcGIS/rest/services/"
                "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"]},
    {"below": "traces", "sourcetype": "raster", "type": "raster",
     "sourceattribution": "Esri - World Hydro Reference Overlay",
     "source": ["https://services.arcgisonline.com/ArcGIS/rest/services/Reference/"
                "World_Hydro_Reference_Overlay/MapServer/tile/{z}/{y}/{x}"]},
])

#: How each sample source is drawn: (legend name, marker color, longitude jitter).
#: The jitter keeps co-located sites individually hoverable.
MAP_SOURCE_STYLE = _json("MAP_SOURCE_STYLE", {
    "usgs_gage": ["USGS discharge + USGS stage", "#08519c", 0.0],
    "colocated": ["USGS discharge + MAGL stage (co-located)", "#e6550d", 0.0007],
    "magl": ["MAGL discharge + MAGL stage", "#31a354", -0.0007],
    "pagaia": ["pagaia station", "#756bb1", 0.0014],
})

#: Drawn for a source not listed above.
MAP_FALLBACK_SOURCE_STYLE = _json("MAP_FALLBACK_SOURCE_STYLE",
                                  ["sites", "#7f7f7f", 0.0])

#: How each kind of external curve is drawn. The two are deliberately far apart: one
#: is a federal agency's published rating and the other is an equation somebody typed
#: into a spreadsheet, and a reader must never have to guess which is which.
MAP_EXTERNAL_STYLE = _json("MAP_EXTERNAL_STYLE", {
    "published_reference": {"color": "#000000", "dash": "dash", "width": 2.5,
                            "legend": "USGS published rating"},
    "spreadsheet": {"color": "#e7298a", "dash": "longdashdot", "width": 3.2,
                    "legend": "field spreadsheet equation"},
})

#: Drawn for an external curve of a kind not listed above.
MAP_FALLBACK_EXTERNAL_STYLE = _json(
    "MAP_FALLBACK_EXTERNAL_STYLE",
    {"color": "#404040", "dash": "dot", "width": 2.5, "legend": "external curve"})

#: Opacity of a credible / prediction band's fill.
MAP_BAND_ALPHA = _float("MAP_BAND_ALPHA", 0.16)

#: Figure height in pixels. The panel geometry below is in paper coordinates and a
#: button row's height in paper units depends on this, so a taller figure leaves gaps
#: and a shorter one makes the control rows collide. Change it and re-check
#: :data:`MAP_BUTTON_ROW_HEIGHT`.
MAP_FIGURE_HEIGHT = _int("MAP_FIGURE_HEIGHT", 900)

# Right-hand panel geometry. One place, so the collision check and the layout read
# the same numbers.

#: Left edge of everything in the inspector, and the map's own horizontal extent.
MAP_PANEL_X = _floats("MAP_PANEL_X", (0.63, 1.0))
MAP_DOMAIN_X = _floats("MAP_DOMAIN_X", (0.0, 0.575))

#: The rating plot.
MAP_PLOT_DOMAIN_Y = _floats("MAP_PLOT_DOMAIN_Y", (0.56, 0.94))

#: The two rows above the plot, both anchored at the top and growing downward: the
#: pane toggle, then the posterior / posterior-predictive toggle under it. The site
#: title sits above both, and :data:`MAP_TOP_MARGIN` has to cover all three or
#: Plotly clips them.
MAP_PANE_ROW_Y = _float("MAP_PANE_ROW_Y", 1.085)
MAP_VIEW_ROW_Y = _float("MAP_VIEW_ROW_Y", 1.043)
MAP_TITLE_Y = _float("MAP_TITLE_Y", 1.105)
MAP_TOP_MARGIN = _int("MAP_TOP_MARGIN", 110)

#: The control row, present only with cross-validation: the model selector, then the
#: fold arrows under it. Both are anchored at the top and grow downward.
MAP_MODEL_ROW_Y = _float("MAP_MODEL_ROW_Y", 0.530)
MAP_FOLD_ROW_Y = _float("MAP_FOLD_ROW_Y", 0.480)

#: A button row's height, in paper units at :data:`MAP_FIGURE_HEIGHT`. Used only by
#: the collision check, which has to know how far a top-anchored row reaches down.
MAP_BUTTON_ROW_HEIGHT = _float("MAP_BUTTON_ROW_HEIGHT", 34.0 / MAP_FIGURE_HEIGHT)

#: The score table. Two variants, because without the control row there is space for
#: more rows and leaving it empty would be waste.
MAP_TABLE_DOMAIN_Y_WITH_CV = _floats("MAP_TABLE_DOMAIN_Y_WITH_CV", (0.02, 0.43))
MAP_TABLE_DOMAIN_Y_NO_CV = _floats("MAP_TABLE_DOMAIN_Y_NO_CV", (0.02, 0.51))

#: What the export archive is called. ``build_map`` names it after the HTML it sits
#: beside - ``hierarchical_map.html`` gets ``hierarchical_map_fits.zip`` - so several
#: maps can share a directory, and the button offers that same name.
#: :data:`MAP_EXPORT_ZIP_NAME` is the fallback for ``build_figure``, which writes
#: nothing and so has no stem to borrow.
MAP_EXPORT_ZIP_SUFFIX = _str("MAP_EXPORT_ZIP_SUFFIX", "_fits.zip")
MAP_EXPORT_ZIP_NAME = _str("MAP_EXPORT_ZIP_NAME", "rating_fits.zip")

#: Where the export button sits: inside the map's top-left corner, clear of the
#: legend in the bottom-left and of the inspector's control rows on the right.
MAP_DOWNLOAD_XY = _floats("MAP_DOWNLOAD_XY", (0.008, 0.988))


def apply_numerical_workarounds() -> None:
    """Set the environment variables PyMC / MKL need before numpy is imported.

    Two collisions bite on Windows with the current numpy / MKL builds and both
    are fixed by an environment variable that has to be in place *before* numpy
    loads, which is why this runs from the package ``__init__``.

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
