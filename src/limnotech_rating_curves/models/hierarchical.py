"""Fit every site in one model, so a short record can borrow what it cannot measure.

A one-segment power law spends three parameters on the mean, so a site with three
measurements has nothing left to estimate scatter from and a site with two cannot be
fitted at all. Here each site's scatter is drawn from a population estimated in the
same fit: sites with data set the population, and the population supplies the sites
without.

The model, for measurement ``j`` at site ``i``, in natural logs of cfs::

    depth_ij  = stage_ij - zero_flow_i
    mu_ij     = level_i + exponent_i * (log(depth_ij) - xbar_i)
    sigma_ij  = sigma_i * exp(-gamma * (mu_ij - log q_ref))
    log q_ij ~ Normal(mu_ij, sigma_ij)

    level_i      ~ Normal(0, 10)                                per site
    exponent_i   ~ TruncatedNormal(2.0, 0.6, lower=1, upper=4)  per site, bounded
    zero_flow_i  = h_min_i - exp(log_depth_i)                   from breakpoint_prior
    log sigma_i  ~ Normal(mu_sigma[cluster_i], tau_sigma)       the hierarchy
    gamma        ~ HalfNormal(0.5)                              one value, all sites

Only sigma is pooled. Level and zero flow are properties of one place - drainage area,
section geometry, a datum - and the exponent is bounded rather than pooled because
sharing it would assert that two sites have the same cross-section shape.

A site with two measurements is fitted because sigma comes from the population, the
exponent is bounded, and zero flow is held at its Johnson estimate: the two points only
have to place the level.

Usage::

    fit = fit_hierarchical(samples)
    fit.rating("magl:SBR-09").predict(5.0)
    fit.population
    fit.summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .. import settings
from ..core import FitResult, Sample, fit_metrics, padded_stage_grid
from ..evaluate import metrics as metrics_module
from .ratingcurve import POWER_LAW_FORM, breakpoint_prior

#: Discharge that ``sigma_i`` refers to (cfs). Scatter grows as flow falls, so sigma is
#: only comparable between sites when quoted at a common discharge; 1 cfs sits in the
#: range these sensors work in.
REFERENCE_DISCHARGE_CFS = 1.0

#: Centre and width of the per-site exponent prior, and the range outside which no
#: channel produces an exponent. Manning with width proportional to ``depth ** m``
#: gives ``m + 5/3``: 1.67 for a rectangular section, 2.17 parabolic, 2.67 for a V.
EXPONENT_PRIOR = (2.0, 0.6)
EXPONENT_BOUNDS = (1.0, 4.0)

#: Prior scales for the population of site scatters, and for the flow dependence of
#: scatter. Loose - these are the quantities being estimated.
SIGMA_MEAN_PRIOR_SCALE = 1.0
SIGMA_SPREAD_PRIOR_SCALE = 0.5
GAMMA_PRIOR_SCALE = 0.5

#: Percentiles of the fitted discharges that bound where the flow dependence of scatter
#: applies. ``gamma`` is the slope of log scatter on log discharge, estimated from the
#: residuals of the sites in the fit, so it is only supported over the discharges those
#: residuals cover. Held at its edge value outside, rather than extrapolated: a straight
#: line in log space grows without bound as discharge falls, and nothing measured says
#: it should. Taken from the data in each fit, never fixed.
GAMMA_SUPPORT_PERCENTILES = (5.0, 95.0)

#: Fewest measurements at which a site's own zero flow is estimated rather than held at
#: its prior centre. Below this the site has no stage range to place it with.
SAMPLE_ZERO_FLOW_FROM = 3

#: Bounds on the log-depth prior spread, and the fallback when a site has too few
#: measurements for ``breakpoint_prior`` to place a breakpoint at all.
LOG_DEPTH_SPREAD_BOUNDS = (0.1, 1.0)
FALLBACK_LOG_DEPTH = (float(np.log(0.5)), 1.0)

#: How far back USGS reference measurements are taken. A channel changes, so older
#: measurements scatter about a different rating than today's.
REFERENCE_YEARS = 5

#: A fit is quotable only below these.
MAX_DIVERGENCES = 0
MAX_R_HAT = 1.01

#: Group a site falls into when it belongs to no known cluster.
UNGROUPED = "other"

#: How a reference gage is keyed in the results, so it is recognisable as one.
REFERENCE_PREFIX = "USGS-"

#: The observed random variable in the joint model. Its log-likelihood group carries
#: one column per measurement, across every site, which is what PSIS-LOO is scored on.
OBSERVED_VAR = "log_discharge"


# --- assembling the sites ----------------------------------------------------


def _as_mapping(samples) -> dict:
    """Normalize a Sample, a list of them, or a mapping into ``{id: Sample}``."""
    if isinstance(samples, Sample):
        samples = [samples]
    if isinstance(samples, dict):
        return dict(samples)
    return {sample.site_id or f"site_{position}": sample
            for position, sample in enumerate(samples)}


def _cluster_of(site_id: str) -> str:
    """The cluster a site belongs to, or :data:`UNGROUPED`.

    A reference gage takes the cluster of the sensor it sits with, so it informs that
    cluster's scatter rather than forming a group of its own.
    """
    from ..data import magl

    prefix, _, name = str(site_id).rpartition(":")
    if name.startswith(REFERENCE_PREFIX):
        prefix, name = "usgs_gage", name[len(REFERENCE_PREFIX):]
    lookup = (magl.gage_cluster if prefix == "usgs_gage" or name.isdigit()
              else magl.station_cluster)
    try:
        cluster = lookup(name)
    except Exception:  # noqa: BLE001 - not a MAGL station or gage
        return UNGROUPED
    return cluster or UNGROUPED


def reference_samples(years: int = REFERENCE_YEARS, *, now=None) -> dict:
    """The colocated USGS gages, as samples to fit alongside the sensors.

    They carry 26-48 measurements each against the sensors' one or two, so they are
    what determines the population. Without them ``mu_sigma`` is unidentified.

    Parameters
    ----------
    years : int
        How far back to take measurements.
    now : pandas.Timestamp, optional
        Reference date the window runs back from. Defaults to today.

    Returns
    -------
    dict
        ``{"USGS-<gage>": Sample}``, keyed so a reference is recognisable in results.
    """
    from ..data import magl, usgs

    end = pd.Timestamp.today() if now is None else pd.Timestamp(now)
    start = end - pd.DateOffset(years=years)
    gages = {}
    for station, gage in magl.colocated_pairs():
        sample = usgs.gage_sample(gage, start=start.date().isoformat(),
                                  end=end.date().isoformat())
        if len(sample):
            gages[f"{REFERENCE_PREFIX}{gage}"] = sample
    return gages


def _usable(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows a power law in log space can use: finite stage, discharge above zero."""
    return frame[np.isfinite(frame["stage_ft"]) & np.isfinite(frame["discharge_cfs"])
                 & (frame["discharge_cfs"] > 0)]


def _observations(samples: dict) -> pd.DataFrame:
    """One row per usable measurement, with its site and cluster.

    A row is dropped when stage or discharge is missing, or discharge is not strictly
    positive - a power law in log space has nothing to say about zero or reverse flow.
    """
    rows = []
    for site_id, sample in samples.items():
        frame = _usable(sample.to_frame())
        if frame.empty:
            continue
        rows.append(frame.assign(site=site_id, cluster=_cluster_of(site_id)))
    if not rows:
        raise ValueError("no usable measurements: every row was missing stage or "
                         "discharge, or had discharge at or below zero")
    return pd.concat(rows, ignore_index=True)


def _zero_flow_priors(observations: pd.DataFrame, sites) -> tuple:
    """Per-site prior on the depth of zero flow below the lowest measurement.

    Uses the package's own breakpoint prior - Johnson's three-point estimate, capped at
    half the headroom to the lowest measurement - converted from stage to log depth, so
    that ``h_min - exp(log_depth)`` stays below the lowest measurement by construction
    and ``log(stage - zero_flow)`` is always defined.

    Returns ``(centres, spreads, lowest)``, one entry per site.
    """
    centres, spreads, lowest = [], [], []
    for site in sites:
        rows = observations[observations["site"] == site]
        stage = rows["stage_ft"].to_numpy(float)
        low = float(stage.min())
        try:
            prior = breakpoint_prior(stage, rows["discharge_cfs"].to_numpy(float), 1)
            depth = max(low - float(prior["mu"][0]), 1e-3)
            centre = float(np.log(depth))
            spread = float(np.clip(float(prior["sigma"][0]) / depth,
                                   *LOG_DEPTH_SPREAD_BOUNDS))
        except Exception:  # noqa: BLE001 - too few measurements to place a breakpoint
            centre, spread = FALLBACK_LOG_DEPTH
        centres.append(centre)
        spreads.append(spread)
        lowest.append(low)
    return np.array(centres), np.array(spreads), np.array(lowest)


# --- the model ---------------------------------------------------------------


def build_model(observations: pd.DataFrame) -> pm.Model:
    """The model in the module docstring, over every site in `observations`.

    Parameters
    ----------
    observations : pandas.DataFrame
        Columns ``site``, ``cluster``, ``stage_ft``, ``discharge_cfs``.

    Returns
    -------
    pymc.Model
    """
    sites = observations["site"].unique()
    site_of = observations["site"].map(
        {name: i for i, name in enumerate(sites)}).to_numpy()

    cluster_of_site = (observations.drop_duplicates("site").set_index("site")["cluster"]
                       .reindex(sites))
    groups = sorted(cluster_of_site.unique())
    group_of_site = np.array([groups.index(cluster_of_site[name]) for name in sites])

    centres, spreads, lowest = _zero_flow_priors(observations, sites)
    counts = observations.groupby("site").size().reindex(sites).to_numpy()
    estimates_zero_flow = np.where(counts >= SAMPLE_ZERO_FLOW_FROM)[0]

    stage = observations["stage_ft"].to_numpy(float)
    log_discharge = np.log(observations["discharge_cfs"].to_numpy(float))
    support = tuple(np.percentile(log_discharge, GAMMA_SUPPORT_PERCENTILES))

    # Log depth is centred per site before it meets the exponent. Uncentred, level and
    # exponent are perfectly correlated at a site with one measurement and the sampler
    # cannot move. The centring value uses the prior depth so it stays a constant.
    prior_depth = stage - lowest[site_of] + np.exp(centres[site_of])
    centring = (pd.Series(np.log(np.maximum(prior_depth, 1e-6)))
                .groupby(site_of).transform("mean").to_numpy())

    coords = {"site": sites, "group": groups, "obs": np.arange(len(observations))}
    with pm.Model(coords=coords) as model:
        level = pm.Normal("level", mu=0.0, sigma=10.0, dims="site")

        # Only sites with a stage range to place it with estimate their own zero flow.
        depth_offset = pm.Normal("zero_flow_offset", 0.0, 1.0,
                                 shape=len(estimates_zero_flow))
        log_depth = pt.set_subtensor(
            pt.as_tensor_variable(centres)[estimates_zero_flow],
            centres[estimates_zero_flow] + depth_offset * spreads[estimates_zero_flow])
        pm.Deterministic("log_zero_flow_depth", log_depth, dims="site")
        zero_flow = pm.Deterministic("zero_flow_stage_ft",
                                     lowest - pm.math.exp(log_depth), dims="site")

        exponent = pm.TruncatedNormal(
            "exponent", mu=EXPONENT_PRIOR[0], sigma=EXPONENT_PRIOR[1],
            lower=EXPONENT_BOUNDS[0], upper=EXPONENT_BOUNDS[1], dims="site")

        mu_sigma = pm.Normal("mu_sigma", mu=np.log(0.5),
                             sigma=SIGMA_MEAN_PRIOR_SCALE, dims="group")
        tau_sigma = pm.HalfNormal("tau_sigma", sigma=SIGMA_SPREAD_PRIOR_SCALE)
        sigma_offset = pm.Normal("sigma_offset", 0.0, 1.0, dims="site")
        sigma = pm.Deterministic(
            "sigma",
            pm.math.exp(mu_sigma[group_of_site] + sigma_offset * tau_sigma),
            dims="site")

        gamma = pm.HalfNormal("gamma", sigma=GAMMA_PRIOR_SCALE)

        mean_log_discharge = (level[site_of] + exponent[site_of]
                              * (pm.math.log(stage - zero_flow[site_of]) - centring))
        # Scatter is written against the fitted mean, never the observed discharge: the
        # variance of an observation must not depend on that observation. The mean is
        # clipped to the discharges gamma was estimated over - see `support`.
        spread_of_observation = sigma[site_of] * pm.math.exp(
            -gamma * (pm.math.clip(mean_log_discharge, *support)
                      - np.log(REFERENCE_DISCHARGE_CFS)))

        pm.Normal("log_discharge", mu=mean_log_discharge, sigma=spread_of_observation,
                  observed=log_discharge, dims="obs")

    model.centring = pd.Series(centring).groupby(site_of).first().to_numpy()
    model.site_names = sites
    model.gamma_support = support
    return model


# --- one site's view of the posterior ----------------------------------------


@dataclass
class SitePosterior:
    """One site's draws, evaluated analytically rather than off a tabulated curve.

    Attributes
    ----------
    site_id : str
    level, exponent, zero_flow, sigma : numpy.ndarray
        Posterior draws, one entry each.
    centring : float
        The site's ``xbar``, subtracted from log depth before the exponent applies.
    gamma : numpy.ndarray
        Draws of the shared flow dependence of scatter.
    gamma_support : tuple
        Lowest and highest log discharge ``gamma`` was estimated over.
    """

    site_id: str
    level: np.ndarray
    exponent: np.ndarray
    zero_flow: np.ndarray
    sigma: np.ndarray
    centring: float
    gamma: np.ndarray
    gamma_support: tuple

    #: The draw vectors a site's curve and scatter are rebuilt from. ``level``,
    #: ``exponent`` and ``zero_flow`` are all :meth:`log_discharge` reads; ``sigma``
    #: and ``gamma`` are what :meth:`scatter` adds on top. Nothing else in the joint
    #: posterior enters either.
    DRAW_VARIABLES = ("level", "exponent", "zero_flow", "sigma", "gamma")

    def save(self, path) -> str:
        """Write this site's draws to an ArviZ NetCDF file.

        One site's marginal posterior is a complete description of its rating. The
        hierarchy shaped the *values* of these draws - each site's parameters were
        pulled toward its cluster's population - but it does not enter the curve's
        form, so the draws in :data:`DRAW_VARIABLES` plus the scalars in the file's
        attributes reproduce :meth:`posterior_draws` and :meth:`posterior_predictive`
        exactly. See ``posterior_clarification.md``.

        What a marginal file cannot answer is a joint question: the population
        parameters and the correlations between sites are not in it, so it cannot
        predict at a site that was never fitted. :func:`fit_hierarchical`'s own
        ``idata`` is what that needs.

        Parameters
        ----------
        path : path-like
            Destination ``.nc`` file.

        Returns
        -------
        str
            The path written.
        """
        drawn = {name: np.asarray(getattr(self, name), float).ravel()[np.newaxis, :]
                 for name in self.DRAW_VARIABLES}
        idata = az.from_dict(posterior=drawn)
        # on the posterior group, not the InferenceData: only a group's attrs survive
        # the NetCDF round trip, and :meth:`load` reads them back from there
        idata.posterior.attrs.update({
            "format": "limnotech-hierarchical-site",
            "site_id": self.site_id,
            "centring": float(self.centring),
            "gamma_support_low": float(self.gamma_support[0]),
            "gamma_support_high": float(self.gamma_support[1]),
            "reference_discharge_cfs": float(REFERENCE_DISCHARGE_CFS)})
        idata.to_netcdf(str(path))
        return str(path)

    @classmethod
    def load(cls, path) -> "SitePosterior":
        """Rebuild a site's posterior from a file :meth:`save` wrote.

        Raises
        ------
        ValueError
            If the file is not one of ours, which is worth saying by name rather
            than failing later on a missing variable.
        """
        idata = az.from_netcdf(str(path))
        attrs = dict(idata.posterior.attrs)
        if attrs.get("format") != "limnotech-hierarchical-site":
            raise ValueError(
                f"{path} is not a hierarchical site posterior (its format attribute "
                f"is {attrs.get('format')!r}); a joint fit's idata carries every "
                f"site at once and is loaded with arviz.from_netcdf directly")
        flat = lambda name: idata.posterior[name].values.ravel()
        return cls(site_id=str(attrs["site_id"]),
                   level=flat("level"), exponent=flat("exponent"),
                   zero_flow=flat("zero_flow"), sigma=flat("sigma"),
                   centring=float(attrs["centring"]), gamma=flat("gamma"),
                   gamma_support=(float(attrs["gamma_support_low"]),
                                  float(attrs["gamma_support_high"])))

    def log_discharge(self, stage) -> np.ndarray:
        """Draws of log discharge at each stage, shaped ``(draws, stages)``."""
        stages = np.atleast_1d(np.asarray(stage, float))
        depth = stages[None, :] - self.zero_flow[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.level[:, None]
                    + self.exponent[:, None] * (np.log(depth) - self.centring))

    def scatter(self, stage) -> np.ndarray:
        """Draws of the scatter at each stage, shaped like :meth:`log_discharge`.

        Outside the discharges ``gamma`` was estimated over the scatter is held at its
        edge value, matching the likelihood the fit used.
        """
        drawn = np.clip(self.log_discharge(stage), *self.gamma_support)
        with np.errstate(invalid="ignore"):
            return self.sigma[:, None] * np.exp(
                -self.gamma[:, None] * (drawn - np.log(REFERENCE_DISCHARGE_CFS)))

    def posterior_predictive(self, stage, *, seed: int = None) -> np.ndarray:
        """Posterior predictive draws of discharge (cfs), one per posterior draw.

        A draw of the rating plus a draw of the site's scatter about it - the range a
        new gaging at this stage would fall in.
        """
        rng = np.random.default_rng(settings.SEED if seed is None else seed)
        drawn = self.log_discharge(stage)
        with np.errstate(invalid="ignore", over="ignore"):
            return np.exp(drawn + self.scatter(stage) * rng.standard_normal(drawn.shape))

    def posterior_draws(self, stage) -> np.ndarray:
        """Draws of the fitted rating (cfs), without gaging scatter.

        :meth:`posterior_predictive` adds a draw of the site's scatter to each of
        these; these are the rating alone.
        """
        with np.errstate(over="ignore"):
            return np.exp(self.log_discharge(stage))

    def posterior_mean(self, stage) -> np.ndarray:
        """Posterior-mean discharge (cfs), ``E[exp(mu)]``.

        The mean of :meth:`posterior_draws`. Averages over everything the fit leaves
        uncertain about the curve, and nothing else.

        Averaging :meth:`posterior_predictive` instead would report the rating times
        ``exp(sigma ** 2 / 2)``, which has no finite value here: ``log sigma`` is
        normal, so ``sigma`` is lognormal, and that factor outgrows a lognormal tail.
        See ``posterior_clarification.md``.
        """
        return np.nanmean(self.posterior_draws(stage), axis=0)

    def predict(self, stage) -> np.ndarray:
        """Posterior-mean discharge (cfs) at the given stage.

        :meth:`posterior_mean`. Evaluated from the parameters, so a stage above the
        highest measurement is a physical extension of the power law rather than a
        missing value.
        """
        return self.posterior_mean(stage)

    def median(self, stage) -> np.ndarray:
        """Median discharge (cfs) at the given stage.

        Unaffected by the scatter: a symmetric error in log space leaves the median
        where it is, so this is the median of the rating and of the posterior
        predictive alike.
        """
        return np.exp(np.nanmedian(self.log_discharge(stage), axis=0))

    def interval(self, stage, level: float = 0.95, *, predictive: bool = True) -> tuple:
        """Lower and upper discharge at the given stage.

        Parameters
        ----------
        stage : array-like
        level : float, default 0.95
            Interval width.
        predictive : bool, default True
            Include the site's scatter, giving the range a new gaging would fall in.
            False gives the credible interval for the rating itself.
        """
        drawn = (self.posterior_predictive(stage) if predictive
                 else self.posterior_draws(stage))
        tail = 100 * (1 - level) / 2
        # draws that put zero flow above the stage contribute no discharge there
        with np.errstate(invalid="ignore"):
            return (np.nanpercentile(drawn, tail, axis=0),
                    np.nanpercentile(drawn, 100 - tail, axis=0))

    def table(self, stage) -> pd.DataFrame:
        """The rating at the given stages, in the columns every family reports.

        ``discharge_cfs`` is the rating, :meth:`posterior_mean`, and
        ``posterior_lower`` / ``posterior_upper`` are its credible interval. The
        summaries of the posterior predictive follow in their own columns, with the
        wider ``lower`` / ``upper`` interval; the predictive mean is reported for
        comparison but has no finite value where sigma is large, so it is not what
        this model predicts with.

        The two medians are equal by construction - a symmetric error in log space
        leaves the median where it is - and both columns are present so a caller can
        read one pair of columns per quantity without special-casing this family.
        """
        stages = np.atleast_1d(np.asarray(stage, float))
        lower, upper = self.interval(stages)
        posterior_lower, posterior_upper = self.interval(stages, predictive=False)
        with np.errstate(invalid="ignore"):
            predictive_mean = np.nanmean(self.posterior_predictive(stages), axis=0)
        return pd.DataFrame({"stage_ft": stages,
                             "discharge_cfs": self.posterior_mean(stages),
                             "discharge_posterior_median_cfs": self.median(stages),
                             "posterior_lower": posterior_lower,
                             "posterior_upper": posterior_upper,
                             "discharge_predictive_mean_cfs": predictive_mean,
                             "discharge_predictive_median_cfs": self.median(stages),
                             "lower": lower, "upper": upper})

    def equation(self) -> dict:
        """Posterior-mean parameters, in the package's usual log-space form.

        Centring is folded into the intercept, so these describe
        :data:`~limnotech_rating_curves.models.ratingcurve.POWER_LAW_FORM` directly.
        """
        exponent = float(np.mean(self.exponent))
        return {"a": float(np.mean(self.level)) - exponent * self.centring,
                "b": np.array([exponent]),
                "hs": np.array([float(np.mean(self.zero_flow))]),
                "ho": np.array([0.0]),
                "form": POWER_LAW_FORM}


# --- the result ---------------------------------------------------------------


@dataclass
class HierarchicalFit:
    """One joint posterior, with a per-site view onto it.

    Attributes
    ----------
    results : dict
        ``{site_id: FitResult}``, the shape every other model family returns.
    skipped : dict
        ``{site_id: reason}`` for requested sites that could not be fitted, so one
        never disappears silently.
    population : pandas.DataFrame
        ``mu_sigma`` per cluster, ``tau_sigma`` and ``gamma``, with credible intervals.
    idata : arviz.InferenceData
        The joint posterior.
    divergences, worst_r_hat : int, float
        Convergence, as measured.
    """

    results: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    population: pd.DataFrame = field(default_factory=pd.DataFrame)
    idata: object = None
    divergences: int = 0
    worst_r_hat: float = float("nan")
    _posteriors: dict = field(default_factory=dict, repr=False)

    @property
    def converged(self) -> bool:
        """Whether the fit is quotable: no divergences and R-hat within tolerance."""
        return (self.divergences <= MAX_DIVERGENCES
                and self.worst_r_hat <= MAX_R_HAT)

    def rating(self, site_id: str) -> "HierarchicalPowerLaw":
        """That site's rating, answering to the same names as :class:`ratings.PowerLaw`."""
        if site_id not in self._posteriors:
            raise KeyError(f"{site_id!r} was not in this fit; "
                           f"have {sorted(self._posteriors)}")
        return HierarchicalPowerLaw(self._posteriors[site_id], self.results[site_id])

    def sigma(self) -> pd.DataFrame:
        """Every site's scatter, with its cluster and measurement count."""
        rows = [{"site": key, "cluster": result.config.get("cluster"), "n": result.n,
                 "sigma": result.config.get("sigma"),
                 "population_sigma": result.config.get("mu_sigma")}
                for key, result in self.results.items()]
        return pd.DataFrame(rows).sort_values(["cluster", "n"], ascending=[True, False])

    def summary(self) -> str:
        """Population, per-site scatter and convergence, as text."""
        lines = [f"hierarchical fit: {len(self.results)} sites,"
                 f" {sum(r.n for r in self.results.values())} measurements"
                 + (f" ({len(self.skipped)} skipped)" if self.skipped else ""),
                 "",
                 "population:", self.population.to_string(), "",
                 "sigma by site:", self.sigma().to_string(index=False), "",
                 f"divergences {self.divergences}, worst r_hat {self.worst_r_hat:.3f}"
                 f" -> {'usable' if self.converged else 'NOT usable, see reason'}"]
        return "\n".join(lines)

    def __repr__(self) -> str:
        state = "converged" if self.converged else "not converged"
        return f"HierarchicalFit({len(self.results)} sites, {state})"


class HierarchicalPowerLaw:
    """One site's rating from a joint fit.

    Answers to the same names as :class:`~limnotech_rating_curves.ratings.PowerLaw`.
    It cannot be fitted on its own - use :func:`fit_hierarchical`, which fits every
    site together.
    """

    def __init__(self, posterior: SitePosterior, result: FitResult):
        self.posterior = posterior
        self.result = result
        self.site_id = posterior.site_id

    def predict(self, stage):
        """Posterior-mean discharge (cfs) at the given stage - :meth:`posterior_mean`."""
        return self.posterior_mean(stage)

    def posterior_draws(self, stage) -> np.ndarray:
        """Draws of the fitted rating (cfs), without gaging scatter."""
        return self.posterior.posterior_draws(stage)

    def posterior_mean(self, stage):
        """Posterior-mean discharge (cfs), ``E[exp(mu)]``."""
        predicted = self.posterior.posterior_mean(stage)
        return float(predicted[0]) if np.ndim(stage) == 0 else predicted

    def posterior_predictive(self, stage, **kwargs) -> np.ndarray:
        """Posterior predictive draws (cfs): the rating plus a draw of the scatter."""
        return self.posterior.posterior_predictive(stage, **kwargs)

    def interval(self, stage, level: float = 0.95, **kwargs) -> tuple:
        """Lower and upper discharge at the given stage."""
        lower, upper = self.posterior.interval(stage, level, **kwargs)
        if np.ndim(stage) == 0:
            return float(lower[0]), float(upper[0])
        return lower, upper

    def curve(self, stage=None, level: float = 0.95) -> pd.DataFrame:
        """The fitted rating as a table, on the fitted grid unless `stage` is given."""
        if stage is None:
            return self.result.curve
        return self.posterior.table(stage)

    def table(self, *, stage_min=None, stage_max=None, step: float = 0.01
              ) -> pd.DataFrame:
        """The rating on an evenly spaced stage grid, for a lookup or a spreadsheet."""
        fitted = self.result.curve["stage_ft"]
        low = fitted.min() if stage_min is None else stage_min
        high = fitted.max() if stage_max is None else stage_max
        return self.posterior.table(np.arange(low, high + step, step))

    def equation(self) -> dict:
        """Posterior-mean parameters in the package's usual log-space form."""
        return self.posterior.equation()

    @property
    def metrics(self) -> dict:
        return self.result.metrics

    def __repr__(self) -> str:
        return f"HierarchicalPowerLaw({self.site_id!r}, n={self.result.n})"


# --- fitting ------------------------------------------------------------------


def fit_hierarchical(samples, *, reference: bool = True, seed: int = settings.SEED,
                     draws: int = 1000, tune: int = 3000,
                     target_accept: float = 0.99, nuts_sampler: str = None,
                     progressbar: bool = False, reloo: bool = False,
                     max_refits: int = None) -> HierarchicalFit:
    """Fit every site in one model.

    Parameters
    ----------
    samples : Sample or list or dict
        The sites to fit. A mapping is keyed by site id; a list takes each sample's
        ``site_id``.
    reference : bool, default True
        Also fit the colocated USGS gages. They carry the measurements that determine
        the population, so without them the sites' scatter rests on its prior alone.
        Turn off only outside MAGL.
    seed : int
        RNG seed.
    draws, tune : int
        Posterior draws and tuning steps per chain, over four chains.
    target_accept : float
        NUTS target acceptance rate.
    nuts_sampler : str, optional
        Which NUTS implementation. Defaults to the package setting.
    progressbar : bool, default False
    reloo : bool, default False
        Refit the measurements PSIS could not handle and rescore them exactly. Off by
        default because it is expensive in a way the other families are not: one refit
        here is the **whole joint model** minus one measurement, not one site, so each
        costs a full fit. The flagged count and the projected wall-clock are logged
        before any refitting starts.
    max_refits : int, optional
        Hard ceiling on refits, on top of ``settings.RELOO_MAX_FRACTION``. Worth
        setting: the fraction alone allows a quarter of every measurement in the fit.

    Returns
    -------
    HierarchicalFit
        Per-site ratings and results, the population parameters, and the convergence
        the fit actually achieved. Check ``converged`` before quoting anything.

    Notes
    -----
    Each site's ``FitResult.bayes`` carries its own PSIS-LOO scores, split out of the
    single joint score - see :func:`_site_scores` for what that number means, which is
    not the same as the score an independently fitted site would earn.
    """
    requested = _as_mapping(samples)
    skipped = {key: "no measurements with positive discharge and finite stage"
               for key, sample in requested.items()
               if not _usable(sample.to_frame()).shape[0]}
    requested = {key: value for key, value in requested.items() if key not in skipped}
    if not requested:
        raise ValueError(f"no site had usable measurements: {skipped}")
    combined = dict(requested)
    if reference:
        combined.update({key: value for key, value in reference_samples().items()
                         if key not in combined})

    observations = _observations(combined)
    model = build_model(observations)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=1, random_seed=seed,
                          target_accept=target_accept,
                          nuts_sampler=nuts_sampler or settings.NUTS_SAMPLER,
                          progressbar=progressbar)
        # PyMC does not compute a pointwise log-likelihood unless asked, and without
        # one there is nothing for PSIS-LOO to score. It is computed here rather than
        # through ``idata_kwargs={"log_likelihood": True}`` because PyMC ignores
        # idata_kwargs for the external samplers, and the default is nutpie - that
        # route sets it silently and leaves the group missing. Costs one pass over
        # (draws, observations), nothing beside the sampling itself.
        try:
            pm.compute_log_likelihood(idata, model=model, extend_inferencedata=True,
                                      progressbar=False)
        except Exception as exc:  # noqa: BLE001 - scoring must not sink the fit
            log.warning("could not compute the log-likelihood, so there will be no "
                        "per-site ELPD (%s: %s)", type(exc).__name__, exc)

    return _assemble(idata, model, observations, requested, skipped,
                     reloo=reloo, max_refits=max_refits, seed=seed,
                     nuts_sampler=nuts_sampler)


def _site_scores(idata, observations: pd.DataFrame, requested: dict) -> dict:
    """Per-site PSIS-LOO and WAIC, split out of the one joint score.

    The joint fit has a single log-likelihood spanning every measurement at every
    site, so :func:`..evaluate.metrics.elpd_by_group` scores it once and slices the
    pointwise result by site. That is exact - PSIS fits each observation's tail
    independently - and it is the only route: scoring each site separately would
    discard the joint posterior these sites were fitted under.

    Read the result as site `i`'s **contribution to the joint score**: how well the
    model predicts one more gaging at site `i` having seen the rest of site `i` and
    every other site. A sparse site scores well here largely because its cluster does,
    which is the model working as designed rather than a flattering number.

    Parameters
    ----------
    idata : arviz.InferenceData
        The joint posterior, carrying a ``log_likelihood`` group.
    observations : pandas.DataFrame
        One row per fitted measurement, in the order the log-likelihood is indexed by,
        carrying the ``site`` column that defines the grouping.
    requested : dict
        The samples that were asked for, used only to notice when a site's fitted
        measurements are fewer than the sample it came from.

    Returns
    -------
    dict
        ``{site_id: scores}``, or an empty dict when nothing could be scored.
    """
    from ..evaluate import metrics as metrics_module

    try:
        scores = metrics_module.elpd_by_group(
            idata, observations["site"].to_numpy(), observed_var=OBSERVED_VAR)
    except Exception as exc:  # noqa: BLE001 - a missing score must not sink the fit
        log.info("no per-site ELPD (%s: %s)", type(exc).__name__, exc)
        return {}

    # Pareto-k is reported in the fitted order, which is the order `predicted` and
    # `metrics` already use. It lines up with the site's Sample rows only when nothing
    # was filtered out of it, so say so when that stops being true rather than letting
    # the map attach a k to the wrong measurement.
    for site, sample in requested.items():
        fitted = int((observations["site"] == site).sum())
        if fitted and fitted != len(sample):
            log.warning("%s: %d of %d measurements were fitted, so per-measurement "
                        "Pareto-k is indexed by the fitted rows, not the sample's",
                        site, fitted, len(sample))
    return scores


def _assemble(idata, model, observations: pd.DataFrame, requested: dict,
              skipped: dict, *, reloo: bool = False, max_refits: int = None,
              seed: int = settings.SEED, nuts_sampler: str = None
              ) -> HierarchicalFit:
    """Turn one joint posterior into per-site ratings and results."""
    posterior = idata.posterior
    sites = list(model.site_names)
    centring = dict(zip(sites, model.centring))
    gamma = posterior["gamma"].values.ravel()

    population = az.summary(idata, var_names=["mu_sigma", "tau_sigma", "gamma"],
                            hdi_prob=0.95)[["mean", "sd", "hdi_2.5%", "hdi_97.5%",
                                            "r_hat", "ess_bulk"]]
    divergences = int(idata.sample_stats["diverging"].sum())
    worst_r_hat = float(az.summary(idata, var_names=["mu_sigma", "tau_sigma", "gamma",
                                                     "sigma"])["r_hat"].max())
    warning = ("" if divergences <= MAX_DIVERGENCES and worst_r_hat <= MAX_R_HAT else
               f"provisional: {divergences} divergence(s), worst r_hat {worst_r_hat:.3f}")

    groups = list(posterior.coords["group"].values)
    cluster_of = (observations.drop_duplicates("site").set_index("site")["cluster"])
    tau_sigma = float(posterior["tau_sigma"].mean())

    scores = _site_scores(idata, observations, requested)
    if reloo:
        from ..evaluate.exact_loo import refine_hierarchical
        scores = refine_hierarchical(
            scores, idata, observations, model, seed=seed,
            nuts_sampler=nuts_sampler, max_refits=max_refits)

    posteriors, results = {}, {}
    for site in sites:
        if site not in requested:
            continue
        cluster = cluster_of[site]
        rows = observations[observations["site"] == site]
        site_posterior = SitePosterior(
            site_id=site,
            level=posterior["level"].sel(site=site).values.ravel(),
            exponent=posterior["exponent"].sel(site=site).values.ravel(),
            zero_flow=posterior["zero_flow_stage_ft"].sel(site=site).values.ravel(),
            sigma=posterior["sigma"].sel(site=site).values.ravel(),
            gamma_support=model.gamma_support,
            centring=float(centring[site]),
            gamma=gamma)

        stage = rows["stage_ft"].to_numpy(float)
        observed = rows["discharge_cfs"].to_numpy(float)
        curve = site_posterior.table(padded_stage_grid(stage))
        curve = curve[np.isfinite(curve["discharge_cfs"])
                      & (curve["discharge_cfs"] > 0)].reset_index(drop=True)
        predicted = site_posterior.predict(stage)

        mu_sigma = float(posterior["mu_sigma"]
                         .isel(group=groups.index(cluster)).mean())
        sigma_mean = float(np.mean(site_posterior.sigma))
        results[site] = FitResult(
            key="hierarchical", label="hierarchical power law", family="hierarchical",
            n=len(rows), status="ok", reason=warning,
            config={"cluster": cluster, "sigma": sigma_mean,
                    "mu_sigma": np.exp(mu_sigma), "tau_sigma": tau_sigma,
                    "gamma": float(np.mean(gamma)),
                    "sigma_over_population": sigma_mean / np.exp(mu_sigma),
                    "exponent": float(np.mean(site_posterior.exponent)),
                    "zero_flow_ft": float(np.mean(site_posterior.zero_flow))},
            curve=curve, predicted=predicted,
            metrics=fit_metrics(observed, predicted),
            bayes=scores.get(site) or metrics_module.unavailable(
                "per-site ELPD unavailable for this fit"),
            rating=site_posterior)
        posteriors[site] = site_posterior

    return HierarchicalFit(results=results, skipped=skipped, population=population,
                           idata=idata, divergences=divergences,
                           worst_r_hat=worst_r_hat, _posteriors=posteriors)
