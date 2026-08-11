import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluate import crossval, metrics as metrics_module
from ..evaluate.exact_loo import refine as refine_elpd
from ..view import mapview, plots
from ..models import catalog
from .. import exports, settings
from ..data import usgs
from ..core import Sample, SiteRating

log = logging.getLogger(__name__)

#: The sample sources the MAGL site source can produce.
SOURCES = ("usgs_gage", "colocated", "magl")


@dataclass
class Report:
    """The result of a batch :func:`run`.

    Attributes
    ----------
    map_path : pathlib.Path
        The interactive map that was written - the primary output.
    results : pandas.DataFrame
        One row per curve per site: identity, in-sample scores, Bayesian scores, and
        a row for every model that was skipped or failed, so nothing is hidden.
    sites : list of SiteRating
        The fitted sites.
    cross_validation : dict
        ``{sample_id: CrossValidation}`` for the sites that were cross-validated.
    """

    map_path: Path
    results: pd.DataFrame
    sites: list = field(default_factory=list)
    cross_validation: dict = field(default_factory=dict)

    def __getitem__(self, sample_id) -> SiteRating:
        """Get one site by ``sample_id``."""
        for site in self.sites:
            if site.sample_id == sample_id:
                return site
        raise KeyError(f"no site {sample_id!r} in this report")

    def cv_table(self) -> pd.DataFrame:
        """Every site's cross-validation headline, in one table.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``sample_id`` and model, with the scheme that ran and the
            held-out scores. Empty if nothing was cross-validated.
        """
        frames = []
        for sample_id, result in self.cross_validation.items():
            headline = result.headline
            if headline.empty:
                continue
            frame = headline.reset_index()
            frame.insert(0, "sample_id", sample_id)
            frame.insert(1, "scheme", result.scheme.get("name"))
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def __repr__(self):
        return f"Report({len(self.sites)} sites -> {self.map_path})"


# ---------------------------------------------------------------------------
# assembling the sites
# ---------------------------------------------------------------------------

def _site_matches(tokens, *, group=None, gage=None, station=None) -> bool:
    """Does a candidate site match any selection token? No tokens matches everything."""
    if not tokens:
        return True
    for raw in tokens:
        token = str(raw).strip()
        if group and token.lower() == str(group).lower():
            return True
        if gage and (token == gage or usgs.normalize_site_id(token) == gage):
            return True
        if station and token.lower().replace("_wl", "") == str(station).lower():
            return True
    return False


def assemble_magl_sites(sites=None, sources=None, min_points: int = 3,
                        refresh: bool = False) -> list:
    """Build a :class:`SiteRating` for each selected MAGL-network site, unfitted.

    Only the selected sites are fetched, so a targeted run stays fast.

    Parameters
    ----------
    sites : sequence of str, optional
        Selection tokens: a MAGL sensor (``"SBR-09"``), a co-located station
        (``"SR-08"``), a cluster (``"SR"``), or a USGS gage id (``"04176356"``).
        ``None`` takes every site.
    sources : sequence of str, optional
        Restrict to these sample sources (see ``SOURCES``).
    min_points : int, default 3
        Fewest discharge measurements before a MAGL sensor is included.
    refresh : bool, default False
        Refetch instead of using the cache.

    Returns
    -------
    list of SiteRating
        With samples and coordinates filled in and ``fits`` still empty.
    """
    from ..data import magl
    from ..data import pagaia

    wanted = lambda source: (not sources) or (source in sources)

    gages = magl.all_gages()
    selected_gages = [gage for gage in gages if wanted("usgs_gage")
                      and _site_matches(sites, group=magl.gage_cluster(gage),
                                        gage=gage)]
    selected_pairs = [(station, gage) for station, gage in magl.colocated_pairs()
                      if wanted("colocated")
                      and _site_matches(sites, group=magl.station_cluster(station),
                                        gage=gage, station=station)]
    selected_sensors = [sensor for sensor in magl.list_sensors(min_points).index
                        if wanted("magl")
                        and _site_matches(sites, group=magl.station_cluster(sensor),
                                          station=sensor)]

    needed_gages = set(selected_gages) | {gage for _, gage in selected_pairs}
    gage_coordinates = usgs.coordinates(sorted(needed_gages), refresh=refresh) \
        if needed_gages else {}
    gage_names = {}
    if needed_gages:
        info = usgs.site_info(sorted(needed_gages), refresh=refresh)
        gage_names = {gage: str(row.get("station_name") or "")
                      for gage, row in info.iterrows()}

    station_names = ([f"{sensor}_WL" for sensor in selected_sensors]
                     + [f"{station}_WL" for station, _ in selected_pairs])
    station_coordinates = {}
    if station_names:
        try:
            station_coordinates = pagaia.station_coordinates(
                magl.pagaia_stations(sorted(set(station_names))))
        except Exception as exc:  # noqa: BLE001 - pagaia down, off VPN, or not installed
            log.warning("MAGL station coordinates unavailable (%s: %s); those sites "
                        "will fall back to their gage's position or be skipped",
                        type(exc).__name__, exc)

    assembled = []

    for gage in selected_gages:
        sample = usgs.gage_sample(gage, refresh=refresh)
        assembled.append(SiteRating(
            sample_id=f"usgs_gage:{gage}", source="usgs_gage",
            label=f"USGS {gage} {gage_names.get(gage, '')}".strip(), sample=sample,
            gage=gage, group=magl.gage_cluster(gage),
            coords=gage_coordinates.get(gage)))

    for station, gage in selected_pairs:
        try:
            sample = magl.colocated_sample(station, gage, refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.warning("co-located %s could not be assembled: %s", station, exc)
            continue
        assembled.append(SiteRating(
            sample_id=f"colocated:{station}", source="colocated",
            label=f"{station} MAGL stage vs USGS {gage} discharge", sample=sample,
            station=f"{station}_WL", gage=gage,
            group=magl.station_cluster(station),
            coords=(station_coordinates.get(f"{station}_WL")
                    or gage_coordinates.get(gage))))

    for sensor in selected_sensors:
        try:
            sample = magl.sensor_sample(sensor, refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.warning("MAGL %s could not be assembled: %s", sensor, exc)
            continue
        assembled.append(SiteRating(
            sample_id=f"magl:{sensor}", source="magl",
            label=f"{sensor} (MAGL discharge)", sample=sample,
            station=f"{sensor}_WL", group=magl.station_cluster(sensor),
            coords=station_coordinates.get(f"{sensor}_WL")))

    return assembled


def sites_from_samples(samples) -> list:
    """Wrap plain samples as :class:`SiteRating` objects so they can go through :func:`run`.

    Parameters
    ----------
    samples : sequence of Sample, or dict
        The samples. A dict is read as ``{sample_id: Sample}``; otherwise each
        sample's ``site_id`` is used.

    Returns
    -------
    list of SiteRating
        A sample with no coordinates cannot appear on the map, but is still fitted
        and scored, and appears in ``results``.
    """
    if isinstance(samples, dict):
        pairs = list(samples.items())
    else:
        pairs = [(sample.site_id or f"sample_{index}", sample)
                 for index, sample in enumerate(samples)]
    return [SiteRating(sample_id=str(sample_id), source=sample.source or "sample",
                       label=sample.site_id or str(sample_id), sample=sample)
            for sample_id, sample in pairs]


# ---------------------------------------------------------------------------
# the stages
# ---------------------------------------------------------------------------

def fit_sites(sites, entries, *, method: str = "nuts", seed: int = settings.SEED,
              zero_flow=None, nuts_sampler=None) -> None:
    """Fit every model on all of each site's sample, in place.

    Parameters
    ----------
    sites : sequence of SiteRating
        Sites to fit. ``site.fits`` is replaced.
    entries : sequence of ModelEntry
        Which models.
    method : {'nuts', 'advi'}, default 'nuts'
        Fitting method.
    seed : int
        RNG seed.
    zero_flow : optional
        Stage-of-zero-flow handling, forwarded to each fit.
    nuts_sampler : str, optional
        Which NUTS implementation.
    """
    total = len(sites)
    for position, site in enumerate(sites, 1):
        if len(site.sample) == 0:
            site.fits = []
            log.info("[%d/%d] %s: empty sample (%s)", position, total, site.sample_id,
                     site.sample.skipped or "no measurements")
            continue
        started = time.perf_counter()
        # with_log_likelihood is what makes the ELPD comparison possible at all
        site.fits = [entry.fit(site.sample, method=method, seed=seed,
                               zero_flow=zero_flow, nuts_sampler=nuts_sampler,
                               with_log_likelihood=True) for entry in entries]
        succeeded = sum(fit.ok for fit in site.fits)
        log.info("[%d/%d] %s: n=%d, fitted %d of %d model(s) in %.1fs", position,
                 total, site.sample_id, len(site.sample), succeeded, len(site.fits),
                 time.perf_counter() - started)


def attach_references(sites, refresh: bool = False) -> None:
    """Attach each USGS gage's published rating as a comparison reference, in place.

    Only sites whose stage is on a USGS gage-height axis get one - that is what the
    published curve is indexed by, so comparing anything else against it would be
    comparing two different stage axes.

    Parameters
    ----------
    sites : sequence of SiteRating
        Sites to annotate.
    refresh : bool, default False
        Refetch instead of using the cache.
    """
    for site in sites:
        site.reference = None
        if not site.gage or site.source not in ("usgs_gage", "colocated"):
            continue
        if len(site.sample) == 0:
            continue
        site.reference = usgs.published_reference(site.gage, site.sample,
                                                 refresh=refresh)


def score_sites(sites) -> None:
    """Compute the Bayesian comparison scores for every successful fit, in place.

    Runs in this process because PSIS-LOO and WAIC need each fit's posterior in
    memory.

    Parameters
    ----------
    sites : sequence of SiteRating
        Sites whose fits are scored.
    """
    for site in sites:
        for fit in site.fits:
            if not fit.ok:
                continue
            try:
                entry = catalog.get(fit.key)
            except KeyError:
                fit.bayes = metrics_module.unavailable(f"unknown model {fit.key}")
                continue
            fit.bayes = entry.bayes_metrics(fit)
            if settings.RELOO:
                fit.bayes = refine_elpd(fit, site.sample, entry)
            scores = fit.bayes
            if np.isfinite(scores.get("elpd_loo", np.nan)):
                log.info("  %s / %s: ELPD_LOO=%.1f±%.1f  p_loo=%.1f  k_max=%.2f%s",
                         site.sample_id, fit.key, scores["elpd_loo"],
                         scores["se_loo"], scores["p_loo"], scores["pareto_k_max"],
                         "  [unreliable]" if scores.get("pct_k_high") else "")
            elif scores.get("note"):
                log.debug("  %s / %s: no ELPD (%s)", site.sample_id, fit.key,
                          scores["note"])


#: Identity columns carried on every results row.
_IDENTITY_COLUMNS = ("sample_id", "source", "label", "station", "gage", "group",
                     "stage_label")


def results_frame(sites) -> pd.DataFrame:
    """One row per curve per site, including the ones that did not fit.

    Parameters
    ----------
    sites : sequence of SiteRating
        The fitted sites.

    Returns
    -------
    pandas.DataFrame
        The identity columns, then the scores from
        :func:`limnotech_rating_curves.evaluate.metrics.comparison_rows`, then a row for
        every skipped or failed model with its reason in ``note``.
    """
    rows = []
    for site in sites:
        identity = {column: getattr(site, column) for column in _IDENTITY_COLUMNS}
        for row in metrics_module.comparison_rows(site):
            rows.append({**identity, **row})
        for fit in site.fits:
            if fit.ok:
                continue
            rows.append({**identity, "model": fit.key, "model_label": fit.label,
                         "family": fit.family, "n": fit.n, "status": fit.status,
                         "note": fit.reason})
    return pd.DataFrame(rows)


def cross_validate_sites(sites, models=None, *, scheme: str = "auto", holdout=None,
                         n_splits=None, n_train=None, seed: int = settings.SEED,
                         fold_method: str = "advi", nuts_sampler=None) -> dict:
    """Cross-validate every site with enough measurements.

    Parameters
    ----------
    sites : sequence of SiteRating
        The sites.
    models : optional
        Which models. ``None`` uses the curated default set.
    scheme : {'auto', 'holdout', 'loo'}, default 'auto'
        Chosen per site by sample size when ``"auto"``, which is what gets a MAGL
        sensor leave-one-out and a USGS gage the holdout sweep in the same run.
    holdout, n_splits, n_train, seed, fold_method, nuts_sampler
        Forwarded to :func:`limnotech_rating_curves.cross_validate`.

    Returns
    -------
    dict
        ``{sample_id: CrossValidation}``, omitting sites that produced no folds.
    """
    results = {}
    candidates = [site for site in sites if len(site.sample) >= settings.CV_MIN_POINTS]
    for position, site in enumerate(candidates, 1):
        started = time.perf_counter()
        result = crossval.cross_validate(
            site.sample, models=models, scheme=scheme, holdout=holdout,
            n_splits=n_splits, n_train=n_train, seed=seed, fold_method=fold_method,
            nuts_sampler=nuts_sampler)
        if not result.curves:
            continue
        results[site.sample_id] = result
        log.info("[cv %d/%d] %s: %s, %d fold fit(s) in %.1fs", position,
                 len(candidates), site.sample_id, result.scheme["name"],
                 len(result.curves), time.perf_counter() - started)
    return results


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def run(models=None, *, samples=None, sites=None, sources=None, min_points: int = 3,
        refresh: bool = False, method: str = "nuts", seed: int = settings.SEED,
        zero_flow=None, nuts_sampler=None, cross_validate: bool = False,
        cv_scheme: str = "auto", cv_splits=None, cv_holdout=None, cv_train_n=None,
        write_csv: bool = True, write_png: bool = False,
        write_posteriors: bool = True, output_dir=None, map_html=None) -> Report:
    """Fit, score and map a set of sites.

    Parameters
    ----------
    models : optional
        Which models to fit - keys, or a group (``"all"``, ``"bdrc"``,
        ``"ratingcurve"``). ``None`` fits the curated default set.
    samples : sequence of Sample or dict, optional
        Run these samples instead of the MAGL network. See
        :func:`sites_from_samples`.
    sites : sequence of str, optional
        MAGL-network selection tokens (sensor, co-located station, cluster, gage
        id). ``None`` runs every site.
    sources : sequence of str, optional
        Restrict to these sample sources.
    min_points : int, default 3
        Fewest discharge measurements before a MAGL sensor is included.
    refresh : bool, default False
        Refetch instead of using the caches.
    method : {'nuts', 'advi'}, default 'nuts'
        All-data fitting method. NUTS by default, because the Bayesian comparison
        is only valid on a properly sampled posterior.
    seed : int
        RNG seed.
    zero_flow : optional
        Stage-of-zero-flow handling, forwarded to every fit.
    nuts_sampler : str, optional
        Which NUTS implementation.
    cross_validate : bool, default False
        Also run the cross-validation sweep and build the map's second pane. This is
        the slow part of a run.
    cv_scheme : {'auto', 'holdout', 'loo'}, default 'auto'
        Cross-validation scheme.
    cv_splits, cv_holdout, cv_train_n : optional
        Cross-validation settings.
    write_csv : bool, default True
        Write ``results.csv`` and ``cross_validation.csv``.
    write_png : bool, default False
        Write the static figures.
    write_posteriors : bool, default True
        Save each fitted rating to ``<output_dir>/fitted_curves`` as a manifest and
        its posterior, which is what :func:`limnotech_rating_curves.load_ratings`
        reads back. See :mod:`limnotech_rating_curves.exports`.
    output_dir : path-like, optional
        Where the outputs go. Defaults to ``settings.OUTPUT_DIR``.
    map_html : path-like, optional
        Where the map goes. Defaults to ``<output_dir>/rating_curves_map.html``.

    Returns
    -------
    Report

    Examples
    --------
    >>> report = run(models="all", sites=["SR"], cross_validate=True)  # doctest: +SKIP
    >>> report.results.head()                                           # doctest: +SKIP
    """
    output_dir = Path(settings.OUTPUT_DIR if output_dir is None else output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = catalog.select(models)
    log.info("models: %s", ", ".join(entry.key for entry in entries))

    log.info("== assembling samples ==")
    if samples is not None:
        site_ratings = sites_from_samples(samples)
    else:
        site_ratings = assemble_magl_sites(sites=sites, sources=sources,
                                          min_points=min_points, refresh=refresh)
    log.info("assembled %d site(s)", len(site_ratings))

    log.info("== fitting on all data (%s) ==", method.upper())
    fit_sites(site_ratings, entries, method=method, seed=seed, zero_flow=zero_flow,
              nuts_sampler=nuts_sampler)

    log.info("== attaching published reference ratings ==")
    attach_references(site_ratings, refresh=refresh)

    log.info("== Bayesian model comparison (PSIS-LOO / WAIC) ==")
    score_sites(site_ratings)

    results = results_frame(site_ratings)
    if write_csv and not results.empty:
        path = output_dir / "results.csv"
        results.to_csv(path, index=False)
        log.info("results -> %s (%d rows)", path, len(results))
    _log_summary(results)

    if write_posteriors:
        posterior_dir = output_dir / "fitted_curves"
        # The manifest is what makes an export loadable: a bare .nc records draws over
        # parameter names and nothing that says which site, which stage axis or what
        # the curve was. Writing the posterior alone (which this used to do) left a
        # directory that load_ratings() reads as empty.
        written = []
        for site in site_ratings:
            written.extend(exports.save_site(site, posterior_dir,
                                             require_posterior=False))
        with_draws = sum(1 for record in written if record["posterior"])
        log.info("saved ratings -> %s (%d manifest(s), %d with posterior draws)",
                 posterior_dir, len(written), with_draws)

    cv_results, cv_payload = {}, None
    if cross_validate:
        log.info("== cross-validation sweep ==")
        cv_results = cross_validate_sites(
            site_ratings, models=models, scheme=cv_scheme, holdout=cv_holdout,
            n_splits=cv_splits, n_train=cv_train_n, seed=seed,
            nuts_sampler=nuts_sampler)
        cv_payload = {sample_id: result.to_map_payload()
                      for sample_id, result in cv_results.items()}
        if write_csv and cv_results:
            frames = []
            for sample_id, result in cv_results.items():
                frame = result.folds.copy()
                frame.insert(0, "sample_id", sample_id)
                frame.insert(1, "scheme", result.scheme["name"])
                frames.append(frame)
            path = output_dir / "cross_validation.csv"
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            log.info("cross-validation -> %s", path)

    if write_png:
        log.info("== static figures ==")
        figure_dir = output_dir / "figures"
        for site in site_ratings:
            if site.fits:
                plots.save_site_figure(site, figure_dir)
            result = cv_results.get(site.sample_id)
            if result:
                plots.save_fold_figures(site, result.curves, figure_dir)
        plots.write_figures_readme(figure_dir)
        log.info("figures -> %s", figure_dir)

    log.info("== interactive map ==")
    map_path = mapview.build_map(
        site_ratings, map_html or (output_dir / "rating_curves_map.html"),
        cv_data=cv_payload)

    return Report(map_path=Path(map_path), results=results, sites=site_ratings,
                  cross_validation=cv_results)


def _log_summary(results: pd.DataFrame) -> None:
    """Log the median fit and predictive score per source and model."""
    if results.empty or "status" not in results:
        return
    succeeded = results[results["status"] == "ok"]
    if succeeded.empty or "elpd_loo" not in succeeded:
        return
    summary = (succeeded.groupby(["source", "model"])[["nse", "elpd_loo"]]
               .median().round(2))
    log.info("median NSE / ELPD_LOO by source and model:\n%s", summary.to_string())
