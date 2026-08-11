import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .. import settings
from ..core import Sample
from ..exports import load_rating

log = logging.getLogger(__name__)

#: Observed working set of one worker: its own PyTensor graphs and posterior draws.
#: Used only to warn, never to silently cap what the caller asked for.
WORKER_MEMORY_GB = 1.0


# ==============================================================================
# what to fit
# ==============================================================================

def _as_sample(value, site_id, stage_datum=None) -> Sample:
    if isinstance(value, Sample):
        return value
    return Sample.of(value, site_id=str(site_id), stage_datum=stage_datum)


def collect_samples(samples, *, discharge=None, stage_datum=None) -> dict:
    """Normalize whatever a caller passed into ``{site_id: Sample}``.

    Parameters
    ----------
    samples : dict, list, Sample, DataFrame, or fb_pagaia Network
        See the module docstring.
    discharge : dict, optional
        Only for a Network: discharge measurements per station id. A station with no
        discharge cannot carry a rating and is skipped with a warning, since a
        network normally holds more stations than you have discharge for.
    stage_datum : optional
        Applied to every sample built here. See
        :mod:`limnotech_rating_curves.data.datum`.

    Returns
    -------
    dict
        ``{site_id: Sample}``, insertion-ordered.
    """
    stations = getattr(samples, "stations", None)
    if isinstance(stations, dict):
        return _samples_from_network(stations, discharge, stage_datum)

    if isinstance(samples, dict):
        return {str(key): _as_sample(value, key, stage_datum)
                for key, value in samples.items()}

    if isinstance(samples, Sample):
        return {samples.site_id or "sample": samples}

    if isinstance(samples, (list, tuple)):
        built = {}
        for index, item in enumerate(samples):
            sample = _as_sample(item, f"sample_{index}", stage_datum)
            built[sample.site_id or f"sample_{index}"] = sample
        return built

    sample = _as_sample(samples, "sample", stage_datum)
    return {sample.site_id or "sample": sample}


def _samples_from_network(stations: dict, discharge, stage_datum) -> dict:
    """One sample per station in an fb_pagaia Network, given discharge per station."""
    from ..data import pagaia

    if not discharge:
        raise ValueError(
            "a Network carries stage but not discharge, so it cannot be fitted on "
            "its own. Pass discharge={station_id: measurements} keyed the same way "
            "as network.stations, or build the samples yourself with "
            "lrc.station_sample() and pass those.")
    built = {}
    for station_id, station in stations.items():
        measured = discharge.get(str(station_id), discharge.get(station_id))
        if measured is None:
            log.warning("no discharge for station %s, so it is skipped", station_id)
            continue
        sample = pagaia.station_sample(station, measured, stage_datum=stage_datum)
        if len(sample) == 0:
            log.warning("station %s built no fittable measurements, so it is skipped "
                        "(%s)", station_id, sample.skipped or "no reason recorded")
            continue
        # the key the caller used is what the results are indexed by, so it wins over
        # the name the station reports itself under
        sample.site_id = str(station_id)
        built[str(station_id)] = sample
    if not built:
        raise ValueError("no station in this Network produced a fittable sample - "
                         "either none of them had discharge passed for it, or no "
                         "discharge measurement had a stage reading to match. The "
                         "warnings above say which, per station.")
    return built


def default_workers(n_tasks=None) -> int:
    """One worker per core bar one, held under what memory can hold.

    Derived rather than fixed, because this runs on a 22-core workstation and on
    whatever else the package is installed to. Never more workers than tasks.
    """
    workers = max(1, (os.cpu_count() or 2) - 1)
    try:
        import psutil
        # leave 4 GB for the OS and this process
        affordable = int(max(1, (psutil.virtual_memory().total / 2 ** 30 - 4)
                             // WORKER_MEMORY_GB))
        workers = min(workers, affordable)
    except ImportError:
        pass
    return workers if n_tasks is None else max(1, min(workers, int(n_tasks)))


# ==============================================================================
# the worker
# ==============================================================================

def _init_worker() -> None:
    """Re-apply the environment in a spawned child, which starts from scratch.

    A spawned interpreter inherits nothing set after import, and both of these have
    to be in place before numpy loads. One BLAS thread per worker as well: the
    parallelism is across fits, so letting each of N workers open its own thread pool
    would oversubscribe the machine N-fold.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


def _fit_one(task: dict) -> dict:
    """Fit one (site, model) and write it to disk. Runs in a worker process.

    Returns a small picklable record - never the fitted model, which could not be
    sent back even if we wanted it.
    """
    from ..exports import save_rating
    from ..ratings import fit_rating

    started = time.perf_counter()
    record = {"site": task["site"], "model": task["model"], "n": task["n"],
              "status": "failed", "reason": "", "manifest": None, "posterior": None,
              "seconds": 0.0}
    try:
        sample = Sample.of(pd.DataFrame(task["frame"]), site_id=task["site"])
        rating = fit_rating(sample, model=task["model"], cores=1, **task["fit_kwargs"])
        if not rating.fitted:
            result = rating.result
            record["status"] = getattr(result, "status", "failed")
            record["reason"] = getattr(result, "reason", "the fit did not succeed")
            return record
        written = save_rating(rating, task["directory"], sample_id=task["site"])
        record.update(status="ok", manifest=written["manifest"],
                      posterior=written["posterior"])
    except Exception as exc:  # noqa: BLE001 - one bad fit must not stop the sweep
        record["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["seconds"] = round(time.perf_counter() - started, 2)
    return record


# ==============================================================================
# the sweep
# ==============================================================================

def fit_many(samples, models=None, *, directory=None, workers=None, discharge=None,
             stage_datum=None, method: str = "nuts", seed: int = settings.SEED,
             zero_flow=None, nuts_sampler=None, min_points: int = 3,
             **fit_kwargs) -> "RatingCollection":
    """Fit every model to every sample, in parallel, and collect the results.

    Parameters
    ----------
    samples : dict, list, Sample, DataFrame, or fb_pagaia Network
        What to fit. See the module docstring.
    models : sequence of str or str, optional
        Model keys or a group name (``"all"``, ``"bdrc"``, ``"ratingcurve"``,
        ``"default"``). Defaults to the curated comparison set.
    directory : path-like, optional
        Where the manifests and posteriors are written. Defaults to
        ``settings.OUTPUT_DIR / "ratings"``. Existing files for the same
        (site, model) are overwritten.
    workers : int, optional
        Worker processes. Defaults to one per core bar one, capped by memory and by
        the number of tasks. ``workers=1`` fits in this process - same files, same
        return type, no pool.
    discharge : dict, optional
        Discharge per station, when `samples` is a Network.
    stage_datum : optional
        Put stage on a gage-height reference before fitting.
    method : {'nuts', 'advi'}, default 'nuts'
        Full MCMC or the fast variational fit.
    seed : int
        RNG seed, the same for every fit so the sweep is reproducible.
    zero_flow : optional
        Stage of zero flow. See :func:`limnotech_rating_curves.fit_rating`.
    nuts_sampler : str, optional
        Which NUTS implementation.
    min_points : int, default 3
        Samples with fewer measurements than this are skipped before any fitting,
        with the count recorded - a rating needs at least three points and there is
        no sense paying for a worker to discover that.
    **fit_kwargs
        Forwarded to :func:`limnotech_rating_curves.fit_rating`.

    Returns
    -------
    RatingCollection

    Notes
    -----
    One task is one (site, model) pair, and workers are reused, so each worker
    compiles a given bdrc variant at most once no matter how many sites it fits -
    see ``PERFORMANCE.md``. ``cores=1`` is forced inside the workers so the sampler
    does not spawn a nested pool.
    """
    from ..models import catalog

    collected = collect_samples(samples, discharge=discharge, stage_datum=stage_datum)
    entries = catalog.select(models)
    directory = Path(settings.OUTPUT_DIR / "ratings" if directory is None
                     else directory)
    directory.mkdir(parents=True, exist_ok=True)

    fit_kwargs = dict(fit_kwargs)
    fit_kwargs.update(method=method, seed=seed, zero_flow=zero_flow,
                      nuts_sampler=nuts_sampler)

    tasks, skipped = [], []
    for site_id, sample in collected.items():
        if len(sample) < min_points:
            skipped.append({"site": site_id, "model": None, "n": len(sample),
                            "status": "skipped", "reason":
                            f"{len(sample)} measurements, fewer than the {min_points} "
                            f"a rating needs", "manifest": None, "posterior": None,
                            "seconds": 0.0})
            continue
        frame = sample.to_frame()
        for entry in entries:
            tasks.append({"site": site_id, "model": entry.key, "n": len(sample),
                          "frame": frame.to_dict(orient="list"),
                          "directory": str(directory), "fit_kwargs": fit_kwargs})

    # Longest first: the gages carry 25-185 measurements and the sensors three to
    # seven, so starting the big ones early is what stops the tail of the run being
    # one worker on a gage while the rest idle.
    tasks.sort(key=lambda task: task["n"], reverse=True)
    if not tasks:
        log.warning("nothing to fit: %d sample(s), all below min_points=%d",
                    len(collected), min_points)
        return RatingCollection(skipped, directory)

    workers = default_workers(len(tasks)) if workers is None else max(1, int(workers))
    log.info("fitting %d model(s) across %d site(s) = %d fits on %d worker(s) -> %s",
             len(entries), len(collected), len(tasks), workers, directory)

    started = time.perf_counter()
    records = list(skipped)
    if workers == 1:
        _init_worker()
        for task in tasks:
            records.append(_report(_fit_one(task)))
    else:
        try:
            pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
            futures = {pool.submit(_fit_one, task): task for task in tasks}
        except RuntimeError as exc:
            # Windows spawns workers by re-importing __main__, so a script that calls
            # this at module level re-enters itself. The multiprocessing traceback for
            # it is a wall of text that never mentions the fix.
            if "bootstrapping phase" not in str(exc):
                raise
            raise RuntimeError(
                "fit_many() started worker processes from a script that has no "
                "`if __name__ == \"__main__\":` guard. On Windows each worker "
                "re-imports the main module, so without the guard the script runs "
                "itself again. Put your call inside a main guard, or pass workers=1 "
                "to fit in this process.") from exc
        with pool:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    records.append(_report(future.result()))
                except Exception as exc:  # noqa: BLE001 - a worker died outright
                    log.error("worker fitting %s / %s died: %s: %s", task["site"],
                              task["model"], type(exc).__name__, exc)
                    records.append({"site": task["site"], "model": task["model"],
                                    "n": task["n"], "status": "failed",
                                    "reason": f"worker died: {type(exc).__name__}: "
                                              f"{exc}",
                                    "manifest": None, "posterior": None,
                                    "seconds": 0.0})

    elapsed = time.perf_counter() - started
    ok = sum(1 for record in records if record["status"] == "ok")
    log.info("%d of %d fits succeeded in %.1f s (%.1f s of fitting per fit)",
             ok, len(tasks), elapsed,
             sum(record["seconds"] for record in records) / max(1, len(tasks)))
    return RatingCollection(records, directory)


def _report(record: dict) -> dict:
    if record["status"] == "ok":
        log.info("  %s / %s: %.1f s", record["site"], record["model"],
                 record["seconds"])
    else:
        log.warning("  %s / %s: %s (%s)", record["site"], record["model"],
                    record["status"], record["reason"][:120])
    return record


# ==============================================================================
# the result
# ==============================================================================

class RatingCollection:
    """Every rating a sweep produced, indexed by site and model.

    Indexing is the point:

    ============================  =============================================
    ``results["04176356"]``       ``{model_key: SavedRating}`` for one site
    ``results["04176356", "spline"]``  one
                                  :class:`~limnotech_rating_curves.exports.SavedRating`
    ``results.metrics``           one row per fit, with its scores
    ``results.best("04176356")``  that site's best by predictive score
    ``results.best_per_site()``   the same for every site, as a table
    ``results.failures``          what did not fit, and why
    ============================  =============================================

    Attributes
    ----------
    directory : pathlib.Path
        Where the manifests and posteriors live.
    records : list of dict
        One per attempted fit: site, model, status, reason, paths, seconds.
    """

    def __init__(self, records, directory):
        self.records = list(records)
        self.directory = Path(directory)
        self._ratings = {}
        for record in self.records:
            if record["status"] != "ok" or not record.get("manifest"):
                continue
            try:
                saved = load_rating(record["manifest"])
            except Exception as exc:  # noqa: BLE001
                log.warning("fitted %s / %s but could not read it back (%s: %s)",
                            record["site"], record["model"], type(exc).__name__, exc)
                continue
            self._ratings[(record["site"], record["model"])] = saved

    # -- loading a previous run ----------------------------------------------

    @classmethod
    def from_directory(cls, directory) -> "RatingCollection":
        """Rebuild a collection from files a previous sweep wrote.

        The same shape as a fresh sweep's return value, so analysis code does not
        care whether the fits just happened.
        """
        from ..exports import MANIFEST_SUFFIX, load_ratings

        directory = Path(directory)
        records = []
        for (site, model), saved in load_ratings(directory).items():
            records.append({
                "site": site, "model": model,
                "n": saved.manifest.get("site", {}).get("n_measurements", 0),
                "status": "ok", "reason": "",
                "manifest": str(saved.path),
                "posterior": (None if saved.posterior_path is None
                              else str(saved.posterior_path)),
                "seconds": float("nan")})
        if not records:
            log.warning("no %s files in %s", MANIFEST_SUFFIX, directory)
        return cls(records, directory)

    # -- access ---------------------------------------------------------------

    def __getitem__(self, key):
        if isinstance(key, tuple):
            site, model = key
            try:
                return self._ratings[(str(site), str(model))]
            except KeyError:
                raise KeyError(
                    f"no fitted {model} for {site}. "
                    f"{self._why_missing(str(site), str(model))}") from None
        for_site = self.site(str(key))
        if not for_site:
            raise KeyError(f"no fitted models for {key}. "
                           f"{self._why_missing(str(key), None)}")
        return for_site

    def _why_missing(self, site, model) -> str:
        """The recorded reason, so a KeyError explains itself."""
        for record in self.records:
            if record["site"] == site and (model is None or record["model"] == model):
                if record["status"] != "ok":
                    return (f"It was {record['status']}: {record['reason']}")
        sites = ", ".join(sorted(self.sites)[:8]) or "none"
        return f"Sites in this collection: {sites}"

    def site(self, site) -> dict:
        """``{model_key: SavedRating}`` for one site, in catalog order."""
        return {model: saved for (found, model), saved in self._ratings.items()
                if found == str(site)}

    def get(self, site, model, default=None):
        """One rating, or `default` if it is not there."""
        return self._ratings.get((str(site), str(model)), default)

    @property
    def ratings(self) -> dict:
        """Every rating, keyed ``(site, model)``."""
        return dict(self._ratings)

    @property
    def sites(self) -> list:
        """Sites with at least one fitted model."""
        return sorted({site for site, _ in self._ratings})

    @property
    def model_keys(self) -> list:
        """Models that fitted anywhere."""
        return sorted({model for _, model in self._ratings})

    def __len__(self):
        return len(self._ratings)

    def __iter__(self):
        return iter(self._ratings.values())

    def __contains__(self, key):
        if isinstance(key, tuple):
            site, model = key
            return (str(site), str(model)) in self._ratings
        return bool(self.site(key))

    # -- tables ---------------------------------------------------------------

    @property
    def metrics(self) -> pd.DataFrame:
        """One row per fitted (site, model), best predictive score first.

        Columns: ``site``, ``model``, ``label``, ``n``, the fit scores, the ELPD
        block, ``worst_r_hat``, ``converged``, ``seconds`` and ``manifest``.
        """
        seconds = {(record["site"], record["model"]): record["seconds"]
                   for record in self.records}
        rows = []
        for (site, model), saved in self._ratings.items():
            scores = saved.metrics
            convergence = saved.convergence
            rows.append({
                "site": site, "model": model, "label": saved.label,
                "n": scores.get("n"),
                "nse": scores.get("nse"), "rmse": scores.get("rmse"),
                "pbias_pct": scores.get("pbias_pct"), "r2_log": scores.get("r2_log"),
                "elpd_loo": scores.get("elpd_loo"), "se_loo": scores.get("se_loo"),
                "pareto_k_max": scores.get("pareto_k_max"),
                "worst_r_hat": convergence.get("worst_r_hat"),
                "converged": convergence.get("converged"),
                "seconds": seconds.get((site, model)),
                "manifest": str(saved.path)})
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["site", "elpd_loo"], ascending=[True, False],
                                 na_position="last").reset_index(drop=True)

    @property
    def failures(self) -> pd.DataFrame:
        """Every fit that did not produce a rating, with the reason it gives."""
        rows = [record for record in self.records if record["status"] != "ok"]
        return pd.DataFrame(rows, columns=["site", "model", "n", "status", "reason",
                                           "seconds"]) if rows else pd.DataFrame(
            columns=["site", "model", "n", "status", "reason", "seconds"])

    # -- ranking --------------------------------------------------------------

    def best(self, site):
        """The best-scoring model for one site.

        Ranked by ``elpd_loo`` and then NSE, the same lexicographic rule
        :class:`~limnotech_rating_curves.ratings.RatingSet` uses, so a sweep and a
        single-site comparison agree on "best".

        Read the ranking with care when some models have no ELPD - every fit made
        with ``method="advi"``, and the least-squares quadratic always. Those fall
        back to NSE, which is measured in-sample and so favours whichever model bends
        most; that is a goodness-of-fit ordering, not a predictive one. For a
        predictive comparison, fit with ``method="nuts"`` and compare models that
        report an ELPD.

        Returns
        -------
        SavedRating or None
            None when nothing fitted for that site.
        """
        candidates = self.site(site)
        if not candidates:
            return None

        def score(saved):
            scores = saved.metrics
            elpd = scores.get("elpd_loo")
            nse = scores.get("nse")
            return (-np.inf if elpd is None else elpd,
                    -np.inf if nse is None else nse)

        return max(candidates.values(), key=score)

    def best_per_site(self) -> pd.DataFrame:
        """One row per site: which model won, and its scores."""
        rows = []
        for site in self.sites:
            saved = self.best(site)
            if saved is None:
                continue
            scores = saved.metrics
            rows.append({"site": site, "model": saved.name, "label": saved.label,
                         "n": scores.get("n"), "nse": scores.get("nse"),
                         "r2_log": scores.get("r2_log"),
                         "elpd_loo": scores.get("elpd_loo"),
                         "converged": saved.convergence.get("converged")})
        return pd.DataFrame(rows)

    def curves(self) -> pd.DataFrame:
        """Every fitted curve in one long frame - ``site``, ``model``, then the table.

        The shape plotting and mapping code wants: one call, then group by site.
        """
        frames = []
        for (site, model), saved in self._ratings.items():
            curve = saved.curve
            if curve.empty:
                continue
            curve = curve.copy()
            curve.insert(0, "site", site)
            curve.insert(1, "model", model)
            frames.append(curve)
        return (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=["site", "model", "stage_ft",
                                           "discharge_cfs"]))

    def __repr__(self):
        failed = len(self.records) - len(self._ratings)
        return (f"RatingCollection({len(self._ratings)} ratings over "
                f"{len(self.sites)} site(s)"
                f"{f', {failed} not fitted' if failed else ''}, "
                f"dir={self.directory.name!r})")
