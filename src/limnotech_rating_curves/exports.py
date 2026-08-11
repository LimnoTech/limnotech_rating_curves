import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings
from .core import Metrics

log = logging.getLogger(__name__)

#: Bumped when the manifest layout changes in a way a reader must notice. Version 2
#: added ``posterior_sha256`` and the canonical model-key normalization of file names.
MANIFEST_VERSION = 2

#: Suffixes of the two files, so callers and tests agree on them.
MANIFEST_SUFFIX = ".rating.json"
POSTERIOR_SUFFIX = ".nc"

#: Bytes read per digest update. The posteriors are a few MB at most, but streaming
#: costs nothing and keeps the checksum honest on a large spline fit.
_DIGEST_BLOCK = 1 << 20


def canonical_model_key(key) -> str:
    """The catalog's own spelling of a model key.

    Older runs wrote ``PowerLawRating_1seg`` where the catalog now says
    ``power_law``, and a file named one way cannot be looked up the other. Every
    name this module writes goes through here, so a directory of exports is keyed
    the same way :func:`limnotech_rating_curves.models.catalog.get` is.

    Parameters
    ----------
    key : str
        A model key, current or legacy.

    Returns
    -------
    str
        The canonical key, or `key` unchanged when the catalog does not know it -
        an unregistered model is still exportable.
    """
    from .models import catalog
    return catalog.LEGACY_KEYS.get(str(key), str(key))


def _file_stem(sample_id, model_key) -> str:
    safe = str(sample_id).replace(":", "__").replace("/", "-").replace("\\", "-")
    return f"{safe}__{canonical_model_key(model_key)}"


def _sha256(path) -> str:
    """SHA-256 of a file, as a hex string."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_DIGEST_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (Path, datetime)):
        return str(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _frame_records(frame) -> list:
    if frame is None:
        return []
    return [_plain(row) for row in frame.to_dict(orient="records")]


class FittedRating:
    """The estimator surface a manifest needs, over a bare fit record.

    A :class:`~limnotech_rating_curves.ratings.RatingModel` is what the library hands
    a user, but the batch pipeline never builds one: it holds a
    :class:`~limnotech_rating_curves.core.FitResult` and the
    :class:`~limnotech_rating_curves.core.Sample` it came from. Those two carry
    everything a manifest records, so rather than duplicate
    :func:`rating_manifest` for the batch path this wraps them in the handful of
    attributes it reads. Every export in the package then goes through one writer.

    Parameters
    ----------
    fit : FitResult
        A completed fit.
    sample : Sample
        The measurements it was fitted to.

    Examples
    --------
    >>> save_rating(FittedRating(fit, site.sample), "out/")   # doctest: +SKIP
    """

    def __init__(self, fit, sample):
        self.result = fit
        self.sample = sample

    @property
    def name(self) -> str:
        return canonical_model_key(self.result.key)

    @property
    def label(self) -> str:
        return self.result.label

    @property
    def family(self) -> str:
        return self.result.family

    @property
    def fitted(self) -> bool:
        return bool(self.result is not None and self.result.ok)

    @property
    def entry(self):
        """The catalog entry, or None for a model the catalog does not list."""
        from .models import catalog
        try:
            return catalog.get(self.result.key)
        except KeyError:
            return None

    @property
    def metrics(self) -> Metrics:
        return Metrics.from_fit(self.result)

    def curve(self):
        """The fitted curve as the fit recorded it."""
        if self.result.curve is None:
            return None
        return self.result.curve.sort_values("stage_ft").reset_index(drop=True)

    def equation(self) -> str:
        rating = self.result.rating
        return str(rating.equation()) if hasattr(rating, "equation") else ""

    def summary(self):
        from .evaluate import diagnostics
        return diagnostics.posterior_summary(self.result)

    def diagnostics(self):
        from .evaluate import diagnostics
        return diagnostics.convergence(self.result)

    def save_posterior(self, directory, sample_id=None):
        entry = self.entry
        name = sample_id or (self.sample.site_id if self.sample else "sample")
        if entry is None:
            from .models import catalog
            return catalog.save_posterior(self.result, directory, name or "sample")
        return entry.save_posterior(self.result, directory, name or "sample")

    def __repr__(self):
        return f"FittedRating({self.name}, fitted={self.fitted})"


def rating_manifest(rating, sample_id=None) -> dict:
    """Everything about a fitted rating except the posterior draws, as a dict.

    Parameters
    ----------
    rating : RatingModel
        A fitted model.
    sample_id : str, optional
        Identifies the site. Defaults to the sample's own ``site_id``.

    Returns
    -------
    dict
        The manifest :func:`save_rating` writes. Useful on its own when you want the
        provenance without writing files.
    """
    fit = rating.result
    sample = rating.sample
    metrics = rating.metrics if rating.fitted else Metrics(0, *[float("nan")] * 4)
    convergence = _convergence_summary(rating)
    site = str(sample_id or (sample.site_id if sample is not None else "") or "sample")

    return {
        "format": "limnotech-rating-curve",
        "format_version": MANIFEST_VERSION,
        "package_version": _package_version(),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site": {
            "id": site,
            "source": getattr(sample, "source", "") if sample is not None else "",
            "n_measurements": int(getattr(sample, "__len__", lambda: 0)()),
        },
        "model": {
            "key": rating.name,
            "label": rating.label,
            "family": rating.family,
            "equation": _equation(rating),
            "has_posterior": bool(getattr(rating.entry, "has_posterior", True)),
        },
        # Every knob that changes the numbers, so a fit can be reproduced exactly
        # and so a file can be audited without rerunning it.
        "sampler": _plain(dict(fit.config or {})),
        "units": {"stage": "ft", "discharge": "cfs",
                  "stage_label": getattr(sample, "stage_label", "stage (ft)")
                                 if sample is not None else "stage (ft)",
                  "datum_note": getattr(sample, "datum_note", "")
                                if sample is not None else ""},
        "measurements": _frame_records(sample.to_frame() if sample is not None else None),
        "curve": _frame_records(rating.curve() if rating.fitted else None),
        "parameters": _parameter_records(rating),
        "metrics": _plain({
            "n": metrics.n, "nse": metrics.nse, "rmse": metrics.rmse,
            "pbias_pct": metrics.pbias_pct, "r2_log": metrics.r2_log,
            "elpd_loo": metrics.elpd_loo, "se_loo": metrics.se_loo,
            "p_loo": metrics.p_loo, "elpd_waic": metrics.elpd_waic,
            "pareto_k_max": metrics.pareto_k_max,
            # without these, a manifest cannot say whether its elpd_loo is the PSIS
            # approximation or one repaired by exact refits
            "elpd_loo_refits": (fit.bayes or {}).get("n_refits", 0),
            "elpd_loo_psis": (fit.bayes or {}).get("elpd_loo_psis")}),
        "convergence": _plain(convergence),
        "native": _plain(dict(fit.native or {})),
    }


def _package_version() -> str:
    from . import __version__
    return __version__


def _equation(rating) -> str:
    try:
        return str(rating.equation())
    except Exception:  # noqa: BLE001 - not every family exposes a closed form
        return ""


def _parameter_records(rating) -> list:
    """The posterior parameter summary, or an empty list when there is none."""
    if not rating.fitted:
        return []
    try:
        summary = rating.summary()
    except Exception as exc:  # noqa: BLE001
        log.debug("no parameter summary for %s (%s)", rating.name, exc)
        return []
    if summary is None or summary.empty:
        return []
    return [{"parameter": str(name), **_plain(row)}
            for name, row in summary.to_dict(orient="index").items()]


def _convergence_summary(rating) -> dict:
    """Worst R-hat and lowest ESS, so a reader can tell whether to trust the file.

    ``converged`` is None rather than False when R-hat could not be computed - a
    variational fit has no chains to compare, and "not measured" is not the same
    claim as "did not converge".
    """
    if not rating.fitted:
        return {"available": False}
    try:
        table = rating.diagnostics()
    except Exception as exc:  # noqa: BLE001
        log.debug("no convergence table for %s (%s)", rating.name, exc)
        return {"available": False}
    if table is None or table.empty:
        return {"available": False}
    worst_r_hat = float(table["r_hat"].max())
    measured = np.isfinite(worst_r_hat)
    return {"available": True,
            "worst_r_hat": worst_r_hat,
            "lowest_ess_bulk": float(table["ess_bulk"].min()),
            "converged": bool(table["converged"].all()) if measured else None,
            "note": "" if measured else
                    "R-hat is not defined for this fit (no MCMC chains), so "
                    "convergence was not measured rather than failed",
            "r_hat_threshold": settings.R_HAT_GOOD,
            "ess_threshold": settings.ESS_GOOD}


def save_rating(rating, directory=None, sample_id=None, *,
                require_posterior: bool = True) -> dict:
    """Write a fitted rating as a manifest and its posterior.

    Parameters
    ----------
    rating : RatingModel
        A fitted model.
    directory : path-like, optional
        Where to write. Defaults to ``settings.POSTERIOR_DIR``.
    sample_id : str, optional
        Identifies the site in the file names. Defaults to the sample's ``site_id``.
    require_posterior : bool, default True
        Raise if the posterior could not be written. The draws are the part that
        cannot be reconstructed, so a silent manifest-only export is treated as a
        failure rather than a partial success. Set False when you deliberately want
        the manifest alone - for a fit that crossed a process boundary, say, whose
        posterior no longer exists in memory.

        Models that have no posterior at all are exempt: the least-squares quadratic
        is fully described by its manifest, so a missing ``.nc`` there is
        completeness, not loss.

    Returns
    -------
    dict
        ``{"manifest": path, "posterior": path or None}``.

    Raises
    ------
    RuntimeError
        If `rating` is not fitted, or `require_posterior` is set and no posterior
        could be written.
    """
    if not rating.fitted:
        raise RuntimeError(f"{rating.name} is not fitted, so there is nothing to save")

    directory = Path(settings.POSTERIOR_DIR if directory is None else directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = rating_manifest(rating, sample_id=sample_id)
    stem = _file_stem(manifest["site"]["id"], rating.name)

    posterior_path = rating.save_posterior(directory, sample_id=manifest["site"]["id"])
    expects_posterior = getattr(rating.entry, "has_posterior", True)
    if posterior_path is None and require_posterior and expects_posterior:
        raise RuntimeError(
            f"the posterior for {rating.name} could not be written, so only the "
            f"manifest would survive and the draws would be lost. This happens when "
            f"a fit no longer holds its posterior - a fit that crossed a process "
            f"boundary, for instance. Refit in this process, or pass "
            f"require_posterior=False to accept a manifest-only export.")
    if posterior_path is not None:
        # the backend names the file from its own key; normalize it so a directory
        # of exports is keyed by the catalog's spelling throughout
        wanted = directory / f"{stem}{POSTERIOR_SUFFIX}"
        written = Path(posterior_path)
        if written.resolve() != wanted.resolve():
            written.replace(wanted)
            posterior_path = wanted
    manifest["posterior_file"] = (Path(posterior_path).name if posterior_path
                                  else None)
    manifest["posterior_sha256"] = (_sha256(posterior_path) if posterior_path
                                    else None)

    manifest_path = directory / f"{stem}{MANIFEST_SUFFIX}"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s%s", manifest_path.name,
             "" if posterior_path else " (no posterior)")
    return {"manifest": str(manifest_path),
            "posterior": None if posterior_path is None else str(posterior_path)}


def save_fit(fit, sample, directory=None, sample_id=None, **kwargs) -> dict:
    """Save one :class:`~limnotech_rating_curves.core.FitResult` and its sample.

    The batch pipeline's entry point into the export format: it holds fit records
    rather than estimators, and this wraps one in :class:`FittedRating` so it goes
    through :func:`save_rating` like everything else.

    Parameters
    ----------
    fit : FitResult
        A completed fit.
    sample : Sample
        The measurements it was fitted to.
    directory : path-like, optional
        Where to write.
    sample_id : str, optional
        Identifies the site in the file names. Defaults to the sample's ``site_id``.
    **kwargs
        Forwarded to :func:`save_rating`.

    Returns
    -------
    dict
        ``{"manifest": path, "posterior": path or None}``.
    """
    return save_rating(FittedRating(fit, sample), directory, sample_id=sample_id,
                       **kwargs)


def save_site(site, directory=None, **kwargs) -> list:
    """Save every successful fit at one :class:`~limnotech_rating_curves.core.SiteRating`.

    Parameters
    ----------
    site : SiteRating
        A fitted site. Fits that were skipped or failed are not written - there is
        no curve and no posterior to record.
    directory : path-like, optional
        Where to write.
    **kwargs
        Forwarded to :func:`save_rating`.

    Returns
    -------
    list of dict
        One ``{"manifest": ..., "posterior": ...}`` per fit written. A fit that could
        not be saved is logged and skipped, so one bad model does not cost the rest.
    """
    written = []
    for fit in site.fits:
        if not fit.ok:
            continue
        try:
            written.append(save_fit(fit, site.sample, directory,
                                    sample_id=site.sample_id, **kwargs))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not save %s / %s: %s: %s", site.sample_id, fit.key,
                        type(exc).__name__, exc)
    return written


def curve_table(saved_ratings) -> pd.DataFrame:
    """Every saved rating's fitted curve in one long table, for a spreadsheet.

    The manifest is the machine-readable record; this is the one a field engineer
    opens. One row per stage point per rating, so a curve can be looked up, plotted
    or pasted into Excel without a JSON reader.

    Parameters
    ----------
    saved_ratings : sequence or dict of SavedRating
        As returned by :func:`load_ratings` (a dict is read by its values).

    Returns
    -------
    pandas.DataFrame
        ``site``, ``model``, ``model_label``, ``family``, ``equation``,
        ``stage_ft``, ``discharge_cfs``, ``lower``, ``upper``, and the fit's ``nse``
        / ``rmse`` / ``r2_log`` repeated on every row so a single sheet is
        self-explanatory.
    """
    if isinstance(saved_ratings, dict):
        saved_ratings = list(saved_ratings.values())
    frames = []
    for saved in saved_ratings:
        curve = saved.curve
        if curve.empty:
            continue
        scores = saved.metrics
        frame = curve.copy()
        frame.insert(0, "site", saved.site_id)
        frame.insert(1, "model", saved.name)
        frame.insert(2, "model_label", saved.label)
        frame.insert(3, "family", saved.family)
        frame.insert(4, "equation", saved.equation)
        for name in ("nse", "rmse", "r2_log"):
            frame[name] = scores.get(name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_ratings(rating_set, directory=None, sample_id=None, **kwargs) -> list:
    """Save every fitted model in a :class:`~limnotech_rating_curves.ratings.RatingSet`.

    Parameters
    ----------
    rating_set : RatingSet
        The comparison to export.
    directory : path-like, optional
        Where to write.
    sample_id : str, optional
        Identifies the site in the file names.
    **kwargs
        Forwarded to :func:`save_rating`.

    Returns
    -------
    list of dict
        One ``{"manifest": ..., "posterior": ...}`` per model that saved. Models
        that could not be saved are logged and skipped, so one bad model does not
        cost you the rest of the export.
    """
    written = []
    for model in rating_set.models:
        try:
            written.append(save_rating(model, directory, sample_id=sample_id,
                                       **kwargs))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not save %s: %s: %s", model.name,
                        type(exc).__name__, exc)
    return written


class SavedRating:
    """A rating read back from disk: its provenance, its curve, and its draws.

    Predicts by interpolating the stored curve, which is what the map and the
    dashboard do with a live fit too. It is deliberately **not** a
    :class:`~limnotech_rating_curves.ratings.RatingModel`: it cannot be refitted,
    and it does not pretend to evaluate the model's parameters exactly.

    Attributes
    ----------
    manifest : dict
        Everything the file recorded.
    path : pathlib.Path
        The manifest's own path.
    """

    def __init__(self, manifest: dict, path):
        self.manifest = manifest
        self.path = Path(path)

    # -- identity -------------------------------------------------------------

    @property
    def site_id(self) -> str:
        return self.manifest.get("site", {}).get("id", "")

    @property
    def name(self) -> str:
        """The model key, e.g. ``"bdrc_gplm0"``."""
        return self.manifest.get("model", {}).get("key", "")

    @property
    def label(self) -> str:
        return self.manifest.get("model", {}).get("label", self.name)

    @property
    def family(self) -> str:
        return self.manifest.get("model", {}).get("family", "")

    @property
    def equation(self) -> str:
        return self.manifest.get("model", {}).get("equation", "")

    @property
    def sampler(self) -> dict:
        """How it was fitted: method, sampler, draws, chains, seed, zero flow."""
        return dict(self.manifest.get("sampler", {}))

    # -- the numbers ----------------------------------------------------------

    @property
    def curve(self) -> pd.DataFrame:
        """The fitted rating as a table, as it was written."""
        return pd.DataFrame(self.manifest.get("curve", []))

    @property
    def stage_range(self) -> tuple:
        """``(low, high)`` stage the stored curve covers.

        For the quadratic this is its effective range - the band the curve may be
        used over - which is why it is recorded rather than inferred.
        """
        effective = self.sampler.get("effective_range")
        if effective:
            return float(effective[0]), float(effective[1])
        curve = self.curve
        if curve.empty:
            return float("nan"), float("nan")
        stage = curve["stage_ft"].to_numpy(float)
        return float(np.nanmin(stage)), float(np.nanmax(stage))

    @property
    def measurements(self) -> pd.DataFrame:
        """The measurements it was fitted to."""
        return pd.DataFrame(self.manifest.get("measurements", []))

    @property
    def metrics(self) -> dict:
        return dict(self.manifest.get("metrics", {}))

    @property
    def convergence(self) -> dict:
        """Worst R-hat and lowest ESS at the time of the fit."""
        return dict(self.manifest.get("convergence", {}))

    @property
    def parameters(self) -> pd.DataFrame:
        return pd.DataFrame(self.manifest.get("parameters", []))

    def predict(self, stage):
        """Discharge (cfs) at `stage`.

        Evaluated exactly when the manifest records a closed form - a polynomial's
        coefficients, or an exponential's amplitude and rate - and interpolated from
        the stored curve otherwise. Either way, stages outside the curve's range come
        back NaN rather than being extrapolated.
        """
        sampler = self.sampler
        coefficients = sampler.get("coefficients")
        amplitude, rate = sampler.get("amplitude"), sampler.get("rate")
        closed_form = None
        if coefficients:
            closed_form = lambda h: np.polyval(np.asarray(coefficients, float), h)
        elif amplitude is not None and rate is not None:
            # the exponential is anchored at a stage offset, without which e^(B*800)
            # overflows; the offset is part of the closed form, not a detail
            offset = float(sampler.get("stage_offset", 0.0) or 0.0)
            closed_form = lambda h: (float(amplitude)
                                     * np.exp(float(rate) * (h - offset)))
        if closed_form is not None:
            scalar = np.ndim(stage) == 0
            stages = np.atleast_1d(np.asarray(stage, float))
            low, high = self.stage_range
            predicted = np.where((stages >= low) & (stages <= high),
                                 closed_form(stages), np.nan)
            return float(predicted[0]) if scalar else predicted

        curve = self.curve.sort_values("stage_ft")
        if curve.empty:
            raise RuntimeError(f"{self.path.name} stored no curve, so it cannot "
                               f"predict")
        scalar = np.ndim(stage) == 0
        stages = np.atleast_1d(np.asarray(stage, float))
        predicted = np.interp(stages, curve["stage_ft"], curve["discharge_cfs"],
                              left=np.nan, right=np.nan)
        return float(predicted[0]) if scalar else predicted

    def interval(self, stage, level=None):
        """The stored credible interval at `stage`, as ``(lower, upper)``.

        The band was written at the level the fit reported (95% unless the fit said
        otherwise), so `level` is accepted only to be refused: a different level
        needs the draws, which is what :meth:`posterior` is for.
        """
        if level is not None:
            raise ValueError(
                "a saved rating stores one credible band, so it cannot be "
                "re-evaluated at a different level; load posterior() and compute it "
                "from the draws instead")
        curve = self.curve.sort_values("stage_ft")
        stages = np.atleast_1d(np.asarray(stage, float))
        lower = np.interp(stages, curve["stage_ft"], curve["lower"],
                          left=np.nan, right=np.nan)
        upper = np.interp(stages, curve["stage_ft"], curve["upper"],
                          left=np.nan, right=np.nan)
        if np.ndim(stage) == 0:
            return float(lower[0]), float(upper[0])
        return lower, upper

    @property
    def posterior_path(self):
        """Where the draws are, or None if the manifest was written without them."""
        name = self.manifest.get("posterior_file")
        if not name:
            return None
        return self.path.parent / name

    @property
    def expects_posterior(self) -> bool:
        """Whether this model's export is *supposed* to include draws.

        False for the least-squares forms, which have no posterior at all - a
        missing ``.nc`` there is completeness, not loss.
        """
        return bool(self.manifest.get("model", {}).get("has_posterior", True))

    def verify(self) -> dict:
        """Check that the pair is intact and that the two files belong together.

        The one thing a copy between machines can get wrong silently: the manifest
        arrives and the posterior does not, or arrives truncated, and every number
        the manifest carries still reads correctly. So the checksum recorded at write
        time is compared here rather than trusted.

        Returns
        -------
        dict
            ``ok`` (bool), ``posterior_present``, ``checksum_recorded``,
            ``checksum_matches`` (None when nothing was recorded to compare against -
            a version-1 manifest, which predates the checksum) and ``problem``, a
            sentence naming what is wrong, empty when nothing is.
        """
        recorded = self.manifest.get("posterior_sha256")
        path = self.posterior_path
        report = {"ok": True, "posterior_present": bool(path and path.exists()),
                  "checksum_recorded": bool(recorded), "checksum_matches": None,
                  "problem": ""}
        if not self.expects_posterior:
            report["problem"] = ""
            return report
        if path is None:
            return {**report, "ok": False,
                    "problem": f"{self.path.name} records no posterior file, and this "
                               f"model has draws that cannot be reconstructed"}
        if not path.exists():
            return {**report, "ok": False,
                    "problem": f"the posterior {path.name} is not next to "
                               f"{self.path.name}; the pair has been split"}
        if recorded:
            report["checksum_matches"] = _sha256(path) == recorded
            if not report["checksum_matches"]:
                return {**report, "ok": False,
                        "problem": f"{path.name} does not match the SHA-256 recorded "
                                   f"in {self.path.name}; it is a different or a "
                                   f"truncated file"}
        return report

    def posterior(self, *, verify: bool = True):
        """The posterior draws, read on demand.

        Parameters
        ----------
        verify : bool, default True
            Check the recorded SHA-256 first, and refuse a posterior that does not
            match. Pass False to read a file you know has been rewritten.

        Returns
        -------
        arviz.InferenceData

        Raises
        ------
        FileNotFoundError
            If the manifest recorded no posterior, or the file is not beside it.
        ValueError
            If the file is there but is not the one this manifest was written with.
        """
        path = self.posterior_path
        if path is None:
            raise FileNotFoundError(
                f"{self.path.name} was written without a posterior, so the draws are "
                f"not recoverable from it"
                + ("" if self.expects_posterior else
                   f" - {self.name} has no posterior, so its manifest is the whole "
                   f"export"))
        if not path.exists():
            raise FileNotFoundError(
                f"the posterior {path.name} is not next to {self.path.name}; the two "
                f"files travel together and one has been moved or deleted. Copy "
                f"both, or refit.")
        if verify:
            checked = self.verify()
            if checked["checksum_matches"] is False:
                raise ValueError(checked["problem"])
        import arviz as az
        return az.from_netcdf(str(path))

    def plot(self, ax=None, *, live=None):
        """The stored curve, its band, and the measurements it was fitted to.

        Drawn from the manifest alone, so it works wherever :meth:`describe` does.
        Pass ``live=`` a fitted model to overlay it: the two lie on top of each other
        inside the stored range and the reloaded one simply stops at its ends, which
        is the whole difference between a saved rating and a live fit.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.
        live : RatingModel, optional
            A live fit to draw underneath, for comparison.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4.8))
        curve = self.curve
        if live is not None:
            live_curve = live.curve()
            ax.plot(live_curve["stage_ft"], live_curve["discharge_cfs"],
                    color="#4c72b0", lw=3, alpha=0.6, label="live fit")
        if "lower" in curve:
            ax.fill_between(curve["stage_ft"], curve["lower"], curve["upper"],
                            color="#dd8452", alpha=0.25, label="stored 95% band")
        ax.plot(curve["stage_ft"], curve["discharge_cfs"], color="#dd8452", lw=1.4,
                ls="--", label=f"{self.name}, from {self.path.name}")
        measurements = self.measurements
        if not measurements.empty:
            ax.plot(measurements["stage_ft"], measurements["discharge_cfs"], "o",
                    color="black", ms=4, label="measurements")
        ax.set(yscale="log", xlabel="stage (ft)", ylabel="discharge (cfs)",
               title=f"{self.site_id}: a saved rating interpolates its stored curve")
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize="small")
        return ax

    def describe(self) -> str:
        """A readable summary of the file: what was fitted, how, and how well.

        Everything here comes out of the manifest, so this prints on a machine with
        no PyMC, no ArviZ and no compiler - which is the point of the format.

        Returns
        -------
        str
            Several lines, ready to print.
        """
        scores = self.metrics
        sampler = self.sampler
        convergence = self.convergence
        low, high = self.stage_range
        curve = self.curve
        checked = self.verify()

        def number(value, spec="{:.4g}"):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return "–"
            try:
                return spec.format(value)
            except (TypeError, ValueError):
                return str(value)

        lines = [
            f"{self.path.name}",
            f"  site         {self.site_id}"
            f"   ({self.manifest.get('site', {}).get('source') or 'unknown source'},"
            f" n={self.manifest.get('site', {}).get('n_measurements')})",
            f"  model        {self.name}  -  {self.label}   [{self.family}]",
            f"  equation     {self.equation or '(no closed form)'}",
            f"  stage range  {number(low)} to {number(high)} "
            f"{self.manifest.get('units', {}).get('stage', 'ft')}"
            f"   ({len(curve)} curve points)",
            f"  stage axis   {self.manifest.get('units', {}).get('stage_label', '')}",
            f"  fitted by    {sampler.get('method', '?')}"
            f"{' / ' + str(sampler.get('nuts_sampler')) if sampler.get('nuts_sampler') else ''}"
            f"   seed {sampler.get('seed', '?')}",
            f"  in-sample    NSE {number(scores.get('nse'), '{:.3f}')}"
            f"   RMSE {number(scores.get('rmse'))} cfs"
            f"   PBIAS {number(scores.get('pbias_pct'), '{:.1f}')}%"
            f"   R²log {number(scores.get('r2_log'), '{:.3f}')}",
            f"  predictive   ELPD_LOO {number(scores.get('elpd_loo'), '{:.1f}')}"
            f" ± {number(scores.get('se_loo'), '{:.1f}')}"
            f"   p_loo {number(scores.get('p_loo'), '{:.1f}')}"
            f"   worst Pareto-k {number(scores.get('pareto_k_max'), '{:.2f}')}",
        ]
        if convergence.get("available"):
            lines.append(
                f"  convergence  worst R-hat "
                f"{number(convergence.get('worst_r_hat'), '{:.4f}')}"
                f"   lowest ESS "
                f"{number(convergence.get('lowest_ess_bulk'), '{:.0f}')}"
                f"   {'converged' if convergence.get('converged') else 'not measured' if convergence.get('converged') is None else 'NOT CONVERGED'}")
        else:
            lines.append("  convergence  not measured (no MCMC chains in this fit)")
        if not self.expects_posterior:
            state = "none - this model has no draws, so the manifest is the whole fit"
        elif checked["checksum_matches"] is True:
            state = f"{self.manifest.get('posterior_file')}   checksum verified"
        elif checked["checksum_matches"] is None:
            state = (f"{self.manifest.get('posterior_file') or 'none'}   "
                     f"no checksum recorded (format v"
                     f"{self.manifest.get('format_version')})")
        else:
            state = f"{self.manifest.get('posterior_file')}   CHECKSUM MISMATCH"
        lines.append(f"  posterior    {state}")
        if checked["problem"]:
            lines.append(f"  PROBLEM      {checked['problem']}")
        lines.append(f"  written      {self.manifest.get('written_utc', '?')}"
                     f"  by limnotech_rating_curves "
                     f"{self.manifest.get('package_version', '?')}"
                     f"  (format v{self.manifest.get('format_version')})")
        return "\n".join(lines)

    def __repr__(self):
        scores = self.metrics
        nse = scores.get("nse")
        return (f"SavedRating({self.site_id!r}, {self.name}, "
                f"n={self.manifest.get('site', {}).get('n_measurements')}, "
                f"nse={'nan' if nse is None else f'{nse:.3f}'})")


def load_rating(path) -> SavedRating:
    """Read a rating back from its manifest.

    Parameters
    ----------
    path : path-like
        The ``.rating.json`` manifest. A directory is rejected - pass one file, or
        use :func:`load_ratings` for a whole directory.

    Returns
    -------
    SavedRating

    Raises
    ------
    ValueError
        If the file is not a rating manifest, or was written by a newer format
        version than this package understands.
    """
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(
            f"{path} is a directory; pass one {MANIFEST_SUFFIX} manifest, or use "
            f"load_ratings({str(path)!r}) to read all of them")
    if not path.exists():
        sibling = path.with_suffix("")            # a .nc handed in by mistake
        hint = ""
        if path.suffix == POSTERIOR_SUFFIX:
            hint = (f" - you passed a posterior; its manifest would be "
                    f"{sibling.name}{MANIFEST_SUFFIX}")
        raise FileNotFoundError(f"no rating manifest at {path}{hint}")
    if path.suffix == POSTERIOR_SUFFIX:
        expected = path.with_suffix("").with_suffix("")
        raise ValueError(
            f"{path.name} is the posterior half of the pair, not the manifest. Load "
            f"{expected.name}{MANIFEST_SUFFIX} instead; the draws come back from "
            f"its .posterior().")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not readable JSON, so it is not a rating "
                         f"manifest ({exc})") from exc
    if manifest.get("format") != "limnotech-rating-curve":
        raise ValueError(f"{path.name} is not a rating manifest "
                         f"(no 'limnotech-rating-curve' format marker)")
    written_version = int(manifest.get("format_version", 0))
    if written_version > MANIFEST_VERSION:
        raise ValueError(
            f"{path.name} is format version {written_version}, and this package "
            f"understands up to {MANIFEST_VERSION}; upgrade "
            f"limnotech_rating_curves to read it")
    return SavedRating(manifest, path)


def load_ratings(directory) -> dict:
    """Read every rating manifest in a directory.

    Parameters
    ----------
    directory : path-like

    Returns
    -------
    dict
        Keyed by ``(site_id, model_key)``, in sorted file order. Files that do not
        parse are logged and skipped.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    loaded = {}
    for path in sorted(directory.glob(f"*{MANIFEST_SUFFIX}")):
        try:
            saved = load_rating(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s (%s: %s)", path.name, type(exc).__name__, exc)
            continue
        loaded[(saved.site_id, saved.name)] = saved
    if not loaded:
        orphans = orphan_posteriors(directory)
        if orphans:
            log.warning(
                "%s holds %d posterior file(s) and no manifest, so none of them can "
                "be loaded: a posterior on its own records neither the site nor the "
                "curve. Refit, or export from a run that writes both halves.",
                directory, len(orphans))
    return loaded


def orphan_posteriors(directory) -> list:
    """Posterior files in a directory with no manifest beside them.

    A ``.nc`` alone is not a saved rating: it holds draws over parameter names and
    nothing that says which site, which stage axis, or what the curve was. This finds
    them so a run can say so instead of a later load quietly returning nothing.

    Parameters
    ----------
    directory : path-like

    Returns
    -------
    list of pathlib.Path
        Sorted.
    """
    directory = Path(directory)
    manifests = {path.name[:-len(MANIFEST_SUFFIX)]
                 for path in directory.glob(f"*{MANIFEST_SUFFIX}")}
    return sorted(path for path in directory.glob(f"*{POSTERIOR_SUFFIX}")
                  if path.stem not in manifests)


def manifest_index(saved_ratings) -> pd.DataFrame:
    """One row per saved rating: identity, scores, stage range, integrity.

    Parameters
    ----------
    saved_ratings : sequence or dict of SavedRating

    Returns
    -------
    pandas.DataFrame
    """
    if isinstance(saved_ratings, dict):
        saved_ratings = list(saved_ratings.values())
    rows = []
    for saved in saved_ratings:
        scores = saved.metrics
        low, high = saved.stage_range
        checked = saved.verify()
        rows.append({
            "site": saved.site_id, "model": saved.name, "model_label": saved.label,
            "family": saved.family,
            "n": saved.manifest.get("site", {}).get("n_measurements"),
            "method": saved.sampler.get("method"),
            "stage_low_ft": low, "stage_high_ft": high,
            "equation": saved.equation,
            "nse": scores.get("nse"), "rmse": scores.get("rmse"),
            "r2_log": scores.get("r2_log"), "elpd_loo": scores.get("elpd_loo"),
            "se_loo": scores.get("se_loo"),
            "pareto_k_max": scores.get("pareto_k_max"),
            "worst_r_hat": saved.convergence.get("worst_r_hat"),
            "manifest_file": saved.path.name,
            "posterior_file": saved.manifest.get("posterior_file"),
            "intact": checked["ok"], "problem": checked["problem"]})
    return pd.DataFrame(rows)


def export_directory(source, destination, *, write_csv: bool = True,
                     copy: bool = True) -> dict:
    """Collect a directory of saved ratings into a portable set for another machine.

    What "portable" means here: the manifest/posterior pairs, an index table saying
    what is in the set, and - because the people who need a rating downstream open
    spreadsheets, not JSON - a plain CSV of every curve. Nothing in the destination
    needs this package to be read except the ``.nc`` files, and those need only
    ArviZ.

    Parameters
    ----------
    source : path-like
        A directory of ``.rating.json`` / ``.nc`` pairs, e.g. a batch run's
        ``output/fitted_curves``.
    destination : path-like
        Where the portable set goes. Created if needed. May be the same as `source`,
        in which case only the tables are written.
    write_csv : bool, default True
        Also write ``ratings_index.csv`` and ``rating_curves.csv``.
    copy : bool, default True
        Copy the pairs into `destination`. False writes only the tables.

    Returns
    -------
    dict
        ``ratings`` (how many pairs), ``files`` (paths written), ``orphans`` (posterior
        files with no manifest, which cannot travel usefully), ``problems`` (list of
        integrity complaints).
    """
    import shutil

    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    loaded = load_ratings(source)
    same_place = source.resolve() == destination.resolve()

    written = []
    for saved in loaded.values():
        if copy and not same_place:
            written.append(str(shutil.copy2(saved.path, destination / saved.path.name)))
            posterior = saved.posterior_path
            if posterior and posterior.exists():
                written.append(str(shutil.copy2(posterior,
                                                destination / posterior.name)))

    index = manifest_index(loaded)
    problems = ([] if index.empty
                else [f"{row.manifest_file}: {row.problem}"
                      for row in index.itertuples() if row.problem])
    if write_csv and not index.empty:
        index_path = destination / "ratings_index.csv"
        index.to_csv(index_path, index=False)
        written.append(str(index_path))
        curves = curve_table(loaded)
        if not curves.empty:
            curve_path = destination / "rating_curves.csv"
            curves.to_csv(curve_path, index=False)
            written.append(str(curve_path))

    orphans = [path.name for path in orphan_posteriors(source)]
    log.info("exported %d rating(s) to %s%s", len(loaded), destination,
             f"; {len(orphans)} orphan posterior(s) left behind" if orphans else "")
    return {"ratings": len(loaded), "files": written, "orphans": orphans,
            "problems": problems}
