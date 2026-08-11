import logging

import numpy as np
import pandas as pd

from . import bdrc, exponential, polynomial, ratingcurve
from ..evaluate import metrics as metrics_module
from .. import settings
from ..core import (FitResult, FoldCurve, Sample, fit_metrics,
                    padded_stage_grid)

log = logging.getLogger(__name__)

#: Line style per family, so a curve's *shape* says what kind of thing it is before
#: its colour says which one. Solid for the two Bayesian families (they are the
#: package's own recommendation), dashed and dotted for the least-squares spreadsheet
#: forms (they are reproductions of somebody else's curve).
FAMILY_DASH = {
    "ratingcurve": "solid",
    "bdrc": "solid",
    "polynomial": "dash",
    "exponential": "dot",
}


def _as_frame(sample) -> pd.DataFrame:
    """A tidy stage/discharge frame from a Sample or a frame-like."""
    if isinstance(sample, Sample):
        return sample.to_frame()
    return pd.DataFrame(sample)


def _padded_grid(sample) -> np.ndarray:
    """A padded stage grid spanning a sample's observed range."""
    return padded_stage_grid(_as_frame(sample)["stage_ft"].to_numpy(float))


class ModelEntry:
    """One rating model in the catalog. Subclasses supply the family specifics.

    Parameters
    ----------
    key : str
        Stable identifier, used in tables, file names and ``models=`` arguments.
    label : str
        Full display name.
    short_label : str
        Compact name, for narrow buttons and legends.
    color : str
        Hex color this model is drawn in, everywhere.
    min_points : int
        Fewest measurements the model needs before it is worth fitting.
    dash : str, optional
        Plotly line style this model is drawn with, everywhere: ``"solid"``,
        ``"dash"``, ``"dot"``, ``"dashdot"``, ``"longdash"``. Defaults to the
        family's own style (see ``FAMILY_DASH``) so a legend is readable by shape as
        well as by hue - which matters on a projector, in print, and to a
        colour-blind reader.

    Attributes
    ----------
    family : str
        Which backend fits it: ``"ratingcurve"``, ``"bdrc"``, ``"polynomial"`` or
        ``"exponential"``.
    has_posterior : bool
        Whether a successful fit produces posterior draws. False for the
        least-squares forms, which have none at all - so an export with no
        ``.nc`` beside it is complete for those and incomplete for every other.
    role : {'model', 'spreadsheet_form'}
        Whether this is a rating the package offers on its own account, or a
        least-squares form included to reproduce a field spreadsheet's trendline.
        See the module docstring.
    """

    family = ""
    has_posterior = True
    role = "model"

    def __init__(self, key: str, label: str, short_label: str, color: str,
                 min_points: int, dash: str = None):
        self.key = key
        self.label = label
        self.short_label = short_label
        self.color = color
        self.min_points = min_points
        self.dash = dash or FAMILY_DASH.get(self.family, "solid")

    def fit(self, sample, **kwargs) -> FitResult:
        """Fit this model to all of a sample.

        Implemented by each family; see :meth:`RatingCurveEntry.fit` and
        :meth:`BdrcEntry.fit` for the keywords each accepts.

        Parameters
        ----------
        sample : Sample or pandas.DataFrame
            The measurements.
        **kwargs
            Family-specific fitting options.

        Returns
        -------
        FitResult
        """
        raise NotImplementedError

    def fit_fold(self, train, test, grid, split, **kwargs):
        """Fit this model to one cross-validation fold.

        Implemented by each family; see :meth:`RatingCurveEntry.fit_fold` and
        :meth:`BdrcEntry.fit_fold`.

        Parameters
        ----------
        train, test : pandas.DataFrame
            The fold's training and held-out measurements.
        grid : array-like
            Shared stage grid the fold curve is evaluated on.
        split : int
            Fold index.
        **kwargs
            Family-specific fitting options.

        Returns
        -------
        FoldCurve or None
        """
        raise NotImplementedError

    def save_posterior(self, fit: FitResult, directory, sample_id):
        """Write a fitted posterior to an ArviZ NetCDF file.

        Parameters
        ----------
        fit : FitResult
            The fit to export.
        directory : path-like
            Destination directory.
        sample_id : str
            Identifies the site in the file name.

        Returns
        -------
        str or None
            The path written, or None when there is nothing to write. See
            :func:`save_posterior` for the two families' different round trips.
        """
        return save_posterior(fit, directory, sample_id)

    def bayes_metrics(self, fit: FitResult) -> dict:
        """PSIS-LOO and WAIC scores for a completed fit.

        Implemented by each family, because the two produce their pointwise
        log-likelihood by different routes and in different observation spaces.

        Parameters
        ----------
        fit : FitResult
            A completed fit.

        Returns
        -------
        dict
            The fields in ``metrics.BAYES_FIELDS``.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} {self.key}>"


class RatingCurveEntry(ModelEntry):
    """A ratingcurve (PyMC) model: the power law at a fixed segment count, or the
    natural spline.

    Parameters
    ----------
    key, label, short_label, color, min_points
        See :class:`ModelEntry`.
    algorithm : {'power_law', 'spline'}
        Which ratingcurve family.
    segments : int or None
        Power-law segment count; ``None`` for the spline.
    """

    family = "ratingcurve"

    def __init__(self, key, label, short_label, color, algorithm, segments,
                 min_points, dash: str = None):
        super().__init__(key, label, short_label, color, min_points, dash=dash)
        self.algorithm = algorithm
        self.segments = segments

    def fit(self, sample, *, method: str = "nuts", seed: int = settings.SEED,
            enforce_min_points: bool = True, zero_flow=None, cores=None,
            advi_iters: int = settings.ADVI_ITERS, nuts_sampler=None,
            config_override=None, with_log_likelihood: bool = False, grid=None
            ) -> FitResult:
        """Fit on all of a sample.

        Parameters
        ----------
        sample : Sample or pandas.DataFrame
            The measurements.
        method : {'nuts', 'advi'}, default 'nuts'
            Full MCMC, or the fast variational fit.
        seed : int
            RNG seed.
        enforce_min_points : bool, default True
            Skip the fit when the sample is smaller than `min_points`.
        zero_flow : optional
            Stage of zero flow for the breakpoint prior; see
            :func:`limnotech_rating_curves.models.ratingcurve.breakpoint_prior`.
        cores : int, optional
            Parallel NUTS chains. Pass 1 inside a worker process.
        advi_iters : int
            ADVI optimization steps.
        nuts_sampler : str, optional
            Which NUTS implementation.
        config_override : dict, optional
            Merged over the adapted model config.
        grid : array-like, optional
            Stage grid the curve is tabulated on. Defaults to the padded grid
            every family is given.
        with_log_likelihood
            Accepted for interface symmetry and unused: a ratingcurve fit keeps
            its live posterior, so the log-likelihood is computed on demand in
            :meth:`bayes_metrics`.

        Returns
        -------
        FitResult
        """
        return ratingcurve.fit(
            sample, key=self.key, label=self.label, algorithm=self.algorithm,
            segments=self.segments, min_points=self.min_points, method=method,
            seed=seed, cores=cores, advi_iters=advi_iters,
            nuts_sampler=nuts_sampler, zero_flow=zero_flow,
            enforce_min_points=enforce_min_points, config_override=config_override,
            grid=_padded_grid(sample) if grid is None else grid)

    def fit_fold(self, train, test, grid, split, *, seed: int = settings.SEED,
                 zero_flow=None, nuts_sampler=None, method: str = "advi"
                 ) -> "FoldCurve | None":
        """Fit one cross-validation fold and score it on the held-out points.

        Parameters
        ----------
        train, test : pandas.DataFrame
            The fold's training and held-out measurements.
        grid : array-like
            Shared stage grid the fold curve is evaluated on, so folds and models
            can be compared elementwise.
        split : int
            Fold index.
        seed : int
            RNG seed for this fold.
        zero_flow : optional
            Stage of zero flow for the breakpoint prior.
        nuts_sampler : str, optional
            Ignored here: a ratingcurve fold is fitted by ADVI, which is gradient
            ascent on the ELBO and not a NUTS chain at all. bdrc folds *are* NUTS,
            so it matters there and the argument exists for symmetry.
        method : {'advi', 'nuts'}, default 'advi'
            Fold fitting method. ADVI by default because a fold sweep is about
            visual robustness across many refits, not about ELPD.

        Returns
        -------
        FoldCurve or None
            None if this fold could not be fitted.
        """
        fit = ratingcurve.fit(
            train, key=self.key, label=self.label, algorithm=self.algorithm,
            segments=self.segments, min_points=self.min_points, method=method,
            seed=seed, zero_flow=zero_flow, enforce_min_points=False)
        if not fit.ok or fit.rating is None:
            return None
        table = fit.rating.table(stage=np.asarray(grid, float))
        predicted = fit.rating.predict(test["stage_ft"].to_numpy(float))
        return FoldCurve(
            model_key=self.key, split=split,
            stage_ft=table["stage_ft"].tolist(),
            discharge_median_cfs=table["discharge_median_cfs"].tolist(),
            lower_cfs=table["lower"].tolist(), upper_cfs=table["upper"].tolist(),
            train_stage=train["stage_ft"].tolist(),
            train_discharge=train["discharge_cfs"].tolist(),
            test_stage=test["stage_ft"].tolist(),
            test_discharge=test["discharge_cfs"].tolist(),
            test_metrics=fit_metrics(test["discharge_cfs"].to_numpy(float), predicted))

    def bayes_metrics(self, fit: FitResult) -> dict:
        """PSIS-LOO and WAIC for a ratingcurve fit.

        The log-likelihood is computed from the live posterior on demand, then
        shifted out of ratingcurve's standardized log-discharge space so the ELPD
        is comparable with bdrc's (see
        :mod:`limnotech_rating_curves.evaluate.metrics`).

        Parameters
        ----------
        fit : FitResult

        Returns
        -------
        dict
            The fields in ``metrics.BAYES_FIELDS``, or NaNs with a note when the
            posterior is unavailable - which happens whenever a fit crossed a
            process boundary, since a PyMC model does not survive that.
        """
        if not fit.ok:
            return metrics_module.unavailable("fit not ok")
        refused = metrics_module.refused_for_advi(fit)
        if refused is not None:
            return refused
        rating = fit.rating
        if rating is None or rating.idata is None:
            return metrics_module.unavailable("no posterior in memory")
        try:
            metrics_module.ensure_log_likelihood(rating)
            log_std = float(np.log(rating.model.q_transform.std_))
        except Exception as exc:  # noqa: BLE001
            return metrics_module.unavailable(f"{type(exc).__name__}: {str(exc)[:80]}")
        return metrics_module.elpd_from_idata(rating.idata, log_offset=-log_std)


class BdrcEntry(ModelEntry):
    """A bdrc Bayesian rating curve - one of ``gplm0``, ``gplm``, ``plm0``, ``plm``.

    Fitted by the native PyMC port in :mod:`limnotech_rating_curves.models.bdrc`; no R is
    involved. Its pointwise log-likelihood comes out of the all-data fit already
    in raw log-discharge space, so its ELPD needs no offset, and its credible band
    comes straight from bdrc's own predictive bounds.

    Parameters
    ----------
    variant : {'gplm0', 'gplm', 'plm0', 'plm'}
        Which bdrc model.
    label, short_label, color
        See :class:`ModelEntry`.
    min_points : int, default 3
        Fewest measurements needed.
    """

    family = "bdrc"

    def __init__(self, variant: str, label: str, short_label: str, color: str,
                 min_points: int = 3, dash: str = None):
        super().__init__(f"bdrc_{variant}", label, short_label, color, min_points,
                         dash=dash)
        self.variant = variant

    def fit(self, sample, *, method: str = "nuts", seed: int = settings.SEED,
            enforce_min_points: bool = True, zero_flow=None, cores=None,
            advi_iters: int = settings.ADVI_ITERS, nuts_sampler=None,
            config_override=None, with_log_likelihood: bool = False, grid=None
            ) -> FitResult:
        """Fit on all of a sample.

        Parameters
        ----------
        sample : Sample or pandas.DataFrame
            The measurements.
        method : {'nuts', 'advi'}, default 'nuts'
            Full MCMC, or the fast variational fit.
        seed : int
            RNG seed.
        enforce_min_points : bool, default True
            Kept for interface symmetry; bdrc's minimum is three measurements and
            is checked by the sample being non-degenerate rather than by a count.
        zero_flow : optional
            Stage of zero flow. Note the difference from the power law: bdrc takes
            this as a **known, fixed** value for its ``c`` parameter, not as a
            prior mean. ``None`` (the default) lets bdrc infer ``c``, which is
            what its priors are designed for. Pass a number or ``"johnson"`` only
            when you genuinely want ``c`` held fixed.
        cores : int, optional
            Parallel NUTS chains. Left at bdrc's own default (1, sequential
            chains) when None, because PyMC's process pool deadlocks under some
            Windows launchers.
        advi_iters : int
            ADVI optimization steps.
        nuts_sampler : str, optional
            Which NUTS implementation.
        config_override : dict, optional
            Extra keywords forwarded to :func:`bdrc.fit_predict`.
        with_log_likelihood : bool, default False
            Also return the pointwise log-likelihood, which is what the ELPD
            comparison needs. A batch run turns this on.
        grid : array-like, optional
            Stage grid to evaluate the curve on. Defaults to a padded grid over
            the observed range.

        Returns
        -------
        FitResult
        """
        frame = _as_frame(sample)
        stage = frame["stage_ft"].to_numpy(float)
        discharge = frame["discharge_cfs"].to_numpy(float)
        zero_flow_ft = _resolve_fixed_zero_flow(zero_flow, stage, discharge)

        # Resolved once and both recorded and used. bdrc carries its own default in
        # bdrc.defaults, so leaving this to the backend would sample with that while
        # the config claimed the package-wide setting - and the export manifest copies
        # the config, so a disagreement here would be recorded as provenance.
        sampler = settings.NUTS_SAMPLER if nuts_sampler is None else nuts_sampler

        result = FitResult(
            key=self.key, label=self.label, family="bdrc", n=len(frame),
            status="skipped", reason="",
            config={"variant": self.variant, "zero_flow_ft": zero_flow_ft,
                    "method": method, "nuts_sampler": sampler})
        if not np.isfinite(stage).any() or not np.isfinite(discharge).any():
            result.reason = "no finite measurements"
            return result

        grid = _padded_grid(frame) if grid is None else np.asarray(grid, float)
        extra = dict(config_override or {})
        if cores is not None:
            extra["cores"] = cores
        extra.setdefault("nuts_sampler", sampler)
        try:
            output = bdrc.fit_predict(
                stage, discharge, grid, model=self.variant, method=method,
                c_param_ft=zero_flow_ft, with_loglik=with_log_likelihood,
                advi_n=advi_iters, seed=seed, **extra)
        except Exception as exc:  # noqa: BLE001
            result.status = "failed"
            result.reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            log.debug("%s failed on n=%d: %s", self.key, len(frame), result.reason)
            return result

        if with_log_likelihood:
            curve = output["curve"]
            result.log_likelihood = output["loglik"]
            result.native = output["native"]
            fitted = output.get("fit")
            result.idata = getattr(fitted, "idata", None)
            # c is drawn as log(h_min - c) in metres and never appears in the idata on
            # its own scale, so the posterior mean is recorded here in feet - the only
            # place a caller can reach it afterwards.
            drawn_c = getattr(fitted, "c_posterior", None)
            if drawn_c is not None:
                result.config["zero_flow_fitted_ft"] = float(
                    np.mean(drawn_c) / bdrc.FT_TO_M)
        else:
            curve = output
        if curve.empty:
            result.reason = "bdrc returned an empty curve"
            return result

        curve = curve.rename(columns={"q_median_cfs": "discharge_median_cfs",
                                      "q_lower_cfs": "lower", "q_upper_cfs": "upper"})
        # bdrc reports the posterior-predictive median; there is no separate mean
        # curve, so the median is what the package's "discharge_cfs" column holds.
        curve["discharge_cfs"] = curve["discharge_median_cfs"]
        predicted = np.interp(stage, curve["stage_ft"], curve["discharge_cfs"],
                              left=np.nan, right=np.nan)
        result.status = "ok"
        result.curve = curve[["stage_ft", "discharge_cfs", "discharge_median_cfs",
                              "lower", "upper"]]
        result.predicted = predicted
        result.metrics = fit_metrics(discharge, predicted)
        return result

    def fit_fold(self, train, test, grid, split, *, seed: int = settings.SEED,
                 zero_flow=None, nuts_sampler=None, method: str = "nuts"
                 ) -> "FoldCurve | None":
        """Fit one cross-validation fold and score it on the held-out points.

        Parameters
        ----------
        train, test : pandas.DataFrame
            The fold's training and held-out measurements.
        grid : array-like
            Shared stage grid.
        split : int
            Fold index.
        seed : int
            RNG seed for this fold. Forwarded, so a cross-validation is
            reproducible.
        zero_flow : optional
            Known stage of zero flow (see :meth:`fit`).
        nuts_sampler : str, optional
            Which NUTS implementation. Unlike a ratingcurve fold, a bdrc fold *is*
            a NUTS chain, so this is where most of a cross-validation's cost lives
            and the choice matters.
        method : {'nuts', 'advi'}, default 'nuts'
            Fold fitting method.

        Returns
        -------
        FoldCurve or None
        """
        train_frame = _as_frame(train)
        zero_flow_ft = _resolve_fixed_zero_flow(
            zero_flow, train_frame["stage_ft"].to_numpy(float),
            train_frame["discharge_cfs"].to_numpy(float))
        extra = {"nuts_sampler": settings.NUTS_SAMPLER if nuts_sampler is None
                                 else nuts_sampler}
        try:
            curve = bdrc.fit_predict(
                train_frame["stage_ft"], train_frame["discharge_cfs"], grid,
                model=self.variant, method=method, c_param_ft=zero_flow_ft,
                seed=seed, **extra)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s fold %d failed: %s", self.key, split, exc)
            return None
        curve = curve.sort_values("stage_ft")
        predicted = np.interp(test["stage_ft"].to_numpy(float), curve["stage_ft"],
                              curve["q_median_cfs"], left=np.nan, right=np.nan)
        return FoldCurve(
            model_key=self.key, split=split,
            stage_ft=curve["stage_ft"].tolist(),
            discharge_median_cfs=curve["q_median_cfs"].tolist(),
            lower_cfs=curve["q_lower_cfs"].tolist(),
            upper_cfs=curve["q_upper_cfs"].tolist(),
            train_stage=train_frame["stage_ft"].tolist(),
            train_discharge=train_frame["discharge_cfs"].tolist(),
            test_stage=test["stage_ft"].tolist(),
            test_discharge=test["discharge_cfs"].tolist(),
            test_metrics=fit_metrics(test["discharge_cfs"].to_numpy(float), predicted))

    def bayes_metrics(self, fit: FitResult) -> dict:
        """PSIS-LOO and WAIC from bdrc's own pointwise log-likelihood.

        Parameters
        ----------
        fit : FitResult
            A fit produced with ``with_log_likelihood=True``.

        Returns
        -------
        dict
            The fields in ``metrics.BAYES_FIELDS``, or NaNs with a note when the
            fit carries no log-likelihood.
        """
        if not fit.ok:
            return metrics_module.unavailable("fit not ok")
        refused = metrics_module.refused_for_advi(fit)
        if refused is not None:
            return refused
        if fit.log_likelihood is None:
            return metrics_module.unavailable(
                "no pointwise log-likelihood - refit with with_log_likelihood=True")
        return metrics_module.elpd_from_log_likelihood(fit.log_likelihood)


class PolynomialEntry(ModelEntry):
    """A least-squares polynomial rating - the Excel chart trendline, in code.

    The odd one out in this catalog, and deliberately so: it is not Bayesian, so it
    has no posterior, no ELPD and no R-hat. It is here to reproduce a field
    spreadsheet's curve exactly and to put it on the same axes as the models that do
    have those things. Rank it by NSE or R-squared, not by predictive score.

    Parameters
    ----------
    key, label, short_label, color
        See :class:`ModelEntry`.
    degree : int
        Polynomial degree - 2 for the quadratic.
    min_points : int, default 4
        A degree-2 fit through 3 points is exact and says nothing about fit quality,
        so four is the fewest worth fitting.
    """

    family = "polynomial"
    has_posterior = False
    role = "spreadsheet_form"

    def __init__(self, key, label, short_label, color, degree: int,
                 min_points: int = 4, dash: str = None):
        super().__init__(key, label, short_label, color, min_points, dash=dash)
        self.degree = degree

    def fit(self, sample, *, stage_range=None, level: float = 0.95,
            enforce_min_points: bool = True, **ignored) -> FitResult:
        """Fit by ordinary least squares.

        Parameters
        ----------
        sample : Sample or pandas.DataFrame
            The measurements.
        stage_range : tuple of float, optional
            Override the effective stage range the curve may be used over.
        level : float, default 0.95
            Width of the reported prediction interval.
        enforce_min_points : bool, default True
            Skip rather than fit when the sample is too small.
        **ignored
            Sampler arguments (``method``, ``seed``, ``zero_flow``, ``cores``,
            ``nuts_sampler``, ...) are accepted for interface symmetry and ignored -
            a least-squares fit has none of them.

        Returns
        -------
        FitResult
        """
        return polynomial.fit(sample, key=self.key, label=self.label,
                              degree=self.degree, min_points=self.min_points,
                              enforce_min_points=enforce_min_points,
                              stage_range=stage_range, level=level)

    def fit_fold(self, train, test, grid, split, **ignored) -> "FoldCurve | None":
        """Fit one cross-validation fold and score it on the held-out points."""
        train_frame = _as_frame(train)
        fit = self.fit(train_frame, enforce_min_points=False)
        if not fit.ok:
            return None
        fitted = fit.rating
        table = fitted.table(stage=np.asarray(grid, float))
        predicted = fitted.predict(test["stage_ft"].to_numpy(float))
        return FoldCurve(
            model_key=self.key, split=split,
            stage_ft=table["stage_ft"].tolist(),
            discharge_median_cfs=table["discharge_cfs"].tolist(),
            lower_cfs=table["lower"].tolist(), upper_cfs=table["upper"].tolist(),
            train_stage=train_frame["stage_ft"].tolist(),
            train_discharge=train_frame["discharge_cfs"].tolist(),
            test_stage=test["stage_ft"].tolist(),
            test_discharge=test["discharge_cfs"].tolist(),
            test_metrics=fit_metrics(test["discharge_cfs"].to_numpy(float), predicted))

    def save_posterior(self, fit: FitResult, directory, sample_id):
        """Nothing to write: a least-squares fit has no posterior draws."""
        return None

    def bayes_metrics(self, fit: FitResult) -> dict:
        """NaNs with a reason - there is no posterior to score."""
        return metrics_module.unavailable(
            "a least-squares polynomial has no posterior, so no ELPD; compare it on "
            "NSE or R-squared instead")


class ExponentialEntry(ModelEntry):
    """A log-linear exponential rating - Excel's ``trendlineType="exp"``, in code.

    :math:`Q = A e^{Bh}`, fitted as ordinary least squares of :math:`\\log Q` on
    stage, which is what Excel computes and therefore what reproducing a field
    workbook's curve requires. Like the polynomial it is not Bayesian: no priors, no
    posterior, no ELPD, no R-hat. Rank it by NSE, and read its R-squared on discharge
    rather than the log-space one Excel prints.

    Parameters
    ----------
    key, label, short_label, color, dash
        See :class:`ModelEntry`.
    min_points : int, default 3
        Two points determine the log-linear fit exactly, so three is the fewest that
        says anything about fit quality.
    """

    family = "exponential"
    has_posterior = False
    role = "spreadsheet_form"

    def __init__(self, key, label, short_label, color, min_points: int = 3,
                 dash: str = None):
        super().__init__(key, label, short_label, color, min_points, dash=dash)

    def fit(self, sample, *, level: float = 0.95,
            enforce_min_points: bool = True, **ignored) -> FitResult:
        """Fit by ordinary least squares on log discharge.

        Parameters
        ----------
        sample : Sample or pandas.DataFrame
            The measurements.
        level : float, default 0.95
            Width of the reported prediction interval, which is computed on log
            discharge and exponentiated, so it is asymmetric in cfs.
        enforce_min_points : bool, default True
            Skip rather than fit when the sample is too small.
        **ignored
            Sampler arguments are accepted for interface symmetry and ignored.

        Returns
        -------
        FitResult
        """
        return exponential.fit(sample, key=self.key, label=self.label,
                               min_points=self.min_points,
                               enforce_min_points=enforce_min_points, level=level)

    def fit_fold(self, train, test, grid, split, **ignored) -> "FoldCurve | None":
        """Fit one cross-validation fold and score it on the held-out points."""
        train_frame = _as_frame(train)
        fit = self.fit(train_frame, enforce_min_points=False)
        if not fit.ok:
            return None
        fitted = fit.rating
        table = fitted.table(stage=np.asarray(grid, float))
        predicted = fitted.predict(test["stage_ft"].to_numpy(float))
        return FoldCurve(
            model_key=self.key, split=split,
            stage_ft=table["stage_ft"].tolist(),
            discharge_median_cfs=table["discharge_cfs"].tolist(),
            lower_cfs=table["lower"].tolist(), upper_cfs=table["upper"].tolist(),
            train_stage=train_frame["stage_ft"].tolist(),
            train_discharge=train_frame["discharge_cfs"].tolist(),
            test_stage=test["stage_ft"].tolist(),
            test_discharge=test["discharge_cfs"].tolist(),
            test_metrics=fit_metrics(test["discharge_cfs"].to_numpy(float), predicted))

    def save_posterior(self, fit: FitResult, directory, sample_id):
        """Nothing to write: a log-linear least-squares fit has no posterior draws."""
        return None

    def bayes_metrics(self, fit: FitResult) -> dict:
        """NaNs with a reason - there is no posterior to score."""
        return metrics_module.unavailable(
            "an exponential least-squares fit has no posterior, so no ELPD; compare "
            "it on NSE or R-squared instead")


def _resolve_fixed_zero_flow(zero_flow, stage, discharge):
    """Turn a ``zero_flow=`` argument into a fixed stage of zero flow, or None.

    Parameters
    ----------
    zero_flow : None or float or str or ZeroFlowEstimate
        ``None`` and ``"infer"`` mean "let the model estimate it" and return None.
        ``"johnson"`` / ``"auto"`` estimate it from the data. A number or estimate
        object is used as given.
    stage, discharge : array-like
        Used only when the value has to be estimated.

    Returns
    -------
    float or None
    """
    from ..data.zero_flow import ZeroFlowEstimate, estimate_zero_flow
    if zero_flow is None:
        return None
    if isinstance(zero_flow, str):
        keyword = zero_flow.strip().lower()
        if keyword in ("infer", "none", "estimate"):
            return None
        if keyword in ("johnson", "auto"):
            return float(estimate_zero_flow(stage, discharge).stage_ft)
        raise ValueError(f"unknown zero_flow {zero_flow!r}; pass a number, "
                         f"'johnson', or 'infer'")
    if isinstance(zero_flow, ZeroFlowEstimate):
        return float(zero_flow.stage_ft)
    return float(zero_flow)


def save_posterior(fit: FitResult, directory, sample_id) -> "str | None":
    """Write one fitted model's posterior to an ArviZ NetCDF file.

    Both families end up as NetCDF, by two paths, and the difference matters when
    you come to reload:

    * A **ratingcurve** fit goes through the package's own ``save()``, which is
      the posterior plus the model and sampler configuration in the file's
      attributes. That is what lets ``PowerLawRating.load(path)`` rebuild a
      working model - the round trip is complete.
    * A **bdrc** fit writes its InferenceData directly. bdrc is not a ratingcurve
      ``ModelBuilder``, so there is no stored model config and no ``load()`` to
      rebuild from: you get the hyperparameter posterior and the pointwise
      log-likelihood back, which is everything the metrics need, but not an object
      that can predict at a new stage. Rebuilding a bdrc curve means re-running
      its closed-form kriging step.

    Parameters
    ----------
    fit : FitResult
        The fit to export.
    directory : path-like
        Destination directory, created if needed.
    sample_id : str
        Identifies the site in the file name.

    Returns
    -------
    str or None
        The path written, or None when the fit did not succeed, its posterior is
        absent (which happens whenever the fit crossed a process boundary), or the
        export failed. An export failure is logged and returns None rather than
        raising: writing an archival copy of a posterior is a side effect of a run,
        and it must not be able to destroy the run's actual results.
    """
    from pathlib import Path
    if not fit.ok:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = str(sample_id).replace(":", "__").replace("/", "-")
    # the canonical key, so a file written by an older run's spelling is still found
    # by catalog.get() and by exports.load_ratings()
    path = directory / f"{safe_id}__{LEGACY_KEYS.get(str(fit.key), str(fit.key))}.nc"

    if fit.family in ("polynomial", "exponential"):
        # a least-squares fit has no draws to write; its manifest is the whole export
        return None
    try:
        if fit.rating is not None:
            return fit.rating.save(path)
        if fit.idata is not None:
            with ratingcurve.coerced_netcdf_attrs(fit.idata):
                fit.idata.to_netcdf(str(path))
            return str(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not export %s / %s to %s (%s: %s)", sample_id, fit.key,
                    path.name, type(exc).__name__, str(exc)[:160])
    return None


#: Every model, in display order: the ratingcurve power laws and spline, then the four
#: bdrc variants, then the three least-squares spreadsheet forms. This list is the
#: single source of truth for keys, labels, colors, line styles and ordering.
#:
#: The colors are chosen by family rather than picked one at a time - blues for the
#: ratingcurve power laws, purple for the spline that is *not* a power law, greens and
#: teals for bdrc, warm ochres for the least-squares forms. With ``FAMILY_DASH`` on top
#: of that, which kind of thing a curve is readable from its shape and which one it is
#: from its hue.
MODELS = [
    RatingCurveEntry("power_law", "power law (1 segment)", "PL 1-seg",
                     "#1f4e9c", "power_law", 1, 3),
    RatingCurveEntry("power_law_2seg", "segmented power law (2 segments)", "PL 2-seg",
                     "#3d7dd8", "power_law", 2, 6),
    RatingCurveEntry("power_law_3seg", "segmented power law (3 segments)", "PL 3-seg",
                     "#7fb1ea", "power_law", 3, 9),
    RatingCurveEntry("spline", "natural spline", "spline",
                     "#9467bd", "spline", None, 5),
    BdrcEntry("gplm0", "bdrc generalized power law (gplm0)", "bdrc gplm0", "#1b7837"),
    BdrcEntry("gplm", "bdrc generalized power law, var(h) (gplm)", "bdrc gplm",
              "#4daf4a"),
    BdrcEntry("plm0", "bdrc power law (plm0)", "bdrc plm0", "#17a89a"),
    BdrcEntry("plm", "bdrc power law, var(h) (plm)", "bdrc plm", "#66c2a5"),
    # The three spreadsheet forms. Their labels say "refit" on purpose: each is this
    # package fitting that form to the measurements, NOT the workbook's own typed
    # coefficients. Confusing the two is the single easiest way to misread the
    # comparison, so the label carries the distinction.
    PolynomialEntry("linear", "refit - least-squares linear", "linear",
                    "#e6a817", 1, min_points=3),
    PolynomialEntry("quadratic", "refit - least-squares quadratic", "quadratic",
                    "#d95f02", 2),
    ExponentialEntry("exponential", "refit - log-linear exponential", "exponential",
                     "#a6611a", min_points=3),
]

_BY_KEY = {entry.key: entry for entry in MODELS}

RATINGCURVE_KEYS = [entry.key for entry in MODELS if entry.family == "ratingcurve"]
BDRC_KEYS = [entry.key for entry in MODELS if entry.family == "bdrc"]
POLYNOMIAL_KEYS = [entry.key for entry in MODELS if entry.family == "polynomial"]
EXPONENTIAL_KEYS = [entry.key for entry in MODELS if entry.family == "exponential"]

#: The three forms the MAGL field workbooks use, refitted by this package. Fitting
#: all three at a sensor costs milliseconds, so the comparison always shows the two
#: forms the workbook did *not* choose alongside the one it did.
SPREADSHEET_FORM_KEYS = [entry.key for entry in MODELS
                         if entry.role == "spreadsheet_form"]

#: The models this package offers on its own account - everything that is not a
#: reproduction of a spreadsheet trendline.
MODEL_KEYS = [entry.key for entry in MODELS if entry.role == "model"]

#: Which workbook form each spreadsheet-form key reproduces, so a report can say
#: whether a refit is of the workbook's own form or of one it did not use.
FORM_KEYS = {"linear": "linear", "quadratic": "quadratic",
             "exponential": "exponential"}

DEFAULT_KEYS = ("power_law", "power_law_2seg", "spline", "bdrc_gplm0", 
                "linear", "quadratic", "exponential")

#: Group names :func:`select` understands, mapped to their member keys.
GROUPS = {
    "all": [entry.key for entry in MODELS],
    "default": list(DEFAULT_KEYS),
    "bdrc": BDRC_KEYS,
    "ratingcurve": RATINGCURVE_KEYS,
    "power_law_family": [k for k in RATINGCURVE_KEYS if k.startswith("power_law")],
    "polynomial": POLYNOMIAL_KEYS,
    "exponential_family": EXPONENTIAL_KEYS,
    # "the non-Bayesian spreadsheet-style curves", which is what a comparison against
    # a field workbook needs: all three forms, whichever one the workbook used
    "spreadsheet_forms": SPREADSHEET_FORM_KEYS,
    "models": MODEL_KEYS,
}

#: Older key spellings, accepted so existing scripts and stored results keep working.
LEGACY_KEYS = {
    "PowerLawRating_1seg": "power_law", "power_law_1seg": "power_law",
    "PowerLawRating_2seg": "power_law_2seg",
    "PowerLawRating_3seg": "power_law_3seg",
    "SplineRating": "spline",
}


def all_keys() -> list:
    """Every model key, in catalog order.

    Returns
    -------
    list of str
    """
    return [entry.key for entry in MODELS]


def get(key) -> ModelEntry:
    """Look up one model entry by key.

    Parameters
    ----------
    key : str
        A model key (older spellings in ``LEGACY_KEYS`` are accepted).

    Returns
    -------
    ModelEntry

    Raises
    ------
    KeyError
        If no such model exists, with the valid keys in the message.
    """
    resolved = LEGACY_KEYS.get(str(key), str(key))
    if resolved not in _BY_KEY:
        raise KeyError(f"unknown model {key!r}; choose from {', '.join(all_keys())} "
                       f"or a group ({', '.join(sorted(GROUPS))})")
    return _BY_KEY[resolved]


def color(key) -> str:
    """The color this model is drawn in, everywhere.

    Parameters
    ----------
    key : str
        A model key.

    Returns
    -------
    str
        Hex color, or a neutral grey for an unknown key - these lookups are used
        while rendering, where a missing model should degrade rather than raise.
    """
    entry = _BY_KEY.get(LEGACY_KEYS.get(str(key), str(key)))
    return entry.color if entry else "#555555"


def dash(key) -> str:
    """The Plotly line style this model is drawn with, everywhere.

    Parameters
    ----------
    key : str
        A model key.

    Returns
    -------
    str
        ``"solid"``, ``"dash"``, ``"dot"``, ... or ``"solid"`` for an unknown key.
    """
    entry = _BY_KEY.get(LEGACY_KEYS.get(str(key), str(key)))
    return entry.dash if entry else "solid"


def role(key) -> str:
    """Whether this is one of the package's models or a spreadsheet form refitted.

    Parameters
    ----------
    key : str
        A model key.

    Returns
    -------
    str
        ``"model"``, ``"spreadsheet_form"``, or ``"model"`` for an unknown key.
    """
    entry = _BY_KEY.get(LEGACY_KEYS.get(str(key), str(key)))
    return entry.role if entry else "model"


def label(key) -> str:
    """The model's full display name.

    Parameters
    ----------
    key : str
        A model key.

    Returns
    -------
    str
        The display name, or the key itself if it is unknown.
    """
    entry = _BY_KEY.get(LEGACY_KEYS.get(str(key), str(key)))
    return entry.label if entry else str(key)


def short_label(key) -> str:
    """The model's compact display name, for narrow buttons and legends.

    Parameters
    ----------
    key : str
        A model key.

    Returns
    -------
    str
        The short name, or the key itself if it is unknown.
    """
    entry = _BY_KEY.get(LEGACY_KEYS.get(str(key), str(key)))
    return entry.short_label if entry else str(key)


def select(spec=None) -> list:
    """Resolve a ``models=`` specification to catalog entries, in catalog order.

    Parameters
    ----------
    spec : None or str or sequence, optional
        ``None`` gives the curated default set. A string or sequence of strings is
        read as model keys and group names; a single string may hold several
        comma-separated tokens, and ``bdrc:all``-style colons are accepted as
        equivalent to the bare group name. Duplicates collapse.

    Returns
    -------
    list of ModelEntry

    Raises
    ------
    KeyError
        On an unrecognized token, listing the valid choices.

    Examples
    --------
    >>> [entry.key for entry in select("bdrc")]
    ['bdrc_gplm0', 'bdrc_gplm', 'bdrc_plm0', 'bdrc_plm']
    >>> [entry.key for entry in select(["power_law", "spline"])]
    ['power_law', 'spline']
    """
    if not spec:
        return [get(key) for key in DEFAULT_KEYS]
    if isinstance(spec, (str, ModelEntry)):
        spec = [spec]

    wanted = []
    for item in spec:
        if isinstance(item, ModelEntry):
            wanted.append(item.key)
            continue
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            # "bdrc:all" and "rc:all" mean the group; ":all" is redundant
            group_token = token.lower().replace(":all", "")
            group_token = {"rc": "ratingcurve", "power_law_all": "power_law_family"
                           }.get(group_token, group_token)
            if group_token in GROUPS:
                wanted.extend(GROUPS[group_token])
            else:
                wanted.append(get(token).key)

    order = all_keys()
    unique = list(dict.fromkeys(wanted))
    return [_BY_KEY[key] for key in sorted(unique, key=order.index)]
