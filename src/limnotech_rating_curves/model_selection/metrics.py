import logging

import numpy as np
from dataclasses import dataclass

from ..core import Fit

log = logging.getLogger(__name__)


@dataclass
class Metrics:
    """How well one fitted rating matches its measurements.

    The first block is ordinary in-sample goodness of fit; the second is Bayesian
    model comparison, where higher `elpd_loo` is better. NaN means the score was
    not available (a fit whose posterior was not kept, a reference curve with no
    posterior at all).

    Attributes
    ----------
    n : int
        Measurements the scores were computed over.
    nse : float
        Nash-Sutcliffe efficiency on discharge. 1 is perfect; 0 means the fit is
        no better than the mean observed discharge.
    rmse : float
        Root-mean-square error in cfs. In the units of the data, so it is
        dominated by the high-flow end.
    r2_log : float
        R-squared computed on log discharge. This is the one to read for a
        rating: it weighs a factor-of-two error at low flow the same as at high
        flow, which is how rating error is normally judged.
    elpd_loo : float
        Expected log pointwise predictive density, estimated by PSIS-LOO - an
        estimate of how well the model would predict a measurement it had not
        seen. Higher is better. Comparable across every model in this package
        because they are all put in the same observation space (see
        :mod:`limnotech_rating_curves.model_selection.elpd`).
    se_loo : float
        Standard error of `elpd_loo`. Two models whose ELPD differ by less than
        about two of these are not distinguishable by this sample.
    p_loo : float
        Effective number of parameters implied by LOO. Much larger than the
        model's actual parameter count is a sign of misfit or of an influential
        observation.
    elpd_waic : float
        The same quantity estimated by WAIC, as a cross-check on `elpd_loo`.
    pareto_k_max : float
        Worst per-observation Pareto-k from the PSIS-LOO importance sampling.
        Above ``settings.PARETO_K_GOOD`` (0.7) the LOO estimate is unreliable for
        that point, which in practice means one measurement dominates the fit.
    """

    n: int
    nse: float
    rmse: float
    r2_log: float
    elpd_loo: float = float("nan")
    se_loo: float = float("nan")
    p_loo: float = float("nan")
    elpd_waic: float = float("nan")
    pareto_k_max: float = float("nan")

    @classmethod
    def from_fit(cls, fit: "Fit") -> "Metrics":
        """Assemble the scores recorded on a :class:`Fit`.

        Parameters
        ----------
        fit : Fit
            A completed fit. Its ``metrics`` dict supplies the in-sample block
            and its ``bayes`` dict the model-comparison block; missing entries
            become NaN.

        Returns
        -------
        Metrics
        """
        in_sample = fit.metrics or {}
        bayes = fit.bayes or {}
        nan = float("nan")
        return cls(
            n=int(in_sample.get("n", 0)),
            nse=in_sample.get("nse", nan), rmse=in_sample.get("rmse", nan),
            r2_log=in_sample.get("r2_log", nan),
            elpd_loo=bayes.get("elpd_loo", nan), se_loo=bayes.get("se_loo", nan),
            p_loo=bayes.get("p_loo", nan), elpd_waic=bayes.get("elpd_waic", nan),
            pareto_k_max=bayes.get("pareto_k_max", nan))

    def to_dict(self) -> dict:
        """The scores as a plain dict, in table order."""
        return {"n": self.n, "nse": self.nse, "rmse": self.rmse,
                "r2_log": self.r2_log,
                "elpd_loo": self.elpd_loo, "se_loo": self.se_loo,
                "p_loo": self.p_loo, "elpd_waic": self.elpd_waic,
                "pareto_k_max": self.pareto_k_max}

    def __repr__(self) -> str:
        return (f"Metrics(n={self.n}, nse={self.nse:.3f}, rmse={self.rmse:.3g}, "
                f"r2_log={self.r2_log:.3f}, elpd_loo={self.elpd_loo:.1f})")


def fit_metrics(observed, predicted) -> dict:
    """In-sample goodness of fit for paired observed and predicted discharge.

    Parameters
    ----------
    observed, predicted : array-like
        Discharge in cfs, same length. Pairs where either value is not finite are
        dropped before scoring, and `n` reports how many survived.

    Returns
    -------
    dict
        ``rmse``, ``nse``, ``r2_log``, ``n`` - see
        :class:`Metrics` for what each means. NSE is NaN when the observations
        have no variance (nothing to explain), and ``r2_log`` is NaN when fewer
        than two pairs are positive.
    """
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    usable = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[usable], predicted[usable]
    if observed.size == 0:
        return {"rmse": np.nan, "nse": np.nan, "r2_log": np.nan, "n": 0}

    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    variance = float(np.sum((observed - observed.mean()) ** 2))
    nse = float(1 - np.sum(residual ** 2) / variance) if variance > 0 else np.nan

    positive = (observed > 0) & (predicted > 0)
    if positive.sum() >= 2:
        log_observed = np.log(observed[positive])
        log_predicted = np.log(predicted[positive])
        residual_sum = float(np.sum((log_observed - log_predicted) ** 2))
        total_sum = float(np.sum((log_observed - log_observed.mean()) ** 2))
        r2_log = float(1 - residual_sum / total_sum) if total_sum > 0 else np.nan
    else:
        r2_log = np.nan

    return {"rmse": rmse, "nse": nse, "r2_log": r2_log,
            "n": int(observed.size)}
