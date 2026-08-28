"""Leave-one-group-out ELPD: PSIS scoring when the observations are not independent.

Why this module exists
----------------------
PSIS-LOO leaves out one *observation* at a time. That is the right unit when the
measurements are independent, which discrete field gaugings essentially are - they
are weeks or months apart. It is the wrong unit for a continuous record: leaving out
one 15-minute reading leaves the readings 15 minutes either side in the training set,
so the "held-out" point is still, in effect, in the sample. The score that comes back
is optimistic, and taking more draws will not repair it, because the problem is the
data's autocorrelation rather than the sampler's.

The fix is to leave out a whole *block* of observations - several hours by default -
so the held-out unit spans more than one reading. This is
leave-one-group-out cross-validation, and it is the same PSIS estimator: the
importance weight for dropping a group is the product of its members' likelihoods, so
summing the pointwise log-likelihood within each group and scoring the resulting
``(draws, groups)`` matrix is all it takes. No refits, and no new statistics - the
grouped matrix goes to
:func:`limnotech_rating_curves.evaluate.metrics.elpd_from_log_likelihood`, which
already applies the relative-efficiency correction ArviZ needs.

One estimator for both data types
---------------------------------
The same blocking is applied to field measurements, rather than switching estimators
by data type. USGS gaugings are effectively one per day - measured across four gages,
between 91% and 99% of gauging days carry exactly one measurement, and the most seen
on any one day was three - so any block of a day or less reduces to plain LOO for
almost every gauging. Using one estimator everywhere means a field fit and a continuous
fit are scored the same way, rather than by two schemes whose numbers would need a
disclaimer before they could be set side by side.

What is and is not comparable
-----------------------------
ELPD is a *sum* over held-out units, so its magnitude grows with the number of
observations. Totals are therefore not comparable across datasets of different size -
which is equally true of LOO against LOO, and is not something grouping introduces.
:func:`elpd_logo` reports ``elpd_per_obs`` beside the total: the total ranks models
within one dataset, the per-observation figure survives a glance across datasets.

Reading the result, and when not to
-----------------------------------
``pareto_k`` comes back one value per *group*, and on a continuous record it is not a
formality. Dropping a whole block removes far more information than dropping one
point, so the importance ratios grow heavy-tailed and PSIS can fail outright.

How badly depends on the block length **and on how much data there is** - neither
alone predicts it. Measured at USGS 06893578, one-segment power law, 15-minute
records of five lengths all ending 2023-06-30, as percent of blocks with Pareto-k
above 0.7:

======  =============  =============  =============  ==============  ==============
block   14 d (n=1437)  30 d (n=2830)  91 d (n=8677)  182 d (n=17106) 365 d (n=34735)
======  =============  =============  =============  ==============  ==============
3h      3.3            0.8            0.4            0.4             0.2
6h      11.5           4.1            0.5            1.0             0.3
12h     19.4           11.5           1.1            2.5             0.8
1D      25.0           51.6           25.0           9.3             7.7
2D      87.5           87.5           47.8           27.5            14.8
3D      83.3           90.9           93.8           48.4            21.1
======  =============  =============  =============  ==============  ==============

Two readings of that table decide the default. A day block never becomes usable, even
on a full year. Six hours is clean from about three months of record onward, and beats
a day at every length - which is why :data:`DEFAULT_BLOCK` is ``"6h"``.

It is not, however, long enough to decorrelate the data. On the 91-day record the
residual integral timescale was about 16 hours once a slow rating drift was removed
(and 12.6 days with the drift left in, which is a trend rather than short-range
dependence). The usual requirement of two to three timescales asks for 32-48 hours,
and at 48 hours PSIS already fails on 15-48% of blocks depending on record length. So
the block that is computable and the block that is decorrelated do not overlap here.
Grouping does not rescue that; it makes the failure visible, which is the useful part.

So read :func:`elpd_logo` as trustworthy when ``reliable`` is True - which in practice
means discrete field gaugings, where a block is leave-one-out anyway - and treat a
False as "this dataset cannot be ranked this way", not as a number to squint at. For a
continuous fit, the honest comparison is against the site's discrete field gaugings,
which that fit never saw: see
:func:`limnotech_rating_curves.core.fit_metrics`. :func:`..evaluate.exact_loo.reloo`
repairs individual points but not whole groups, so it is not a way out here either.
"""

import logging

import numpy as np
import pandas as pd

from . import metrics as metrics_module

log = logging.getLogger(__name__)

#: Share of groups whose Pareto-k may exceed ``settings.PARETO_K_GOOD`` before
#: :func:`elpd_logo` declares the estimate unusable. A few bad groups out of hundreds
#: is tolerable; a quarter of them, which is what a day block on a 15-minute record
#: produced in testing, is not. Defined in ``evaluate.metrics`` and re-exported here,
#: so this module and :func:`..evaluate.metrics.elpd_by_group` cannot drift apart on
#: what counts as unusable.
MAX_PCT_K_HIGH = metrics_module.MAX_PCT_K_HIGH

#: Default block length. Six hours, chosen empirically: measured on 15-minute records
#: of five different lengths at one gage, a six-hour block was the longest that stayed
#: reliably computable, and it beat a one-day block at every record length (see the
#: table in this module's docstring). On discrete field gaugings it still collapses to
#: leave-one-out, so one estimator serves both data types.
#:
#: It is a default, not a guarantee: six hours is well under the measured residual
#: correlation length of a 15-minute record, and it is only computable on records of
#: about three months or more. Check ``reliable`` on the result either way.
DEFAULT_BLOCK = "6h"


def time_blocks(time, block: str = DEFAULT_BLOCK) -> np.ndarray:
    """Assign each observation to a time block, for :func:`elpd_logo`.

    Parameters
    ----------
    time : array-like
        One timestamp per observation, in the order the model was fitted on.
    block : str, default :data:`DEFAULT_BLOCK`
        Block length as a pandas offset alias (``"6h"``, ``"12h"``, ``"1D"``, ``"7D"``).

    Returns
    -------
    numpy.ndarray
        Integer group index per observation, one distinct value per occupied block.
        Empty blocks are not numbered, so the indices are contiguous.

    Raises
    ------
    ValueError
        If any timestamp is missing. An observation that cannot be placed in time
        cannot be grouped, and dropping it silently would change what was scored.
    """
    stamps = pd.to_datetime(pd.Series(np.asarray(time).ravel()))
    if stamps.isna().any():
        raise ValueError(
            f"{int(stamps.isna().sum())} of {len(stamps)} timestamps are missing; "
            "every observation needs a time to be placed in a block")
    return pd.factorize(stamps.dt.floor(block), sort=True)[0]


def stage_bands(stage, n_bands: int = 12) -> np.ndarray:
    """Assign each observation to a stage band, for :func:`elpd_logo`.

    The complement of :func:`time_blocks`: it asks whether the curve generalizes to a
    flow range it never saw, rather than to a time it never saw. Bands are equal-width
    in stage, so a band over a sparsely measured high-flow range may hold very few
    observations - which is the point, since that is where a rating is least
    constrained.

    Parameters
    ----------
    stage : array-like
        Stage (ft), one value per observation.
    n_bands : int, default 12
        Number of equal-width bands across the observed stage range.

    Returns
    -------
    numpy.ndarray
        Integer group index per observation, contiguous over the occupied bands.
    """
    stage = np.asarray(stage, float).ravel()
    if n_bands < 2:
        raise ValueError(f"n_bands must be at least 2, got {n_bands}")
    low, high = float(np.nanmin(stage)), float(np.nanmax(stage))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("stage must span a finite, positive range to be banded")
    edges = np.linspace(low, high, n_bands + 1)
    # digitize against the interior edges keeps the maximum stage in the last band
    # rather than in a band of its own past the final edge
    index = np.clip(np.digitize(stage, edges[1:-1], right=False), 0, n_bands - 1)
    return pd.factorize(index, sort=True)[0]


def group_log_likelihood(log_likelihood, groups, log_offset: float = 0.0):
    """Sum a pointwise log-likelihood within each group.

    The whole of the leave-one-group-out machinery, in one function: the
    log-likelihood of dropping a group is the sum of its members' pointwise
    log-likelihoods, because observations enter the likelihood as a product.

    Parameters
    ----------
    log_likelihood : array-like
        Shape ``(draws, observations)``.
    groups : array-like
        Group label per observation, as from :func:`time_blocks`.
    log_offset : float, default 0.0
        Constant added to every *observation's* log-likelihood before summing, to
        move it into a common observation space. Applied per observation rather than
        per group, so a group of 96 readings is shifted by ``96 * log_offset``.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The ``(draws, n_groups)`` grouped matrix, and the number of observations in
        each group.
    """
    log_likelihood = np.asarray(log_likelihood, float)
    groups = np.asarray(groups).ravel()
    if log_likelihood.ndim != 2:
        raise ValueError("log_likelihood must be 2-D (draws, observations), got "
                         f"shape {log_likelihood.shape}")
    if groups.size != log_likelihood.shape[1]:
        raise ValueError(
            f"got {groups.size} group labels for {log_likelihood.shape[1]} "
            "observations; they must line up one to one")

    codes = pd.factorize(groups, sort=True)[0]
    n_groups = int(codes.max()) + 1 if codes.size else 0
    sizes = np.bincount(codes, minlength=n_groups)
    grouped = np.zeros((log_likelihood.shape[0], n_groups), float)
    # one accumulate pass over columns, rather than a python loop per group
    np.add.at(grouped.T, codes, log_likelihood.T)
    grouped += sizes * log_offset
    return grouped, sizes


def elpd_logo(log_likelihood, groups, log_offset: float = 0.0) -> dict:
    """PSIS leave-one-group-out ELPD from a pointwise log-likelihood.

    Parameters
    ----------
    log_likelihood : array-like
        Shape ``(draws, observations)``.
    groups : array-like
        Group label per observation - :func:`time_blocks` for time blocks,
        :func:`stage_bands` for flow ranges.
    log_offset : float, default 0.0
        Per-observation shift into a common observation space (see
        :func:`group_log_likelihood`).

    Returns
    -------
    dict
        The fields :data:`limnotech_rating_curves.evaluate.metrics.BAYES_FIELDS`
        carries, where ``n_obs`` counts the *groups* scored and ``pareto_k`` holds one
        value per group, plus:

        ``n_groups``
            Groups the ELPD was computed over.
        ``n_observations``
            Observations behind those groups.
        ``obs_per_group``
            Mean observations per group. 1.0 means this is plain leave-one-out.
        ``elpd_per_obs``
            ``elpd_loo`` divided by ``n_observations`` - the figure to compare across
            datasets of different size, since the total scales with observation count.
        ``reliable``
            False when too large a share of groups has ``pareto_k`` above
            ``settings.PARETO_K_GOOD``, which on a continuous record is the normal
            outcome at day-length blocks. A False here means the number must not be
            used to rank models - see this module's docstring for the measurements
            behind that. `note` says so too.

        NaNs with a ``note`` when the scores cannot be computed.
    """
    grouped, sizes = group_log_likelihood(log_likelihood, groups, log_offset=log_offset)
    n_observations = int(sizes.sum())
    shape = {"n_groups": int(grouped.shape[1]),
             "n_observations": n_observations,
             "obs_per_group": float(sizes.mean()) if sizes.size else np.nan}

    if grouped.shape[1] < 2:
        return {**metrics_module.unavailable(
            "leave-one-group-out needs at least 2 groups, got "
            f"{grouped.shape[1]}"), **shape,
            "elpd_per_obs": np.nan, "reliable": False}

    scores = metrics_module.elpd_from_log_likelihood(grouped)
    elpd = scores.get("elpd_loo", np.nan)
    scores.update(shape)
    scores["elpd_per_obs"] = (float(elpd) / n_observations
                              if n_observations and np.isfinite(elpd) else np.nan)

    pct_high = scores.get("pct_k_high", np.nan)
    scores["reliable"] = bool(np.isfinite(elpd) and np.isfinite(pct_high)
                              and pct_high <= MAX_PCT_K_HIGH)
    if not scores["reliable"] and np.isfinite(pct_high) and not scores["note"]:
        scores["note"] = (
            f"PSIS failed on {pct_high:.0f}% of groups (Pareto-k above "
            f"{metrics_module.PARETO_K_GOOD}); dropping a group of "
            f"{scores['obs_per_group']:.0f} observations perturbs the posterior too "
            "much to reweight. Do not rank models on this - score the fit against "
            "discrete field measurements instead.")
    return scores


def residual_correlation_length(time, residual, threshold: float = 1 / np.e):
    """How long residuals stay correlated - evidence for choosing a block length.

    A block only makes leave-one-group-out honest if it is longer than the residuals
    stay correlated. This measures that rather than assuming it, so a block length in
    a report can be defended.

    The series is resampled onto its own median sampling interval before the
    autocorrelation is taken. That is exact for an evenly sampled continuous record
    and an approximation for irregular field gaugings - on irregular data read the
    result as an order of magnitude, not a precise interval.

    Parameters
    ----------
    time : array-like
        One timestamp per residual.
    residual : array-like
        Residuals in whatever space the model was scored in. Only their correlation
        structure is used, so the scale does not matter.
    threshold : float, default ``1/e``
        Autocorrelation below which the series is called decorrelated.

    Returns
    -------
    pandas.Timedelta or None
        The smallest lag whose autocorrelation falls below `threshold`, or None when
        the series never decorrelates within half its own length, or is too short to
        say (fewer than four usable points).
    """
    frame = pd.DataFrame({
        "time": pd.to_datetime(pd.Series(np.asarray(time).ravel())),
        "residual": np.asarray(residual, float).ravel(),
    }).dropna().sort_values("time")
    if len(frame) < 4:
        return None

    gaps = frame["time"].diff().dropna()
    step = gaps[gaps > pd.Timedelta(0)].median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return None

    series = (frame.set_index("time")["residual"]
              .resample(step).mean().interpolate(limit_direction="both"))
    values = series.to_numpy(float)
    values = values - values.mean()
    denominator = float(np.dot(values, values))
    if len(values) < 4 or denominator <= 0:
        return None

    for lag in range(1, len(values) // 2):
        correlation = float(np.dot(values[:-lag], values[lag:])) / denominator
        if correlation < threshold:
            return lag * step
    return None
