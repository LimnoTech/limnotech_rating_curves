from . import settings

# PyMC and numpy need these set before they are imported (see the function's
# docstring), which is why this runs at package import rather than at fit time.
settings.apply_numerical_workarounds()

from . import support  # noqa: E402

# Console logging at INFO, sampler chatter and sampling-stack warnings silenced. Set
# LRC_NO_AUTO_LOGGING to skip it. Runs before the imports below so that the warnings
# PyMC raises as it loads are filtered too.
support.logging_setup.auto_configure()

from . import (core, data, evaluate, exports, models, ratings,  # noqa: E402
               view, workflows)
from .core import (ExternalCurve, FitResult, FoldCurve, Metrics,  # noqa: E402
                   Sample, SiteRating)
from .data import (StageDatum, ZeroFlowEstimate, datum_agreement,  # noqa: E402
                   distance_to_stage, estimate_zero_flow, flow_sheet_sample,
                   gage_sample, johnson_offset, lid_for_usgs_site,
                   published_rating, rating_curve_sheet, sensor_sample,
                   station_sample, to_gage_height)
from .evaluate import (CrossValidation, acquisition_curve,  # noqa: E402
                       convergence, convergence_report, cross_validate,
                       elpd_logo, plot_acquisition,
                       residual_correlation_length, stage_bands, stage_weight,
                       time_blocks)
from .exports import (SavedRating, curve_table, export_directory,  # noqa: E402
                      load_rating, load_ratings, manifest_index, save_fit,
                      save_rating, save_ratings, save_site)
from .ratings import (Bdrc, Exponential, PowerLaw, Quadratic,  # noqa: E402
                      RatingModel, RatingSet, Spline, compare, fit_rating,
                      rating_model)
from .models.hierarchical import (HierarchicalFit,  # noqa: E402
                                  HierarchicalPowerLaw, fit_hierarchical)
from .workflows import RatingCollection, fit_many  # noqa: E402

__version__ = "1.1.0"

#: Alias kept so existing callers of the older ``lrc.fit(...)`` keep working; it is
#: :func:`compare`, which fits several models and returns a :class:`RatingSet`.
fit = compare


def report(**kwargs):
    """Fit, score and map many MAGL sites at once.

    A re-export of :func:`limnotech_rating_curves.workflows.batch.run`.

    Parameters
    ----------
    **kwargs
        Forwarded to :func:`limnotech_rating_curves.workflows.batch.run`.

    Returns
    -------
    workflows.batch.Report
    """
    return workflows.batch.run(**kwargs)


__all__ = [
    # fitting
    "fit_rating", "compare", "fit", "rating_model",
    "PowerLaw", "Spline", "Bdrc", "Quadratic", "Exponential", "RatingModel",
    "RatingSet",
    # every site in one fit, so a short record borrows what it cannot measure
    "fit_hierarchical", "HierarchicalFit", "HierarchicalPowerLaw",
    # many at once
    "fit_many", "RatingCollection", "report",
    # data
    "Sample", "Metrics", "FitResult", "FoldCurve", "SiteRating", "ExternalCurve",
    "gage_sample", "station_sample", "sensor_sample", "published_rating",
    "rating_curve_sheet", "flow_sheet_sample",
    "lid_for_usgs_site", "datum_agreement",
    # stage datum and zero flow
    "StageDatum", "to_gage_height", "distance_to_stage",
    "johnson_offset", "estimate_zero_flow", "ZeroFlowEstimate",
    # evaluation
    "cross_validate", "CrossValidation", "convergence", "convergence_report",
    "elpd_logo", "time_blocks", "stage_bands", "residual_correlation_length",
    # where to gauge next
    "acquisition_curve", "stage_weight", "plot_acquisition",
    # saving and loading
    "save_rating", "save_ratings", "save_fit", "save_site", "load_rating",
    "load_ratings", "SavedRating", "curve_table", "manifest_index",
    "export_directory",
    # subpackages and modules
    "data", "evaluate", "view", "models", "workflows", "support",
    "core", "ratings", "exports", "settings",
    "__version__",
]
