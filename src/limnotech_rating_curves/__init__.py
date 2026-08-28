from . import settings

# PyMC and numpy need these set before they are imported (see the function's
# docstring), which is why this runs at package import rather than at fit time.
settings.apply_numerical_workarounds()

from . import helpers  # noqa: E402

# Console logging at INFO, sampler chatter and sampling-stack warnings silenced. Set
# LRC_NO_AUTO_LOGGING to skip it. Runs before the imports below so that the warnings
# PyMC raises as it loads are filtered too.
helpers.logging_setup.auto_configure()

from . import (active_learning, core, data, export, model_selection,  # noqa: E402
               models, ratings, site)
from .core import ExternalCurve, Fit, FoldCurve, Sample  # noqa: E402
from .model_selection.metrics import Metrics  # noqa: E402
from .site import SiteRating  # noqa: E402
from .data import (StageDatum, ZeroFlowEstimate, datum_agreement,  # noqa: E402
                   distance_to_stage, estimate_zero_flow, flow_sheet_sample,
                   gage_sample, johnson_offset, lid_for_usgs_site,
                   published_rating, rating_curve_sheet, sensor_sample,
                   station_sample, to_gage_height)
from .model_selection import (CrossValidation, convergence,  # noqa: E402
                              convergence_report, cross_validate)
from .active_learning import (acquisition_curve, plot_acquisition,  # noqa: E402
                              stage_weight)
from .export.exports import (SavedRating, curve_table,  # noqa: E402
                             export_directory, load_rating, load_ratings,
                             manifest_index, save_fit, save_rating,
                             save_ratings, save_site)
from .ratings import (Bdrc, Exponential, PowerLaw, Quadratic,  # noqa: E402
                      RatingModel, RatingSet, Spline, compare, fit_rating,
                      rating_model)
from .models.hierarchical import (HierarchicalFit,  # noqa: E402
                                  HierarchicalPowerLaw, fit_hierarchical)

__version__ = "1.1.0"

#: Alias kept so existing callers of the older ``lrc.fit(...)`` keep working; it is
#: :func:`compare`, which fits several models and returns a :class:`RatingSet`.
fit = compare


__all__ = [
    # fitting
    "fit_rating", "compare", "fit", "rating_model",
    "PowerLaw", "Spline", "Bdrc", "Quadratic", "Exponential", "RatingModel",
    "RatingSet",
    # every site in one fit, so a short record borrows what it cannot measure
    "fit_hierarchical", "HierarchicalFit", "HierarchicalPowerLaw",
    # data
    "Sample", "Metrics", "Fit", "FoldCurve", "SiteRating", "ExternalCurve",
    "gage_sample", "station_sample", "sensor_sample", "published_rating",
    "rating_curve_sheet", "flow_sheet_sample",
    "lid_for_usgs_site", "datum_agreement",
    # stage datum and zero flow
    "StageDatum", "to_gage_height", "distance_to_stage",
    "johnson_offset", "estimate_zero_flow", "ZeroFlowEstimate",
    # evaluation
    "cross_validate", "CrossValidation", "convergence", "convergence_report",
    # where to gauge next
    "acquisition_curve", "stage_weight", "plot_acquisition",
    # saving and loading
    "save_rating", "save_ratings", "save_fit", "save_site", "load_rating",
    "load_ratings", "SavedRating", "curve_table", "manifest_index",
    "export_directory",
    # subpackages and modules
    "data", "model_selection", "active_learning", "export", "models",
    "helpers", "core", "site", "ratings", "settings",
    "__version__",
]
