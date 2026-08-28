"""One monitoring site and every rating fitted at it.

This sits a layer above :mod:`limnotech_rating_curves.core`: a `SiteRating` collects
what the model families produce, so it can name their fit types where `core` -
the foundation they are all built on - cannot.
"""

import logging
from dataclasses import dataclass, field

from .core import ExternalCurve, Fit, Sample
from .models.hierarchical import HierarchicalFit

log = logging.getLogger(__name__)


@dataclass
class SiteRating:
    """Every model fitted at one monitoring site, plus where the site is.

    This is what a run produces per site and what the map draws.

    Attributes
    ----------
    sample_id : str
        Which rating this is: ``"<source>:<site>"``, one of
        ``"usgs_gage:04176356"``, ``"colocated:SR-08"`` or ``"magl:SBR-09"``.

        The source is in the key because ``site_id`` alone collides. A MAGL sensor
        that is also co-located with a gage produces two ratings from one sensor:
        ``"magl:SBR-01"`` is its stage against MAGL flow-sheet discharge, and
        ``"colocated:SBR-01"`` is the same stage against USGS field gagings. Both
        samples carry ``site_id="SBR-01_WL"``, so keying on that would let one
        overwrite the other. LC-01, ND-01 and SBR-01 are all in this position.

        This is not :attr:`Sample.site_id`, which is the bare place the
        measurements came from (``"04176356"``, ``"SBR-09"``) and is what titles a
        plot and names an export file. ``sample_id`` keys a site in a run over many of them;
        ``site_id`` labels a sample for a human.
    source : str
        Which sample construction it came from.
    label : str
        Human name for the site.
    sample : Sample
        The measurements.
    fits : list of Fit
        One entry per model attempted, successful or not.
    reference : dict or None
        The **USGS published rating** to compare against, as
        ``{"label", "curve", "metrics"}``. Only USGS gages have one, and nothing else
        belongs here - a curve that is not a published agency rating must not be typed
        as one. See :attr:`extra_curves`.
    extra_curves : list of ExternalCurve
        Any other curve this package did not fit: a field spreadsheet's typed
        equation, a curve from a previous study. Each carries its own ``kind``, so the
        results table and the map can say what sort of thing it is.
    coords : tuple or None
        ``(latitude, longitude)``, for the map.
    group : str or None
        Optional grouping label (for MAGL, the cluster).
    station : str or None
        The sensor / station name, where one applies.
    gage : str or None
        The USGS gage number, where one applies.
    """

    sample_id: str
    source: str
    label: str
    sample: Sample
    fits: list = field(default_factory=list)
    reference: dict | None = None
    extra_curves: list = field(default_factory=list)
    coords: tuple | None = None
    group: str | None = None
    station: str | None = None
    gage: str | None = None

    @property
    def stage_label(self) -> str:
        """The sample's stage-axis description."""
        return self.sample.stage_label

    @property
    def successful_fits(self) -> list[Fit]:
        """Only the fits that succeeded."""
        return [fit for fit in self.fits if fit.ok]

    def attach_fit(self, fit: Fit) -> Fit:
        """Add one fitted model's fit to the :attr:`fits` list.

        Fits accumulate, so a site can hold a fit from each model that was tried
        and the score table can compare them. A fit replaces an earlier one with
        the same :attr:`Fit.key`, so re-attaching re-reads rather than
        duplicating.

        Parameters
        ----------
        fit : Fit
            The fit to attach.

        Returns
        -------
        Fit
            The fit that was attached, unchanged.
        """
        self.fits = [existing for existing in self.fits
                     if existing.key != fit.key] + [fit]
        return fit

    def attach_marginal_fit(self, fit: HierarchicalFit) -> Fit | None:
        """Take this site's marginal out of a joint fit and attach it.

        A hierarchical fit is one posterior over every site at once. This site's
        rating is a marginal of it, filed in ``fit.results`` under
        :attr:`sample_id`; pulling it out puts it on :attr:`fits` where the map and
        the results table read it like any other model's.

        Parameters
        ----------
        fit : HierarchicalFit
            The joint fit to read from.

        Returns
        -------
        Fit or None
            The marginal that was attached, or ``None`` if the joint fit skipped
            this site - in which case :attr:`fits` is left as it was.

        Examples
        --------
        >>> for site in sites:                 # doctest: +SKIP
        ...     site.attach_marginal_fit(fit)      # doctest: +SKIP
        """
        marginal = fit.results.get(self.sample_id)
        return self.attach_fit(marginal) if marginal is not None else None

    def external_curves(self) -> list[ExternalCurve]:
        """Every curve at this site that this package did not fit, in one list.

        The USGS published rating (if any) first, then :attr:`extra_curves`. Callers
        that draw or score "everything that is not a fit" go through here so the
        published rating and a spreadsheet equation are handled by one code path
        while keeping their different ``kind``.

        Returns
        -------
        list of ExternalCurve
        """
        curves = []
        if self.reference:
            curves.append(ExternalCurve(
                key="published_reference", label=self.reference["label"],
                kind="published_reference", curve=self.reference["curve"],
                metrics=self.reference.get("metrics", {}),
                detail={"source": "USGS published rating",
                        "gage": self.gage or ""}))
        curves.extend(self.extra_curves)
        return curves
