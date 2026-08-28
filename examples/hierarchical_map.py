"""Fit every MAGL sensor and its colocated USGS gages in one hierarchical model,
then draw the result on the interactive map.

One joint fit, so a sensor with two or three measurements borrows the scatter it
cannot measure from the population its cluster belongs to. The USGS gages are fitted
alongside as ordinary sites over the last five years - they carry the measurements
that determine the population.

    python examples/hierarchical_map.py --out hierarchical_map.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

from limnotech_rating_curves.models.hierarchical import fit_hierarchical
from limnotech_rating_curves.export.mapview import build_map
from limnotech_rating_curves.data.magl import assemble_magl_sites

#: Fewest measurements to include a sensor. Two is the point of the hierarchical
#: model: a standalone power law spends three parameters on the mean and cannot fit it.
MIN_POINTS = 2

#: How far back USGS measurements are taken. A channel changes, so an old gaging
#: describes a rating that no longer applies.
USGS_YEARS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("hierarchical_map.html"),
                        help="where to write the map")
    parser.add_argument("--sources", nargs="+", default=["magl", "usgs_gage"],
                        help="which sample sources to include")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="restrict to these sensors, clusters or gage ids")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS)
    parser.add_argument("--years", type=int, default=USGS_YEARS,
                        help="how far back to take USGS measurements")
    parser.add_argument("--exports", action="store_true",
                        help="embed each fit's manifest and posterior in the map as a "
                             "downloadable zip, and add the button that saves it")
    parser.add_argument("--reloo", action="store_true",
                        help="refit the measurements PSIS could not handle and rescore "
                             "them exactly. One refit is a whole joint fit, so budget "
                             "it with --max-refits and expect about a minute each")
    parser.add_argument("--max-refits", type=int, default=None,
                        help="hard ceiling on --reloo refits")
    args = parser.parse_args()

    sites = assemble_magl_sites(sites=args.sites, sources=args.sources,
                                min_points=args.min_points, years=args.years)

    fit = fit_hierarchical([site.sample for site in sites], reference=False,
                           reloo=args.reloo, max_refits=args.max_refits)
    for site in sites:
        site.attach_marginal_fit(fit)
    print(fit.summary())

    path = build_map(sites, output_html=args.out, exports=args.exports,
                     title="Hierarchical power law. Hover a site to see its fit.")
    print(f"map written to {path}")


if __name__ == "__main__":
    main()
