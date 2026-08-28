"""Fit each site on its own, the way the package has always done it, and draw the map.

One independent fit per site per model - the counterpart to
:mod:`examples.hierarchical_map`, which fits every site together. Run both and compare:
the sites with two or three measurements are where they part company.

    python examples/traditional_map.py --out traditional_map.html

A sensor needs at least ``--min-points`` measurements to be fitted at all. A one-segment
power law spends three parameters on the mean, so three is the fewest that can be fitted
independently and four the fewest that leaves anything to estimate scatter from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from limnotech_rating_curves import settings
from limnotech_rating_curves.models import catalog
from limnotech_rating_curves.export.mapview import build_map
from limnotech_rating_curves.data.magl import assemble_magl_sites

#: Fewest measurements to attempt an independent fit. Below this there is nothing to
#: fit three parameters to.
MIN_POINTS = 3

#: How far back USGS measurements are taken. A channel changes, so an old gaging
#: describes a rating that no longer applies.
USGS_YEARS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("traditional_map.html"),
                        help="where to write the map")
    parser.add_argument("--models", nargs="+", default=["power_law"],
                        help="model keys, as catalog.select accepts them")
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
    args = parser.parse_args()

    sites = assemble_magl_sites(sites=args.sites, sources=args.sources,
                                min_points=args.min_points, years=args.years)
    entries = catalog.select(args.models)

    for site in sites:
        if len(site.sample) == 0:
            print(f"{site.sample_id}: no measurements, skipped")
            continue
        for entry in entries:
            fit = entry.fit(site.sample)
            site.attach_fit(fit)
            print(f"{site.sample_id} / {entry.key}: n={fit.n}, {fit.status}")

    path = build_map(sites, output_html=args.out, exports=args.exports,
                     title="Independent fits. Hover a site to see its models.")
    print(f"map written to {path}")


if __name__ == "__main__":
    main()
