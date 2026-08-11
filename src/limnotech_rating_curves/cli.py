import argparse
import logging
import sys
from pathlib import Path

from .workflows import batch
from . import exports, ratings, settings
from .support import logging_setup
from .models import catalog

log = logging.getLogger(__name__)

#: What ``rating-curves --help`` prints above the options. Held here rather than
#: taken from the module docstring, which lives in ``docs/cli.md``.
DESCRIPTION = """\
Command line for the batch pipeline and for the ratings it writes.

Fitting, the default with no subcommand:

    rating-curves --list-sites                 the available sites
    rating-curves -v                           every site, curated models, then the map
    rating-curves --site SBR-09 SR --cv        one sensor and one cluster, with the
                                               cross-validation pane
    rating-curves --models bdrc --png          the four bdrc variants, plus the static
                                               figures
    rating-curves --data measurements.csv      fit a CSV of measurements

Working with what a run saved:

    rating-curves list output/fitted_curves            the contents of a directory
    rating-curves inspect output/fitted_curves/*.rating.json    one rating in full
    rating-curves export output/fitted_curves --to ship/        a portable copy + CSVs

list and inspect read the manifests only, so they run on a machine with neither
PyMC nor a compiler."""

#: Subcommands the CLI dispatches on. Anything else is read as an option of the
#: default fitting run, so ``rating-curves -v`` keeps working unchanged.
SUBCOMMANDS = ("export", "inspect", "show", "list")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="rating-curves", description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    what = parser.add_argument_group("what to run")
    what.add_argument("--data", metavar="CSV", default=None,
                      help="fit a CSV of your own measurements instead of the MAGL "
                           "network; it needs a stage column and a discharge column")
    what.add_argument("--stage-column", default=None,
                      help="--data: name of the stage column, if it is not one of "
                           "the recognized names")
    what.add_argument("--discharge-column", default=None,
                      help="--data: name of the discharge column")
    what.add_argument("--stage-datum", default=None,
                      help="--data: convert stage to gage height first - a number "
                           "(the elevation of gage height zero) or 'lowest'")
    what.add_argument("--site", nargs="*", default=None, metavar="TOKEN",
                      help="MAGL-network sites to run: a sensor (SBR-09), a "
                           "co-located station (SR-08), a cluster (SR), or a USGS "
                           "gage id (04176356). Default: all. See --list-sites.")
    what.add_argument("--source", nargs="*", default=None, choices=batch.SOURCES,
                      dest="sources",
                      help="restrict to these sample sources (default: all)")
    what.add_argument("--models", nargs="*", default=None,
                      help="models to fit (default: a curated set). Keys, or a "
                           "group: all / bdrc / ratingcurve. Keys: "
                           + ", ".join(catalog.all_keys()))
    what.add_argument("--list-sites", action="store_true",
                      help="print the MAGL-network sites you can select, and exit")
    what.add_argument("--list-models", action="store_true",
                      help="print the models you can fit, and exit")

    how = parser.add_argument_group("how to fit")
    how.add_argument("--method", default="nuts", choices=settings.METHODS,
                     help="all-data fitting method: nuts (default; needed for a "
                          "valid predictive score) or advi (faster, but the ELPD "
                          "and Pareto-k are then untrustworthy)")
    how.add_argument("--nuts-sampler", default=None, choices=settings.NUTS_SAMPLERS,
                     help=f"which NUTS implementation (default "
                          f"{settings.NUTS_SAMPLER}). Same posterior either way.")
    how.add_argument("--zero-flow", default=None,
                     help="stage of zero flow: a number, 'johnson' to estimate it "
                          "by Johnson's three-point method, or 'infer' to leave it "
                          "to each model (default: each model's own best choice)")
    how.add_argument("--seed", type=int, default=settings.SEED, help="RNG seed")
    how.add_argument("--min-points", type=int, default=3,
                     help="fewest discharge measurements for a MAGL sensor to be "
                          "included (default 3)")
    how.add_argument("--refresh", action="store_true",
                     help="bypass the fetch caches and refetch")

    cv = parser.add_argument_group("cross-validation (opt-in; this is the slow part)")
    cv.add_argument("--cv", action="store_true",
                    help="also refit on subsets and score the held-out measurements, "
                         "and build the map's cross-validation pane")
    cv.add_argument("--cv-scheme", default="auto", choices=("auto", "holdout", "loo"),
                    help="auto (default) picks leave-one-out for small samples and "
                         "the holdout sweep for large ones")
    cv.add_argument("--cv-splits", type=int, default=None,
                    help=f"folds per site (holdout default {settings.CV_SPLITS})")
    cv.add_argument("--cv-holdout", type=float, default=None,
                    help=f"fraction held out per fold (default {settings.CV_HOLDOUT})")
    cv.add_argument("--cv-train-n", type=int, default=None,
                    help="absolute training points per fold, overriding --cv-holdout")

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", default=None,
                        help=f"where the outputs go (default {settings.OUTPUT_DIR})")
    output.add_argument("--png", action="store_true",
                        help="also write the static figures")
    output.add_argument("--no-csv", action="store_true",
                        help="skip the results tables")
    output.add_argument("--no-posteriors", action="store_true",
                        help="skip exporting the fitted posteriors to NetCDF")

    noise = parser.add_argument_group("logging")
    noise.add_argument("-v", "--verbose", action="store_true",
                       help="debug-level logging")
    noise.add_argument("-q", "--quiet", action="store_true",
                       help="warnings and errors only")
    return parser


def _print_sites(min_points: int) -> None:
    """Print the selectable MAGL-network sites."""
    from .data import magl
    directory = magl.site_directory(min_points)
    print("Sites you can pass to --site (a MAGL sensor, a co-located station, a "
          "cluster label, or a USGS gage id):\n")
    print(f"  clusters : {', '.join(directory['clusters'])}")
    print(f"  gages    : {', '.join(directory['gages'])}")
    print(f"  stations : {', '.join(directory['colocated_stations'])}"
          f"   (co-located with a gage)")
    print(f"  sensors  : {', '.join(directory['sensors'])}"
          f"   (MAGL, at least {min_points} discharge measurements)")
    print("\n  --source restricts the sample type: " + ", ".join(batch.SOURCES))


def _print_models() -> None:
    """Print the model catalog."""
    print("Models you can pass to --models:\n")
    for entry in catalog.MODELS:
        print(f"  {entry.key:<16} {entry.label}  "
              f"(needs {entry.min_points}+ measurements)")
    print("\nGroups: " + ", ".join(sorted(catalog.GROUPS)))
    print("Default: " + ", ".join(catalog.DEFAULT_KEYS))


def _run_own_data(args) -> int:
    """Fit and report on a CSV of the caller's own measurements."""
    import pandas as pd

    frame = pd.read_csv(args.data)
    stage_datum = args.stage_datum
    if stage_datum is not None:
        try:
            stage_datum = float(stage_datum)
        except ValueError:
            pass                        # a keyword like "lowest"

    comparison = ratings.compare(
        frame, discharge=args.discharge_column, stage=args.stage_column,
        models=args.models, stage_datum=stage_datum, method=args.method,
        seed=args.seed, zero_flow=args.zero_flow, nuts_sampler=args.nuts_sampler)
    print(comparison.metrics.to_string(index=False))
    print()
    print(comparison.ranking().to_string(index=False))
    for key, reason in comparison.failures.items():
        print(f"  {key} did not fit: {reason}")

    if args.cv:
        result = comparison.cross_validate(scheme=args.cv_scheme,
                                          holdout=args.cv_holdout,
                                          n_splits=args.cv_splits,
                                          n_train=args.cv_train_n, seed=args.seed)
        print(f"\ncross-validation - {result.scheme['description']}")
        print(result.headline.to_string())
    return 0


# ---------------------------------------------------------------------------
# the saved-rating subcommands
# ---------------------------------------------------------------------------

def build_export_parser() -> argparse.ArgumentParser:
    """Parser for ``rating-curves export``."""
    parser = argparse.ArgumentParser(
        prog="rating-curves export",
        description="Collect a directory of saved ratings into a portable set: the "
                    "manifest/posterior pairs, an index of what is in it, and a "
                    "plain CSV of every fitted curve for a spreadsheet.")
    parser.add_argument("source", help="a directory of .rating.json / .nc pairs, "
                                       "e.g. output/fitted_curves")
    parser.add_argument("--to", dest="destination", default=None,
                        help="where the portable set goes (default: the source "
                             "directory, so only the CSVs are written)")
    parser.add_argument("--no-csv", action="store_true",
                        help="skip ratings_index.csv and rating_curves.csv")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def build_inspect_parser(prog: str) -> argparse.ArgumentParser:
    """Parser for ``rating-curves inspect`` / ``show``."""
    parser = argparse.ArgumentParser(
        prog=f"rating-curves {prog}",
        description="Print what a saved rating holds - site, model, equation, stage "
                    "range, scores, convergence, and whether its posterior is "
                    "intact. Reads the manifest only, so it needs no PyMC.")
    parser.add_argument("paths", nargs="+",
                        help="one or more .rating.json manifests, or a directory")
    parser.add_argument("--curve", action="store_true",
                        help="also print the fitted curve as a table")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def build_list_parser() -> argparse.ArgumentParser:
    """Parser for ``rating-curves list``."""
    parser = argparse.ArgumentParser(
        prog="rating-curves list",
        description="One line per saved rating in a directory, with its scores and "
                    "whether the pair is intact.")
    parser.add_argument("directory", nargs="?", default=str(settings.POSTERIOR_DIR),
                        help=f"directory to read (default {settings.POSTERIOR_DIR})")
    parser.add_argument("--csv", default=None,
                        help="also write the index to this CSV path")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _manifest_paths(paths) -> list:
    """Expand the paths given to ``inspect`` into manifest files."""
    resolved = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(sorted(path.glob(f"*{exports.MANIFEST_SUFFIX}")))
        else:
            resolved.append(path)
    return resolved


def _run_export(argv) -> int:
    args = build_export_parser().parse_args(argv)
    logging_setup.configure(verbose=args.verbose)
    destination = args.destination or args.source
    outcome = exports.export_directory(args.source, destination,
                                       write_csv=not args.no_csv)
    print(f"{outcome['ratings']} rating(s) -> {destination}")
    for path in outcome["files"]:
        if path.endswith(".csv"):
            print(f"  table   {path}")
    for problem in outcome["problems"]:
        print(f"  PROBLEM {problem}")
    if outcome["orphans"]:
        print(f"  {len(outcome['orphans'])} posterior file(s) in {args.source} have "
              f"no manifest and were not exported - a .nc alone records no site and "
              f"no curve, so it cannot be loaded:")
        for name in outcome["orphans"][:10]:
            print(f"    {name}")
        if len(outcome["orphans"]) > 10:
            print(f"    ... and {len(outcome['orphans']) - 10} more")
    if outcome["ratings"] == 0:
        return 1
    return 0 if not outcome["problems"] else 1


def _run_inspect(argv, prog: str) -> int:
    args = build_inspect_parser(prog).parse_args(argv)
    logging_setup.configure(verbose=args.verbose)
    paths = _manifest_paths(args.paths)
    if not paths:
        print("no rating manifests found")
        return 1
    failed = 0
    for path in paths:
        try:
            saved = exports.load_rating(path)
        except Exception as exc:  # noqa: BLE001 - report and carry on
            print(f"{path}: {type(exc).__name__}: {exc}")
            failed += 1
            continue
        print(saved.describe())
        if args.curve:
            print(saved.curve.to_string(index=False))
        print()
    return 1 if failed else 0


def _run_list(argv) -> int:
    args = build_list_parser().parse_args(argv)
    logging_setup.configure(verbose=args.verbose)
    loaded = exports.load_ratings(args.directory)
    index = exports.manifest_index(loaded)
    if index.empty:
        print(f"no saved ratings in {args.directory}")
        orphans = exports.orphan_posteriors(args.directory)
        if orphans:
            print(f"  ({len(orphans)} posterior file(s) with no manifest - not "
                  f"loadable; see `rating-curves export`)")
        return 1
    columns = ["site", "model", "n", "method", "nse", "r2_log", "elpd_loo",
               "pareto_k_max", "worst_r_hat", "intact"]
    print(index[columns].to_string(index=False))
    print(f"\n{len(index)} rating(s) in {args.directory}")
    broken = index[~index["intact"]]
    for row in broken.itertuples():
        print(f"  PROBLEM {row.problem}")
    if args.csv:
        index.to_csv(args.csv, index=False)
        print(f"index -> {args.csv}")
    return 1 if len(broken) else 0


def main(argv=None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments. Defaults to ``sys.argv[1:]``. A first argument naming one of
        ``SUBCOMMANDS`` dispatches there; anything else is the default fitting run.

    Returns
    -------
    int
        Process exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        command, rest = argv[0], argv[1:]
        if command == "export":
            return _run_export(rest)
        if command == "list":
            return _run_list(rest)
        return _run_inspect(rest, command)

    parser = build_parser()
    args = parser.parse_args(argv)
    logging_setup.configure(verbose=args.verbose, quiet=args.quiet)

    if args.list_models:
        _print_models()
        return 0
    if args.list_sites:
        _print_sites(args.min_points)
        return 0

    try:
        catalog.select(args.models)
    except KeyError as exc:
        parser.error(str(exc).strip('"'))

    if args.data:
        return _run_own_data(args)

    report = batch.run(
        models=args.models, sites=args.site, sources=args.sources,
        min_points=args.min_points, refresh=args.refresh, method=args.method,
        seed=args.seed, zero_flow=args.zero_flow, nuts_sampler=args.nuts_sampler,
        cross_validate=args.cv, cv_scheme=args.cv_scheme, cv_splits=args.cv_splits,
        cv_holdout=args.cv_holdout, cv_train_n=args.cv_train_n,
        write_csv=not args.no_csv, write_png=args.png,
        write_posteriors=not args.no_posteriors, output_dir=args.output_dir)
    print(f"\nmap: {report.map_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
