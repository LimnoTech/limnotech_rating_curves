"""Configure the package from a .env file, and print what it actually took.

Every tunable value in the package lives in
:mod:`limnotech_rating_curves.settings` and is read from ``os.environ`` under
``LRC_`` + the setting's name. So a ``.env`` file configures the whole package with no
code change and nothing installed in a particular place - which is what makes this
work for a copy installed with ``pip install git+...`` as well as for a checkout.

    cp .env.example .env
    python examples/settings_from_env.py

The one rule: **load the file before importing the package.** ``settings.py`` reads
the environment once, when it is first imported, so a ``load_dotenv()`` afterwards
changes nothing already read. That is why the import of ``limnotech_rating_curves``
below sits inside ``main()`` rather than at the top of the file.

Each setting printed is marked ``from env`` or ``default``, so editing your ``.env``
and rerunning shows exactly what changed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


#: The settings printed: enough to cover every type a setting can have - a path, an
#: int, a float, a bool, a tuple, a string.
SHOWN = ("DATA_DIR", "MAGL_SITES_DIR", "OUTPUT_DIR",
         "SEED", "NUTS_DRAWS", "NUTS_CHAINS", "NUTS_SAMPLER",
         "NUTS_TARGET_ACCEPT", "RELOO", "RELOO_MAX_FRACTION",
         "DEFAULT_MODEL_KEYS", "HIER_EXPONENT_BOUNDS",
         "NOAA_TIMEOUT", "USGS_TIMEOUT", "MAGL_STAGE_MATCH_TOLERANCE",
         "MAP_FIGURE_HEIGHT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, default=None,
                        help="the file to load; by default python-dotenv searches "
                             "upward from the working directory for a .env")
    parser.add_argument("--override", action="store_true",
                        help="let the file win over variables already set in the "
                             "shell (python-dotenv leaves the shell's value alone "
                             "by default)")
    args = parser.parse_args()

    from dotenv import find_dotenv, load_dotenv

    path = str(args.env_file) if args.env_file else find_dotenv(usecwd=True)
    if path and Path(path).exists():
        load_dotenv(path, override=args.override)
        print(f"loaded {path}")
    else:
        print("no .env found - showing the package's own defaults. "
              "Copy .env.example to .env and edit it to change them.")

    # Imported only now, because settings.py reads os.environ at import time.
    from limnotech_rating_curves import settings

    width = max(len(name) for name in SHOWN)
    print(f"\n{'setting':<{width}}  {'source':<8}  value")
    print("-" * (width + 50))
    for name in SHOWN:
        source = "from env" if f"LRC_{name}" in os.environ else "default"
        print(f"{name:<{width}}  {source:<8}  {getattr(settings, name)}")

    unshown = sorted(name for name in os.environ
                     if name.startswith("LRC_") and name[4:] not in SHOWN)
    if unshown:
        print(f"\nalso set, not printed above: {', '.join(unshown)}")

    print("\nEvery setting in settings.py works the same way; that file lists them "
          "all with their defaults.")


if __name__ == "__main__":
    main()
