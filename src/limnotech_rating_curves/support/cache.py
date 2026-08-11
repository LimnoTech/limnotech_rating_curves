import hashlib
import logging
from pathlib import Path

import pandas as pd

from .. import settings

log = logging.getLogger(__name__)


def cache_key(key_parts) -> str:
    """A short, stable hash of the arguments identifying one cached fetch.

    Parameters
    ----------
    key_parts : object
        Anything with a stable ``repr`` - normally a tuple of the call's
        arguments. Note that ``repr`` is the identity here, so two arguments that
        are equal but print differently (``1`` and ``1.0``) key differently.

    Returns
    -------
    str
        The first 12 hex characters of the md5 of ``repr(key_parts)``.
    """
    return hashlib.md5(repr(key_parts).encode()).hexdigest()[:12]


def cached(name, key_parts, builder, refresh: bool = False, directory=None):
    """Return a cached object, building and storing it on a miss.

    Parameters
    ----------
    name : str
        Cache-file prefix, used to group and clear related entries (e.g.
        ``"usgs_measurements"``).
    key_parts : object
        The arguments identifying this particular fetch (see :func:`cache_key`).
    builder : callable
        Zero-argument callable producing the object to cache. Called only on a
        miss.
    refresh : bool, default False
        Ignore any stored copy and rebuild.
    directory : path-like, optional
        Where to store the pickle. Defaults to ``settings.CACHE_DIR``.

    Returns
    -------
    object
        Whatever ``builder`` returned (or the previously stored copy of it).
    """
    directory = Path(settings.CACHE_DIR if directory is None else directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{cache_key(key_parts)}.pkl"
    if path.exists() and not refresh:
        log.debug("cache hit %s", path.name)
        return pd.read_pickle(path)
    log.debug("cache miss %s - fetching", path.name)
    obj = builder()
    pd.to_pickle(obj, path)
    return obj


def clear(name=None, directory=None) -> int:
    """Delete cached pickles and report how many were removed.

    Parameters
    ----------
    name : str, optional
        Clear only the entries stored under this prefix. ``None`` clears
        everything in the cache directory.
    directory : path-like, optional
        Cache directory. Defaults to ``settings.CACHE_DIR``.

    Returns
    -------
    int
        Number of files deleted.
    """
    directory = Path(settings.CACHE_DIR if directory is None else directory)
    if not directory.exists():
        return 0
    pattern = f"{name}_*.pkl" if name else "*.pkl"
    removed = 0
    for path in directory.glob(pattern):
        path.unlink()
        removed += 1
    log.info("cleared %d cached file(s) matching %s", removed, pattern)
    return removed
