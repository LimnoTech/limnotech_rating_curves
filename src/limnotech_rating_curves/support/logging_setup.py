import logging
import os
import warnings

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

#: The package's root logger name.
LOGGER_NAME = "limnotech_rating_curves"

#: The loggers a fit chatters through that are rarely what the caller is watching for.
SAMPLER_LOGGERS = ("pymc", "pymc.sampling", "arviz", "pytensor", "nutpie")

#: Modules whose warnings are noise from this package's point of view: the sampling
#: stack and the array libraries underneath it. A warning raised by the caller's own
#: code is not on this list and still prints.
NOISY_MODULES = ("pymc", "pytensor", "arviz", "nutpie", "ratingcurve",
                 "numpy", "scipy", "pandas", "openpyxl")

#: Set this environment variable to any value to import the package without touching
#: logging or warning filters at all.
OPT_OUT_VARIABLE = "LRC_NO_AUTO_LOGGING"

_auto_configured = False


def configure(verbose: bool = False, quiet: bool = False, *,
              package_only: bool = False) -> logging.Logger:
    """Set up console logging.

    Parameters
    ----------
    verbose : bool, default False
        Log at DEBUG - every cache hit, every skipped model, every fold.
    quiet : bool, default False
        Log at WARNING only. Wins if both are set.
    package_only : bool, default False
        Configure only this package's logger, leaving the root logger alone. Use
        this inside an application that has its own logging set up, so the two do
        not fight.

    Returns
    -------
    logging.Logger
        The logger that was configured.
    """
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    if package_only:
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(handler)
        logger.propagate = False
        return logger
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATE_FORMAT)
    logging.getLogger().setLevel(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    return logger


def silence_samplers() -> None:
    """Quieten PyMC's and ArviZ's own loggers.

    A fit emits sampler chatter that is rarely what you are watching for. This drops
    those loggers to WARNING without touching this package's own messages. It is the
    default, applied at import by :func:`auto_configure`.
    """
    for name in SAMPLER_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def unsilence_samplers() -> None:
    """Let the sampler loggers speak again.

    Undoes :func:`silence_samplers` by clearing the level on each of
    :data:`SAMPLER_LOGGERS`, so they inherit whatever the root logger is set to. Call
    this when a fit is behaving oddly and you want to watch the sampler itself.
    """
    for name in SAMPLER_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)


def silence_warnings() -> None:
    """Ignore warnings raised inside the sampling stack.

    Only the packages in :data:`NOISY_MODULES` are filtered - the filter's ``module`` is
    matched against the name of the module the warning is raised in, so a package name
    covers its submodules. Warnings raised on your own code's behalf, which is what
    ``stacklevel`` in a library call means, still reach you.

    This is the default, applied at import by :func:`auto_configure`.
    """
    for module in NOISY_MODULES:
        warnings.filterwarnings("ignore", module=module)


def unsilence_warnings() -> None:
    """Show every warning again, by resetting the warning filters."""
    warnings.resetwarnings()


def auto_configure() -> None:
    """Set the defaults a session almost always wants, once, at import.

    Console logging at INFO on this package's own logger, sampler chatter silenced, and
    warnings from the sampling stack ignored. Only this package's logger gets a handler,
    so an application with its own logging is left alone.

    Runs once per process and does nothing if the ``LRC_NO_AUTO_LOGGING`` environment
    variable is set. To change any part of it afterwards, call :func:`configure`,
    :func:`unsilence_samplers` or :func:`unsilence_warnings`.
    """
    global _auto_configured
    if _auto_configured or os.environ.get(OPT_OUT_VARIABLE):
        return
    configure(package_only=True)
    silence_samplers()
    silence_warnings()
    _auto_configured = True
