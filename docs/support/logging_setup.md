# limnotech_rating_curves.support.logging_setup

Logging configuration for the package.

Every module logs through the standard library
(`log = logging.getLogger(__name__)`) and nothing is printed. A library that prints
cannot be quietened or redirected to a file, and it clutters a notebook. Fitting a
rating is slow enough that progress messages are useful, so they are emitted at INFO
and the caller decides whether to display them.

## What happens on import

`import limnotech_rating_curves` calls `auto_configure()` once, which:

- attaches a console handler to this package's logger at INFO — and only to this
  package's logger, so an application with its own logging is left alone;
- silences the sampler loggers (`pymc`, `pymc.sampling`, `arviz`, `pytensor`,
  `nutpie`);
- ignores warnings raised inside the sampling stack and the array libraries under it
  (`NOISY_MODULES`). Warnings from your own code still print.

So the usual session needs no setup at all:

```
import limnotech_rating_curves as lrc
```

Set the `LRC_NO_AUTO_LOGGING` environment variable to import without touching logging
or warning filters at all.

## Changing the defaults

```
lrc.support.logging_setup.configure(verbose=True)     # DEBUG as well
lrc.support.logging_setup.configure(quiet=True)       # warnings and errors only
lrc.support.logging_setup.unsilence_samplers()        # let PyMC report on itself
lrc.support.logging_setup.unsilence_warnings()        # show every warning again
```

`unsilence_samplers` clears the level on those loggers so they inherit the root
level; pair it with `configure(verbose=True)` to see everything a fit does.

The command line calls `configure` from its `-v` and `-q` flags.
