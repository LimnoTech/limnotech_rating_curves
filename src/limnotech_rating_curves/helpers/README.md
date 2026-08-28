# support

Housekeeping: the on-disk fetch cache and logging setup.

## `cache.py` — pickle cache for slow fetches

Every loader in [`data/`](../data/) that hits a network takes `refresh=False` and
goes through this.

| Name | Purpose |
| --- | --- |
| `cached(name, key_parts, builder, refresh=False)` | Return a cached object, building and storing it on a miss |
| `cache_key(key_parts)` | A short, stable hash of the arguments identifying one fetch |
| `clear(name=None)` | Delete cached pickles and report how many went |

## `logging_setup.py` — console logging

`auto_configure()` runs at package import: console logging at INFO, sampler
chatter and sampling-stack warnings silenced. Set `LRC_NO_AUTO_LOGGING` to skip
it.

| Name | Purpose |
| --- | --- |
| `configure(verbose=False, quiet=False, package_only=False)` | Set up console logging yourself |
| `silence_samplers()` / `unsilence_samplers()` | Quieten PyMC's and ArviZ's loggers |
| `silence_warnings()` / `unsilence_warnings()` | Ignore warnings raised inside the sampling stack |
| `auto_configure()` | The defaults a session almost always wants, once, at import |

For long runs, attach a `FileHandler` — sampler progress does not survive being
piped.
