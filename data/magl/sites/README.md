# sites

The MAGL site registry, one CSV per kind of entry. This is the one place the set of
stations and gages is declared; `limnotech_rating_curves.data.magl.site_registry()`
reads these four files and everything else - cluster lookups, `--site` selection,
which sensors get listed - is derived from them. Add a site by adding a row.

A *cluster* is a stream-network group labeled by the station-name prefix, so
`SR-08_WL` belongs to cluster `SR`. Gages inherit a cluster from a co-located station
or from `standalone_gages.csv`.

| File | Columns | What it declares |
| --- | --- | --- |
| `clusters.csv` | `cluster`, `name` | Optional per-cluster description. Clusters are also inferred from the files below, so a cluster missing here still exists |
| `colocated_pairs.csv` | `station`, `gage` | A LimnoTech station (base name, no `_WL`) and the USGS gage at the same location, treated as immediately downstream of it |
| `standalone_gages.csv` | `gage`, `cluster` | USGS gages with no co-located station |
| `special_stations.csv` | `station`, `pickle`, `entry_key` | Stations read from the spreadsheet pickle rather than the pagaia API. `entry_key` indexes into that pickle; distances there are in feet. Paths are relative to the data directory |

## Gage ids are strings

Leading zeros are significant: `04176356` is an eight-digit id, `4176356` is a
different (nonexistent) seven-digit one. The loader reads every id as text, so the
files are correct as written - **but a spreadsheet editor will strip the zeros if you
open one of these and save it.** Edit them as text, or set the column to text before
saving. `site_registry()` warns about any id that has lost its leading zero.
