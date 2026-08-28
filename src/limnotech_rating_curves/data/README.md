# data

Getting measurements in, putting stage on a common vertical reference, and
estimating the stage of zero flow. Every loader returns a
[`Sample`](../core.py) or a tidy DataFrame in feet and cfs.

```python
from limnotech_rating_curves import data

data.gage_sample("04166500")            # USGS field measurements
data.sensor_sample("SBR-09")            # MAGL discharge vs MAGL stage
data.colocated_sample("SBR-09")         # USGS discharge vs MAGL stage
data.station_sample(station, discharge) # any pagaia station + your discharge
```

## `noaa.py` — NOAA / NWS published ratings

The NWS keeps its own rating at each forecast point, separate from the USGS one for
the same gage. The NWPS API cannot be searched by USGS site number, so the reverse
lookup goes through NOAA's HADS crosswalk.

| Name | Purpose |
| --- | --- |
| `lid_for_usgs_site(site)` | The NWS location id for a USGS site, via the HADS crosswalk |
| `published_rating(lid)` | The NWS rating as a tidy `stage_ft` / `discharge_cfs` curve |
| `published_reference(lid, sample)` | That rating packaged as a comparison curve |
| `gauge_info(lid)` | Gauge metadata, including the USGS site it corresponds to |
| `discharge_at(curve, stage)` | Read the table; NaN outside it, never extrapolated |
| `datum_agreement(lid)` | Whether NWS and USGS are on the same stage axis - reported, never silently corrected |

## `usgs.py` — public USGS data

| Name | Purpose |
| --- | --- |
| `gage_sample(site)` | Field measurements as a fittable `Sample` |
| `measurements(site)` | The same pairs as a DataFrame |
| `published_rating(site)` | The gage's published rating as a tidy curve |
| `published_reference(site, sample)` | That rating packaged as a comparison curve |
| `rating_table(site, file_type="exsa")` | A raw published rating file |
| `segment_breakpoints(site)` | Stages where the published rating changes log-log slope |
| `rating_deviation(site)` | Every field measurement against the published rating |
| `continuous_record(site, start, end)` | Paired gage height and discharge from the continuous record |
| `continuous_stage`, `continuous_discharge` | One variable at a time |
| `rating_stability(record)` | Within-year vs between-year spread at each stage |
| `site_info(sites)`, `coordinates(sites)`, `gage_datum(site)` | Site metadata |
| `normalize_site_id(site)` | Site ids are zero-padded strings, never ints |

## `pagaia.py` — the ODM2 sensor database

| Name | Purpose |
| --- | --- |
| `station_sample(station, discharge)` | Sensor stage matched to your discharge measurements |
| `raw_station_series(station, variable="stage")` | One variable's timeseries, unconverted, with the units the database claims |
| `station_name(station)` | The station's name, whichever attribute it carries it in |
| `station_coordinates(stations)` | Coordinates, for the map |
| `installed()`, `require()` | Is the optional `fb_pagaia` client available |

Turning a pagaia reading into feet is three steps, written out by the caller so that
the units correction is never silent:

```python
raw    = pagaia.raw_station_series(station, "distance", start, end)
meters = pagaia_corrections.to_meters(raw, pagaia.station_name(station))
feet   = datum.in_units(meters, "ft")
```

## `pagaia_corrections.py` — units the database gets wrong

pagaia reports every water-level variable as `millimeter`. The values are really
meters, except on the five Geolux stations, whose readings were millimeters until
2026-08-19 14:30 UTC. Both corrections are applied here, by the caller, never by
default inside a fetch.

| Name | Purpose |
| --- | --- |
| `to_meters(raw, station)` | Both corrections, returning `(values, "m")` |
| `geolux_stations()`, `is_geolux(station)` | Which stations report millimeters |

Temporary — delete this module once the database is corrected. `to_meters` raises as
soon as pagaia stops claiming millimeters, so it cannot keep applying unnoticed. A
cached reading carries the units claimed when it was fetched, so that check fires only
after `refresh=True` or `cache.clear("pagaia_series_units")`.

## `magl.py` — the MAGL monitoring network

Site registry, co-location, control-point survey, and the discharge exports.

| Name | Purpose |
| --- | --- |
| `sensor_sample(sensor)` | MAGL discharge against MAGL stage — the rating you would use |
| `colocated_sample(station, gage=None)` | USGS field discharge against the co-located MAGL sensor's stage |
| `list_sites()`, `list_sensors()`, `site_directory()` | What can be fitted, and with which loader |
| `discharge_measurements()`, `discharge_report()` | The manual discharge exports, cleaned |
| `site_registry()`, `clusters()`, `station_cluster`, `gage_cluster`, `gages_in_cluster`, `all_gages` | The registry, a directory of CSVs - see [`data/magl/sites/`](../../../data/magl/sites/README.md) |
| `colocated_gage`, `colocated_station`, `colocated_pairs` | Station ↔ gage pairing |
| `water_surface_elevation_ft(station)` | NAVD88 water surface for one station |
| `distance_record_ft(station)` | Merged distance-to-surface record (spreadsheet + pagaia) |
| `pagaia_distance_ft(station)` | The database source alone (the spreadsheet source is in `magl_spreadsheets.py`) |
| `control_point_elevations()`, `sensor_moves()`, `reference_elevation(station, index)` | The survey and its changes over time |
| `sensor_elevation(station, index)` | Elevation the distance readings are measured from; subtract a distance to get water-surface elevation |
| `stage_at_times(station, times)` | Stage at the instants discharge was measured |
| `rating_curve_sheet(path)`, `flow_sheet_sample(path)` | Measurements out of a flow workbook |
| `pagaia_session()`, `pagaia_stations(stations)` | The network's pagaia connection |
| `cross_validate(sample)` | Leave-one-out |

## `magl_spreadsheets.py` — the historical stage record

The scraped Excel timeseries (`magl_spreadsheet_timeseries.pkl`): one frame per
station, in feet. Still the only source for the earliest part of every record, and
the only place the control-point offset can be measured, because these frames carry
both the raw distance reading and the spreadsheet's own derived elevation.

| Name | Purpose |
| --- | --- |
| `frames()` | `{station: DataFrame}` out of the pickle, memoized |
| `distance_ft(station)` | Distance-to-surface from the spreadsheet, in feet, QA/QC applied |
| `control_point_offsets()`, `control_point_offset(station)` | Gap between the surveyed control point and the sensor's own datum |
| `clean_series(series)` | tz-naive, sorted, de-duplicated float series |


## `magl_workbook.py` — the field workbooks' own ratings

| Name | Purpose |
| --- | --- |
| `stored_equation(sensor)` | The rating a workbook already carries, as a `WorkbookRating` |
| `WorkbookRating` | `.discharge`, `.discharge_at_stage`, `.equation`, `.curve`, `.ok` |
| `reference(sensor, sample)` | That rating packaged the way the map draws references |
| `status()`, `fittable_sensors()`, `workbook_paths()` | Which workbooks hold what |
| `rating_curve_sheet(path)`, `summary_sheet_name(path)`, `datum_tie(...)` | Reading the sheets |

## `datum.py` — vertical references

`StageDatum` is the one type every loader accepts as `stage_datum=`.

| Constructor | Meaning |
| --- | --- |
| `StageDatum.none()` | Stage is already gage height |
| `StageDatum.constant(elevation_ft)` | Subtract one fixed elevation |
| `StageDatum.usgs_gage_datum(alt_va)` | The datum a USGS gage's stage is measured from |
| `StageDatum.lowest_observed()` | Reference to a little below where flow reaches zero |
| `StageDatum.timeseries(series)` | A reference that changes over time |

Also: `to_gage_height(stage, datum)`, `distance_to_stage(distance,
reference_elevation)`, `stepwise_reference_elevation(index, base_elevation_ft,
changes)`, `reference_at_times(series, times)`, `stage_at_times(stage_series,
times)`, `as_stage_datum(datum)`, `ConvertedStage.to_elevation`.

## `zero_flow.py` — stage of zero flow

| Name | Purpose |
| --- | --- |
| `estimate_zero_flow(data)` | Johnson's method, falling back rather than failing |
| `johnson_offset(data)` | Johnson's three-point method, strict |
| `ZeroFlowEstimate` | The estimate and the evidence for it |
| `johnson_three_points`, `gage_height_of_discharge` | The points and the smooth median curve they are read off |
| `loglog_r2(stage, discharge, zero_flow)` | Straightness of log Q vs log(h − e) |
| `plot_johnson(data)` | Show what the method did |
