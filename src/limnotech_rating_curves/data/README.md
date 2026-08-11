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
| `station_series(station, variable="stage")` | One variable's timeseries |
| `station_coordinates(stations)` | Coordinates, for the map |
| `installed()`, `require()` | Is the optional `fb_pagaia` client available |

## `magl.py` — the MAGL monitoring network

Site registry, co-location, control-point survey, and the discharge exports.

| Name | Purpose |
| --- | --- |
| `sensor_sample(sensor)` | MAGL discharge against MAGL stage — the rating you would use |
| `colocated_sample(station, gage=None)` | USGS field discharge against the co-located MAGL sensor's stage |
| `list_sites()`, `list_sensors()`, `site_directory()` | What can be fitted, and with which loader |
| `discharge_measurements()`, `discharge_report()` | The manual discharge exports, cleaned |
| `site_registry()`, `clusters()`, `station_cluster`, `gage_cluster`, `gages_in_cluster`, `all_gages` | The registry in [`magl_sites.yaml`](magl_sites.yaml) |
| `colocated_gage`, `colocated_station`, `colocated_pairs` | Station ↔ gage pairing |
| `water_surface_elevation_ft(station)` | NAVD88 water surface for one station |
| `distance_record_ft(station)` | Merged distance-to-surface record (spreadsheet + pagaia) |
| `spreadsheet_distance_ft`, `pagaia_distance_ft` | Either source alone |
| `control_point_elevations()`, `sensor_moves()`, `reference_elevation(station, index)` | The survey and its changes over time |
| `control_point_offsets()`, `control_point_offset(station)` | Gap between surveyed control point and the sensor's own datum |
| `stage_at_times(station, times)` | Stage at the instants discharge was measured |
| `rating_curve_sheet(path)`, `flow_sheet_sample(path)` | Measurements out of a flow workbook |
| `pagaia_session()`, `pagaia_stations(stations)` | The network's pagaia connection |
| `cross_validate(sample)` | Leave-one-out, because MAGL samples are tiny |

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
