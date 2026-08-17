# rating_curve_dashboard

A self-contained Streamlit app for building and inspecting USGS stage-discharge
rating curves interactively: fetch live field measurements (or upload your
own), fit and compare segmented power-law models, and check whether a rating
is stable enough to trust.

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Deliberately standalone: it does not import `limnotech_rating_curves` itself,
since that would pull PyMC/PyTensor into the same process as this app's
matplotlib Agg backend (a known Windows conflict) and slow every refit down.
It reuses only the *approach* this package's Bayesian models were built to
validate — a segmented log-log power law — reimplemented here in plain
numpy/scipy so every fetch/upload can be refit in well under a second. See
`engine.py`'s module docstring for the model family and `app.py`'s for the
full dependency rationale.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit UI — data source (USGS fetch or upload), model fitting/comparison, stability diagnostic, stage↔discharge lookup, downloads |
| `engine.py` | The fitting engine — segmented power-law fit, cross-validated model selection, residual-vs-time diagnostic. No Streamlit dependency; usable standalone |
| `usgs_fetch.py` | Live NWIS access via `dataretrieval` — site lookup, available data types, field measurements / daily / instantaneous fetch |
| `requirements.txt` | Everything needed to run the app; no compiler, no GPU, no `pymc` |

## What it does

| Feature | Notes |
| --- | --- |
| Data source | Fetch any USGS site directly (site number, auto-detected data types, date range bounded to the real period of record), or upload your own CSV/Excel (auto-detects stage/discharge/date columns) |
| Model fitting | Fits 1-, 2-, and 3-segment power laws (0, 1, or 2 breakpoints), cross-validates each, and auto-selects the simplest one within a configurable margin of the best cross-validated fit. Any candidate can be viewed instead via the model selector |
| Stability diagnostic | Plots percent residual vs. time — a trend or step (not random scatter) means the gage's physical rating shifted partway through the fitted record, which is usually a bigger accuracy lever than the curve family; see `engine.residual_trend` |
| Output | Equation, stage↔discharge lookup (either direction), full lookup table, and CSV/JSON/PNG downloads |

## Relationship to this package

`limnotech_rating_curves` is the Bayesian/PyMC reference implementation this
dashboard's power-law approach was validated against; this dashboard is the
lightweight, fast-refit tool built on that validated approach for interactive
use. It is not registered as an installable subpackage — `pyproject.toml`'s
`[tool.setuptools.packages.find]` only picks up directories with an
`__init__.py`, which this one deliberately omits — so run it directly with
`streamlit run app.py` as shown above.
