# limnotech_rating_curves

Bayesian stage-discharge rating curves: fit a rating to paired stage and discharge measurements, compare model families, and run cross-validation.

```python
import limnotech_rating_curves as lrc

rating = lrc.fit_rating(measurements)
rating.predict(5.0) # discharge at stage = 5 ft
rating.plot()
```

Units are feet (stage) and cfs (discharge) throughout. See the [examples/getting_started.ipynb](./examples/getting_started.ipynb) for a walkthrough of basic functionality.

## Setup

Install [Miniconda](https://www.anaconda.com/download/success), then from this folder in the Anaconda PowerShell Prompt:

```powershell
git clone https://github.com/LimnoTech/limnotech_rating_curves
cd limnotech_rating_curves
conda env create -f environment.yml
conda activate rating_curves
pip install -e .
pytest tests/test_environment.py -q
```

That installs everything except integration with `pagaia`, which can optionally be included:

```powershell
pip install -e ".[pagaia]"
```

You may have to request access to the private LimnoTech `pagaia` repostitory. You will also have to set `PAGAIA_AUTH_TOKEN` in your environment variables to your Freeboard API Key.

## VS Code

`Ctrl+Shift+P` => **Python: Select Interpreter** => `rating_curves`. If conda envs don't appear, enter the path directly:

```
C:\Users\<you>\AppData\Local\miniconda3\envs\rating_curves\python.exe
```

For terminals that activate conda themselves, run `conda init powershell` once; if the profile then won't load, it's the execution policy: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`.

## Running a notebook against your own kernel

You can optional start a jupyter kernel in the terminal and point VS Code at it.

```powershell
conda activate rating_curves
jupyter notebook --no-browser
```

It prints a URL with a token, like `http://localhost:8888/lab?token=<long hex>`. Copy that whole line, then in VS Code: `Ctrl+Shift+P` => **Notebook: Select Notebook Kernel** => **Select Another Kernel** => **Existing Jupyter Server** => paste the URL => pick the `Python 3 (ipykernel)` kernel.

`Ctrl+C` twice in the terminal to stop the kernel.

## Next

[`examples/getting_started.ipynb`](examples/getting_started.ipynb) — fitting, comparing
models, cross-validation, saving a fit. Runs on public USGS data.
[`examples/extras.ipynb`](examples/extras.ipynb) — datums, zero flow, diagnostics, the
map, many sites at once. [`docs/`](docs/) — one page per module. `rating-curves --help`
— the CLI. [`settings.py`](src/limnotech_rating_curves/settings.py) — every tunable
value, listed with its default and each overridable from the environment as `LRC_` +
its name; copy [`.env.example`](.env.example) to `.env` and see
[`examples/settings_from_env.py`](examples/settings_from_env.py).
