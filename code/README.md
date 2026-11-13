# Run the Python scripts directly

The desktop application and the command-line route use the same numerical
implementation. Choose either the supplied Python runtime or your own virtual
environment, then invoke `reproduce_all.py` directly.

## What the two Python files do

| File | Responsibility |
| --- | --- |
| `code/reproduce_all.py` | Command-line entry point. Selects experiments and seeds, starts/resumes the solvers, writes statistics, tables, figures and run documentation, and verifies the experiment matrix. |
| `code/solvers.py` | Numerical library: problem definitions, configuration classes, `BaseSolver` and concrete solvers, FEM, training, search, caching and `PublicationExperimentSuite`. Imported automatically by the driver. |
| `code/paper_support/` | Editable Typst diagrams needed by the complete figure build. |
| `verification/check_results.py` | Independent check of the released current data, budgets and statistics, without new optimisation. |

Keep `reproduce_all.py` and `solvers.py` together in `code/`. Start the driver;
`python code/solvers.py` does not run an experiment or generate the paper results.
Within the library, a solver's `solve(seed=...)` returns a `SolverResult` with
the design, compliance, FEM counts, trajectory, timings and evidence.
`PublicationExperimentSuite` coordinates the benchmark families. Calling these
classes directly is useful for development, but the driver supplies the complete
paper protocol and reporting steps.

## A. Use the Python runtime included in an application ZIP

Download and extract the application ZIP for your operating system. Open a
terminal in the extracted directory containing `PISCO`, `PISCO.exe` or
`PISCO.app`. No Python installation is needed for this route.

**Linux — terminal:**

```sh
PISCO_BUNDLE="$PWD"
PISCO_PY="$PISCO_BUNDLE/runtime/bin/python3"
PISCO_RUNS="$HOME/PISCO-script-results"
```

**macOS on Apple Silicon — terminal:**

```sh
PISCO_BUNDLE="$PWD/PISCO.app/Contents/Resources"
PISCO_PY="$PISCO_BUNDLE/runtime/bin/python3"
PISCO_RUNS="$HOME/PISCO-script-results"
```

For either Linux or macOS, then run:

```sh
export PATH="$PISCO_BUNDLE/tools/bin:$PATH"
export TYPST_PACKAGE_CACHE_PATH="$PISCO_BUNDLE/tools/cache/typst/packages"
export TYPST_FONT_PATHS="$PISCO_BUNDLE/tools/fonts"
export FONTCONFIG_FILE="$PISCO_BUNDLE/tools/fonts.conf"
export MPLCONFIGDIR="$PISCO_RUNS/cache/matplotlib"
cd "$PISCO_BUNDLE/source"
"$PISCO_PY" --version
"$PISCO_PY" -B -X utf8 code/reproduce_all.py --list
```

**Windows x64 — PowerShell:**

```powershell
$PISCO_BUNDLE = (Get-Location).Path
$PISCO_PY = Join-Path $PISCO_BUNDLE 'runtime/python.exe'
$PISCO_RUNS = Join-Path $env:USERPROFILE 'PISCO-script-results'
$env:PATH = "$PISCO_BUNDLE\tools\bin;$env:PATH"
$env:TYPST_PACKAGE_CACHE_PATH = "$PISCO_BUNDLE\tools\cache\typst\packages"
$env:TYPST_FONT_PATHS = "$PISCO_BUNDLE\tools\fonts"
$env:FONTCONFIG_FILE = "$PISCO_BUNDLE\tools\fonts.conf"
$env:MPLCONFIGDIR = "$PISCO_RUNS\cache\matplotlib"
Set-Location "$PISCO_BUNDLE\source"
& $PISCO_PY --version
& $PISCO_PY -B -X utf8 code/reproduce_all.py --list
```

The following commands now run from the package's `source/` directory. Output
is kept in `PISCO-script-results` in your home directory, outside the application.
Keep this terminal open so the interpreter and tool settings remain available.

## B. Use a separate Python environment and repository checkout

Alternatively, obtain the source repository:

```text
git clone https://github.com/Arnold-Invest-GmbH/PISCO-public.git
cd PISCO-public
```

Install **CPython 3.14 for the matching CPU architecture** first; the released
packages use 3.14.8. The locked wheels below target CPython 3.14 and their stated
operating system/architecture. `code/requirements.txt` lists the main numerical
versions; the lock files additionally fix transitive dependencies and wheel
hashes. The Windows/macOS locks also include the desktop and rendering libraries,
which can remain installed when only using the numerical scripts.

**Linux x86-64:**

```sh
python3.14 -m venv .venv
PISCO_PY="$PWD/.venv/bin/python"
PISCO_RUNS="$HOME/PISCO-script-results"
"$PISCO_PY" -m pip install --only-binary=:all: --require-hashes -r container/requirements.lock
"$PISCO_PY" -m pip check
```

**macOS 14+, Apple Silicon:**

```sh
python3.14 -m venv .venv
PISCO_PY="$PWD/.venv/bin/python"
PISCO_RUNS="$HOME/PISCO-script-results"
"$PISCO_PY" -m pip install --only-binary=:all: --require-hashes -r portable/packages-macos-arm64.lock
"$PISCO_PY" -m pip check
```

**Windows x64 — with CPython 3.14 available as `python`:**

```powershell
python -m venv .venv
$PISCO_PY = Join-Path (Get-Location).Path '.venv/Scripts/python.exe'
$PISCO_RUNS = Join-Path $env:USERPROFILE 'PISCO-script-results'
& $PISCO_PY -m pip install --only-binary=:all: --require-hashes -r portable/packages-windows-x86_64.lock
& $PISCO_PY -m pip check
```

These commands select the environment's interpreter explicitly, so running an
activation script is unnecessary. See the [Python venv documentation](https://docs.python.org/3.14/library/venv.html).
Installation downloads packages; it is separate from running the calculations.

Before a **complete run**, also provide `typst` and `pdftoppm` on `PATH`.
You can reuse the matching application ZIP's `tools/` and preloaded Typst
packages: set `PISCO_BUNDLE` to its absolute bundle path and apply the tool and
font environment settings from route A, while retaining your virtual-environment
`PISCO_PY` and staying in the repository checkout. Alternatively, install
Typst 0.15.1 and Poppler's `pdftoppm` with your operating system's package tools.
A separately installed Typst may download the packages declared by the templates
on first use; the supplied tool cache supports offline use. Check the tools
before starting a long computation:

```text
typst --version
pdftoppm -v
```

The short test and saved-data verifier do not need these external figure tools.

## Run a short test, then the complete experiment matrix

**Linux/macOS:**

```sh
"$PISCO_PY" -B -X utf8 code/reproduce_all.py --help
"$PISCO_PY" -B -X utf8 code/reproduce_all.py --list
"$PISCO_PY" -B -X utf8 -u code/reproduce_all.py --smoke --workers 4 --fresh --output "$PISCO_RUNS/smoke"
"$PISCO_PY" -B -X utf8 -u code/reproduce_all.py --all --workers 12 --progress-interval 30 --fresh --output "$PISCO_RUNS/full"
```

**Windows — PowerShell:**

```powershell
& $PISCO_PY -B -X utf8 code/reproduce_all.py --help
& $PISCO_PY -B -X utf8 code/reproduce_all.py --list
& $PISCO_PY -B -X utf8 -u code/reproduce_all.py --smoke --workers 4 --fresh --output "$PISCO_RUNS\smoke"
& $PISCO_PY -B -X utf8 -u code/reproduce_all.py --all --workers 12 --progress-interval 30 --fresh --output "$PISCO_RUNS\full"
```

Run these lines individually. The last line starts the full calculation and can
take days. The short test creates 16 reduced-budget result rows and reports
success in `smoke_verification.json`; it is not a publication run.
`--all` includes the default seeds, controls, ablations and budget comparisons,
then builds the documentation and runs the full-matrix verification.
`--workers 12` selects concurrent work packets; native numerical kernels within
a packet use one thread. Choose a smaller positive number if needed.
`--progress-interval 30` controls the status heartbeat; `-u` flushes Python output.

`--fresh` requires an empty/new output directory and protects existing evidence.
For another independent calculation, choose a new output directory. To **resume
an interrupted full run**, repeat the same full command **without `--fresh`**:

```sh
# Linux/macOS
"$PISCO_PY" -B -X utf8 -u code/reproduce_all.py --all --workers 12 --progress-interval 30 --output "$PISCO_RUNS/full"
```

```powershell
# Windows
& $PISCO_PY -B -X utf8 -u code/reproduce_all.py --all --workers 12 --progress-interval 30 --output "$PISCO_RUNS\full"
```

Use Ctrl+C once to interrupt and let the process finish stopping. Keep the same
sources, environment, worker count and experiment options when resuming.
Completed runs are reused; available solver checkpoints are retained. A task
interrupted before a saved checkpoint may need to be recomputed. An environment
or source mismatch is reported instead of silently mixing results. Run only one
direct driver process per output directory; the GUI's workspace lock is not a
feature of the raw driver.

## Further driver operations

Append these arguments to the same interpreter prefix and `code/reproduce_all.py`:

| Purpose | Arguments |
| --- | --- |
| One example with its default seed set | `--example example1 --output <new-output-directory>` |
| Selected-seed diagnostic run | `--example example1 --seeds 7 --output <new-output-directory>` |
| Verify a complete saved run | `--verify --output <full-run-directory>` |
| Rebuild figures, statistics and documentation from your saved run | `--build --output <full-run-directory>` |

`example1` is the ellipse, `example2` the L-bracket and `example3` the fin.
A selected example/seed set is a partial run and does not satisfy full-matrix
verification. Changing `--training-policy`, `--seeds` or `--without-controls`
changes the protocol or coverage; use the unmodified `--all` command above for
the paper matrix.

`--build` needs the locally generated `publication_results.pkl` and writes output
files. It reuses the solver results, but some illustrations perform additional
FEM evaluations, recorded in `figure_generation_evidence.json`. Use it on your
own completed run, not on the released data-only export. Check
`figure_generation_error.json`: if present, figure generation failed even if
numerical completeness checks passed. Install/fix the required tools and rerun
`--build`; a new optimisation is not needed for that repair.

## Check the published results without recomputing them

Download [PISCO-results.zip](https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/download/r2-2026-11-13/PISCO-results.zip)
and extract it to a separate data directory. The selected input directory must
contain `documentation/` directly. Set its absolute path in place of
`/path/to/data` or `C:/path/to/data` below. From the repository/package `source/`
directory, run:

```sh
# Linux/macOS
"$PISCO_PY" -B -X utf8 -u verification/check_results.py --results /path/to/data --source "$PWD" --output "$PISCO_RUNS/saved-data-check"
```

```powershell
# Windows
& $PISCO_PY -B -X utf8 -u verification/check_results.py --results C:/path/to/data --source . --output "$PISCO_RUNS\saved-data-check"
```

Keep the report directory outside both the source checkout and the data directory.
The check covers 526 rows, 4,248 artifact hashes, 92 numerical summaries and
40 statistical comparisons. It writes `verification.json` and detailed reports.
The data-only export contains complete current per-run documentation, but no
resumable execution checkpoints or `publication_results.pkl`. It supports this
verification route, not restarting the original computation or `--build`.

## Where to read the output

| File/directory inside your output directory | Contents |
| --- | --- |
| `progress.json`, `progress.jsonl` | Current status and progress events. |
| `experiment_plan.json`, `reproduction_environment.json` | Requested matrix, source/execution settings and environment fingerprint. |
| `documentation/REPORT.html`, `documentation/REPORT.md` | Overview and links to run evidence. |
| `documentation/completeness.json`, `documentation/run_index.json` | Matrix completeness and individual result rows. |
| `documentation/publication_results.csv`, `.tex` | Numerical summaries and table data. |
| `documentation/manuscript/` | Detailed statistics, comparisons and supporting numerical summaries. |
| `documentation/<family>/<method>/seed_<seed>/` | Configuration, training, FEM events, predictions, designs and diagnostics. |
| `figures/`, `figure_generation_evidence.json` | Generated publication figures and recorded figure-generation evaluations. |
| `runs/`, `publication_results.pkl` | Locally generated persisted results used for continuation and rebuilding. |

The scientific sources and measured publication data are unchanged. The platform
receipts identify the tested package environments. Matching dependencies and
worker-count checks do not guarantee bitwise equality between architectures,
operating systems or compiled-library builds. See the [validation scope](../portable/VALIDATION.md).
