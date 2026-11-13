# PISCO paper reproduction

The repository contains the current manuscript, numerical source, dependency
locks, desktop interface and verification tools. The current dataset is supplied
with the downloadable application.

## Start

1. Download [PISCO-linux-x86_64.zip](https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/download/r2-2026-11-13/PISCO-linux-x86_64.zip) from the [release](https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/tag/r2-2026-11-13).
2. Extract the ZIP and open **PISCO** inside the extracted directory.
3. Choose an output folder and select **Check paper results**.

The ZIP includes the saved data, Python runtime, numerical libraries and figure
tools. Normal use requires no Python installation, Docker, GPU or internet.
It targets **Linux x86-64, glibc 2.36 or later**, with X11/XWayland for the window.
Use a filesystem that permits execution. The release is unsigned; SHA-256
checksums identify the supplied files. See `portable/VALIDATION.md` for the
actual tests and limits. Other operating systems have not been validated.

The interface provides installation checks, saved-data verification, a short
calculation, a complete calculation, progress, stop and resume. A complete
calculation can take days. Resume requires the same environment and worker
count. Saved-result verification does not start an optimisation. Allow about
4 GiB free space for extraction and verification; new full calculations need
additional space and memory. Outputs and logs go to the selected folder.

Command-line use from the extracted application directory:

```text
./PISCO --action verify --workspace /path/to/output
./PISCO --action smoke --workers 4 --workspace /path/to/output
./PISCO --action full --workers 12 --workspace /path/to/output
```

## Scientific data and source

`PISCO_en.pdf` is the current manuscript. `code/solvers.py` implements the
solvers; `code/reproduce_all.py --all` produces the complete experiment matrix
and its documentation. These are the exact numerical sources used for the
saved run; their SHA-256 values are recorded in `release.json` and the dataset.
They retain their embedded reference definitions to preserve source identity.

The included `results.zip` contains all current per-run documentation: **526
rows across 66 configurations**, representing 520 solver runs and six reused
sensitivity rows. It includes FEM events, designs, training/test diagnostics,
environment metadata, 92 current numerical summaries and 40 statistical
comparisons. `documentation/REPORT.md` explains the files. Execution checkpoints
and redundant caches are excluded; the data export is not a resume checkpoint.

Current measurements and numerical summary fields are unchanged. The public
summary tables contain current results only. `verification/check_results.py`
independently checks arithmetic, budgets, pairings, source hashes and statistical
families from the saved evidence. With the recorded dependencies installed:

```text
python verification/check_results.py --results /unpacked/results \
  --source /path/to/this/repository --output /new/verification-report
python code/reproduce_all.py --all --workers 12 --output /new/full-run
```

Continuous and binary compliance must be interpreted separately. E2 statistical
tests are exploratory; confidence intervals are marginal and Holm corrections
apply within the stated families. The verifier reports test-sensitive decisions
and budget/trajectory limitations. Dependency locks and worker-count checks do
not establish bitwise equality across processor architectures or library builds.
The original full-run environment is included in the data. The complete matrix
has not been rerun inside the bundled runtime.

## Maintainer build

See `portable/README.md` for the pinned build recipe and
`portable/acceptance.py` for the executable acceptance sequence.
`portable/public-source-files.json` is the explicit source export allowlist.
Only listed source files can enter a package. `LICENSE` and `THIRD_PARTY.md`
describe project and dependency licensing.
