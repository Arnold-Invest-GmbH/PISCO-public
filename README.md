# PISCO paper reproduction

The repository contains the current manuscript, numerical source, dependency
locks, desktop interface and verification tools. The current dataset is supplied
with every downloadable application and separately as
[PISCO-results.zip](https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/download/r2-2026-11-13/PISCO-results.zip).

## Start

1. Download the ZIP matching your computer from the [release](https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/tag/r2-2026-11-13):

   | System | Download | Open after extraction |
   | --- | --- | --- |
   | Windows x64 | [PISCO-windows-x86_64.zip](https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/download/r2-2026-11-13/PISCO-windows-x86_64.zip) | `PISCO.exe` |
   | macOS 14+, Apple Silicon | [PISCO-macos-arm64.zip](https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/download/r2-2026-11-13/PISCO-macos-arm64.zip) | `PISCO.app` |
   | Linux x86-64, glibc 2.36+, X11/XWayland | [PISCO-linux-x86_64.zip](https://github.com/Arnold-Invest-GmbH/PISCO-public/releases/download/r2-2026-11-13/PISCO-linux-x86_64.zip) | `PISCO` |

2. Extract the complete ZIP and open the program listed above.
3. Choose an output folder and select **Check paper results**.

Every application ZIP includes the same saved data, its own Python runtime,
numerical libraries and figure tools. Normal use requires no Python installation,
Docker, GPU or internet. Use a filesystem that permits execution. The downloads
are unsigned; macOS applications are not Apple-notarized. Your system may ask
for approval before opening a downloaded program. Use normal operating-system
approval after checking the published checksum; do not disable system protection.

Windows and macOS packages passed automated tests on their respective operating
systems; Linux was tested locally and in isolated Debian 12 userland.
See [validation](portable/VALIDATION.md), [platforms](portable/BUILD_MATRIX.md)
and the release's per-platform receipts for the exact hosts and scope.
Intel Macs and Windows ARM are not covered by these packages.

The interface provides installation checks, saved-data verification, a short
calculation, a complete calculation, progress, stop and resume. A complete
calculation can take days. Resume requires the same environment and worker
count. Saved-result verification does not start an optimisation. Allow about
8 GiB free space for extraction and verification; new full calculations need
additional space and memory. Outputs and logs go to the selected folder. Initial integrity checks can take
several minutes on Windows; progress remains visible in the window.

Command-line saved-data checking, from the extracted application directory:

```text
# Windows PowerShell
.\PISCO.exe --action verify --workspace C:\PISCO-results
# macOS Terminal
./PISCO.app/Contents/MacOS/PISCO --action verify --workspace ~/PISCO-results
# Linux terminal
./PISCO --action verify --workspace ~/PISCO-results
```

Use `--action smoke --workers 4` for a short computation or
`--action full --workers 12` for the complete matrix. The window provides the
same actions. Command-line use is optional.

## Run the Python scripts directly

The [script guide](code/README.md) describes the alternative terminal route for
Linux, Windows and macOS: use the included Python runtime or create a virtual
environment, install the locked dependencies, then run `reproduce_all.py`
directly. It gives complete commands for a short test, the full matrix,
continuation, rebuilding outputs and checking the saved dataset.

`reproduce_all.py` is the entry point. It imports `solvers.py`, which contains
the numerical solver classes; that library does not need a separate launch.
The same solver-class and central-driver structure is used by the application.

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
