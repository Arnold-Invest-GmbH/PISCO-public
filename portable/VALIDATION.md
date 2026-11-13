# Native package validation — 13 November 2026

## Linux

The current native Linux distribution passed the following local checks.
`validation/summary.json` records the concise machine-readable results;
`portable/acceptance.py` contains the executable acceptance sequence.

| Check | Result |
| --- | --- |
| Saved experiment data | 526 rows, 66 configurations, 4,248 artifact hashes, 92 numerical summaries and 40 statistical comparisons; 14,537 checks passed |
| Offline execution | Passed in isolated Debian 12 userland, with a read-only package, no external Python and networking disabled |
| Real short computations | Four and twelve workers; 16 result rows each; result/documentation verification passed |
| Worker comparison | All selected numerical fields exactly equal, including designs, FEM events, training/model hashes and predictions; elapsed times excluded |
| Stop and continue | Real short computation interrupted and continued successfully |
| Figures | All five supplied editable templates compiled and rasterised offline |
| Interface | Native desktop and offscreen start/action/shutdown passed |
| Launcher/backend tests | 15 passed, covering integrity, unsafe paths, missing/damaged archives, locking, existing-output protection and cancellation |

Tests used one AMD Ryzen AI 9 HX 370 laptop running CachyOS with glibc 2.44.
The additional Debian 12/glibc 2.36 userland shared that CPU and kernel.
It was not another physical machine or an independent editorial test.
The native desktop can emit host Fontconfig warnings; the tested action and
shutdown succeeded. Linux requires x86-64 and glibc 2.36 or later; the window
requires X11/XWayland. This local Linux check does not establish support on every x86-64 processor.

The package includes Python 3.14.8, CPU PyTorch 2.14.1+cpu, NumPy 2.5.4,
SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2 and Qt/PySide6 6.12.0.
Versions and component hashes are included. Fixed dependencies and short-run
worker agreement do not guarantee bitwise equality across architectures,
operating systems or compiled numerical-library builds.

No complete experiment matrix was rerun in the bundled runtime. Saved results
retain their original recorded environment. The original host computation took
about two days with twelve workers; that is not a speed guarantee for other
machines. Full-run peak memory has not been measured. Saved-data verification
checks evidence and arithmetic without rerunning FEM, training or optimisation.
The report distinguishes continuous/binary metrics, comparison budgets,
exploratory tests, test-sensitive conclusions and geometric limitations.

## Windows and macOS

The native workflow builds and tests Windows x64 on Windows Server 2025 and
macOS ARM64 on macOS 15. Published packages passed all of the following on their
respective target systems:

- Saved-data verification: 526 rows, 4,248 artifact hashes, 92 current numerical
  summaries, 40 statistical contrasts and 14,537 checks.
- Real four/twelve-worker short computations, 16 rows each, with exact equality
  of the selected numerical fields between worker counts on that test host.
- Native desktop and offscreen Qt window operation, real interruption/continuation, 15 backend tests
  and compilation/rasterisation of all five editable figure templates.
- Execution with external network access denied and external Python excluded.
- Final ZIP integrity and saved-data verification after extraction into a fresh
  path containing spaces and a non-ASCII character.

Each new ZIP contains `validation/summary.json` (inside the application resources
on macOS). The release also provides `release-windows-x86_64.json` and
`release-macos-arm64.json`, recording the build commit, CI run, actual host,
archive hash and test scope. `release.json` and `validation/summary.json` at the
repository root retain the original Linux package manifest/evidence;
`platforms.json` indexes all three deliveries.

These are automated maintainer tests, not independent editorial confirmation or
manual desktop testing on every supported OS version. No complete optimisation
matrix was recomputed, and results are not claimed bitwise equal between OSes.
Numerical sources and saved data have the same hashes across all three packages.
Windows/macOS use PDFium via pypdfium2 5.15.2 for figure rasterisation; Linux uses
Poppler. Rendering may differ without changing stored numerical results.

Completed target-system run: [43427817370](https://github.com/Arnold-Invest-GmbH/PISCO-public/actions/runs/43427817370).
The Windows/macOS receipts record build commit `bd8b94ca3e0581440d8cb0c05ce25f327814a1b7`.
