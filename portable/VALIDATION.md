# Linux package validation — 13 November 2026

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
requires X11/XWayland. Other processors and operating systems are unvalidated.

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
