# Native desktop applications

Extract the matching platform ZIP, open `PISCO.exe` on Windows, `PISCO.app`
on macOS or `PISCO` on Linux, select an output folder and choose
**Check paper results**. The saved results are included. See
`START_HERE.html` and `VALIDATION.md` for actions and tested conditions.
No Python installation or Docker is required for recipients.

For direct Python execution without the interface, see the
[script guide](../code/README.md). Package building is a separate maintainer task.

## Additional native platform builds

`build_native.py` builds Windows x64 and macOS Apple Silicon packages on those
operating systems. Every runtime, wheel, figure-tool and data download is pinned
by SHA-256. The Windows package uses a native executable; macOS uses an `.app`.
The scientific sources and current data are identical to the Linux edition.
The additional figure renderer uses PDFium through `pypdfium2` for the exact
command-line options needed by the frozen scientific driver.

The manually dispatched `Native Windows and macOS packages` workflow builds on
GitHub's target-OS runners, runs saved-data and real short-computation checks,
compares four/twelve-worker results, checks the offscreen window, stop/continue,
and all five figures, and verifies a freshly extracted ZIP. Publication requires
both platform jobs to pass. A build script or runtime download alone is not a
validated platform release. These distributions are unsigned; macOS builds are
not Apple-notarized and can require normal operating-system user approval.

Maintainer commands on the target OS:

```text
python portable/build_native.py --platform windows-x86_64 --output /new/PISCO-windows-x86_64 --cache /build/cache
python portable/build_native.py --platform macos-arm64 --output /new/PISCO-macos-arm64 --cache /build/cache
```

Use only the command matching the build machine. macOS dependencies require
macOS 14 or later and Apple Silicon. No Intel Mac or Windows ARM release is
established by these builds. Linux build instructions follow below.

## Build from source

Build on Linux x86-64 with Python, a C compiler, make, ar and readelf.
The Python runtime, CPU wheels, Qt wheels and musl toolchain are hash-locked.
Download the optional `PISCO-native-inputs.tar.gz` maintainer asset from the
same release; its hash is in `native-inputs.lock.json`. It contains only the
native figure tools, libraries, fonts, Typst packages and licence notices.
No container image is used.

```text
python portable/prepare_native_root.py \
  --archive /download/PISCO-native-inputs.tar.gz --output /new/native-root
python portable/build_portable.py --output /new/PISCO-linux-x86_64 \
  --cache /build/cache --native-root /new/native-root \
  --results-zip /downloaded-package/results.zip \
  --results-release /downloaded-package/release.json
```

`--results-zip` and `--results-release` may be omitted for a computation-only
candidate. Building downloads locked dependencies; normal recipient operation
is offline. The explicit `public-source-files.json` allowlist prevents accidental
inclusion of other project folders. Update it deliberately for source changes.

A build remains a candidate until tested:

```text
python portable/acceptance.py --package /new/PISCO-linux-x86_64 \
  --output /new/acceptance-output --with-archive
python -m unittest discover -s portable -p 'test_*.py'
```

Run acceptance in a clean, offline environment as well as the intended desktop.
It checks saved results, real 4/12-worker calculations, selected numerical
agreement, stop/resume, the window and all five figure templates. Keep detailed
build logs locally; publish only the matching concise validation report and
checksums. A short test does not establish complete-matrix or cross-CPU equality.

`runtimes.lock.json` records the pinned runtime inputs. The release assets and
their platform-specific validation receipts identify the packages that passed
acceptance. The application displays errors and preserves console logs in the
selected output folder.
