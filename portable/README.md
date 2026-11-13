# Native Linux application

Extract `PISCO-linux-x86_64.zip`, open `PISCO`, select an output folder and
choose **Check paper results**. The saved results are included. See
`START_HERE.html` and `VALIDATION.md` for actions and tested conditions.
No Python installation or Docker is required for recipients.

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

`runtimes.lock.json` records possible runtime inputs for additional systems;
only the Linux package has been built and tested. The application displays
errors and preserves console logs in the selected output folder.
