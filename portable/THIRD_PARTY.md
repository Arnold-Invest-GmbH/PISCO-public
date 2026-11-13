# Included components and provenance

PISCO's repository licence is in `LICENSE`. Third-party components retain their
own licences. Readable PISCO source is under `source/`; the native C launchers
and Python desktop code are under `source/portable/`.

- **CPython / python-build-standalone**: exact official release URL and archive
  SHA-256 in `release.json` and `source/portable/runtimes.lock.json`. The runtime
  includes its own licence notices. Project: https://github.com/astral-sh/python-build-standalone
- **Numerical Python packages**: hash-locked CPU wheel requirements in
  `source/container/requirements.lock`; actual versions in `python-packages.txt`.
  Package licences/notices are retained under `runtime/lib/python3.14/site-packages/`.
- **Qt for Python / PySide6 Essentials and Shiboken 6.12.0**: hashes in
  `source/portable/gui-linux.lock`. LGPLv3 and GPLv3 texts are under
  `source/portable/third_party/`. The Qt libraries remain separate dynamically
  loaded files. Corresponding upstream source: https://code.qt.io/cgit/pyside/pyside-setup.git/?h=v6.12.0
  and https://download.qt.io/archive/qt/6.12/
- **Poppler, supporting native libraries and fonts**: Debian licence/copyright
  notices in `tools/licenses/`, precise copied-library hashes in
  `tools/native-libraries.json`. Debian source packages are identified by those
  notices and the recorded package versions; archive: https://snapshot.debian.org/
- **Typst 0.15.1**: official static binary, source/licence at
  https://github.com/typst/typst/tree/v0.15.1 ; included licence under
  `source/portable/third_party/Typst-LICENSE.txt`. CeTZ and oxifmt source and
  licence files are retained in `tools/cache/typst/packages/preview/`.
- **musl 1.2.5**: static launcher runtime; licence in `tools/licenses/musl.txt`,
  exact source URL/hash in `release.json`. Build recipe: `source/portable/build_toolchain.py`.

`validation/native-root-provenance.json` identifies the exact native-library
build inputs. Hashes identify files and detect accidental changes; they are
not a publisher signature. No publisher signature is claimed for this local build.
