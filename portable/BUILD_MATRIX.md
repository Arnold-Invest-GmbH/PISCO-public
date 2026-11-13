# Tested platforms

| Package | Test environment | Scope |
| --- | --- | --- |
| Linux x86-64 | Author's AMD Ryzen AI 9 HX 370, CachyOS/glibc 2.44 and isolated Debian 12/glibc 2.36 userland | Native desktop, offline acceptance, read-only package |
| Windows x64 | GitHub-hosted Windows Server 2025 x64 | Native executable, offline acceptance with outbound firewall rules, native and offscreen window |
| macOS Apple Silicon | GitHub-hosted macOS 15 ARM64 | Native application, offline acceptance with network denied, native and offscreen window |

Each package passed saved-data checks, real short calculations with four and
twelve workers, exact agreement of selected numerical fields between those two
runs, stop/continue, and all five diagram templates. The actual delivery ZIP
was extracted into a fresh location and its saved-data action passed again.
Windows/macOS extraction also tested a path containing spaces and a non-ASCII
character. See VALIDATION.md and the release receipts for evidence.

Linux requires glibc 2.36+ and X11/XWayland for its window. macOS dependencies
require macOS 14+ and Apple Silicon. Intel Macs and Windows ARM are not covered.
Hosted OS tests do not cover every end-user machine. The Debian userland shares
the author's laptop CPU/kernel. No complete experiment matrix was rerun in
these bundled runtimes, and no cross-platform bitwise equality is claimed.
The applications are unsigned; the macOS application is not Apple-notarized.
