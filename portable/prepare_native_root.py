"""Unpack the hash-locked native build inputs; no container is required."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name('native-inputs.lock.json').read_text())
    with args.archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != lock['sha256']:
        parser.error('Native build-input checksum mismatch.')
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('Use a new/empty build directory.')
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive) as archive:
        archive.extractall(root, filter='data')
    print('Prepared native build inputs:', root)


if __name__ == '__main__':
    main()
