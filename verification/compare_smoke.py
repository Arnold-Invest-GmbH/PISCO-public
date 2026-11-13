#!/usr/bin/env python3
"""Compare saved short-run evidence, without importing the numerical program.

Exact equality refers to decoded JSON numerical fields, not whole-file hashes.
Timings and environment descriptions are deliberately excluded. No tolerance
is used to turn cross-build differences into an equality claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield float(value)


def numerical(value):
    if isinstance(value, dict):
        return {k: numerical(v) for k, v in value.items()
                if k not in ('elapsed_s', 'wall_time_s', 'exclusive_seconds')}
    if isinstance(value, list):
        return [numerical(v) for v in value]
    return value


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compare(left, right):
    indices = [{r['evidence']: r for r in read(p / 'documentation/run_index.json')}
               for p in (left, right)]
    if set(indices[0]) != set(indices[1]) or not indices[0]:
        raise ValueError('The two run matrices differ or are empty.')
    rows = []
    for key in sorted(indices[0]):
        a, b = [read(p / 'documentation' / key) for p in (left, right)]
        ea, eb = [r['extra']['r2_evidence'] for r in (a, b)]
        ca, cb = a['best_compliance'], b['best_compliance']
        if not all(math.isfinite(c) for c in (ca, cb)):
            raise ValueError('Non-finite compliance: ' + key)
        da, db = list(flatten(a['best_design'])), list(flatten(b['best_design']))
        if len(da) != len(db):
            raise ValueError('Design dimensions differ: ' + key)
        absolute = abs(ca - cb)
        row = {'key': key, 'compliance_left': ca, 'compliance_right': cb,
               'absolute_compliance_difference': absolute,
               'relative_compliance_difference': absolute / max(abs(ca), abs(cb), 1e-300),
               'compliance_exact': ca == cb,
               'design_exact': a['best_design'] == b['best_design'],
               'max_absolute_design_difference': max(abs(x-y) for x,y in zip(da, db)),
               'trajectory_exact': a['trajectory'] == b['trajectory'],
               'fem_calls_equal': all(indices[0][key][k] == indices[1][key][k]
                                      for k in ('logical_fem_calls', 'recorded_fem_calls', 'fem_calls_by_role'))}
        for field in ('fem_events', 'training', 'predictions', 'test_predictions', 'incumbents'):
            va, vb = numerical(ea[field]), numerical(eb[field])
            row[field + '_exact'] = va == vb
            row[field + '_sha256'] = [signature(va), signature(vb)]
        rows.append(row)
    exact_fields = [k for k in rows[0] if k.endswith('_exact')] + ['fem_calls_equal']
    return {'left': str(left), 'right': str(right), 'rows': rows,
            'scope': 'Saved 16-row reduced-budget matrix; decoded numerical fields only, no timings; not a full-run or cross-hardware guarantee',
            'compared_rows': len(rows),
            'all_selected_numerical_fields_exact': all(r[k] for r in rows for k in exact_fields),
            'equality_counts': {k: sum(r[k] for r in rows) for k in exact_fields},
            'max_absolute_compliance_difference': max(r['absolute_compliance_difference'] for r in rows),
            'max_relative_compliance_difference': max(r['relative_compliance_difference'] for r in rows),
            'max_absolute_design_difference': max(r['max_absolute_design_difference'] for r in rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-exact', action='store_true')
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    raise SystemExit(1 if args.require_exact and not result['all_selected_numerical_fields_exact'] else 0)
