"""
Merge partial result JSONs produced by cluster array jobs into a single eval JSON.

Usage:
  python -m scripts.evaluate.aggregate --label c_pomcp_n50_s30
  python -m scripts.evaluate.aggregate --label c_pomcp_n50_s30 --out results/c_pomcp_final.json

Expects files:  results/<label>_<red>_<maxsteps>_partial.json
Writes to:      results/<label>_eval.json  (or --out path)
"""
import os
import sys
import json
import argparse
import glob

from scripts.config import RESULTS_DIR

RED_AGENTS = ['bline', 'meander', 'sleep']
EVAL_STEPS  = [30, 50, 100]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, help='Label used when running evaluate.py')
    parser.add_argument('--out',   default=None,  help='Output path (default: results/<label>_eval.json)')
    args = parser.parse_args()

    expected = {f"{r}_{s}" for r in RED_AGENTS for s in EVAL_STEPS}
    found = {}

    for red in RED_AGENTS:
        for steps in EVAL_STEPS:
            key  = f"{red}_{steps}"
            path = f"{RESULTS_DIR}/{args.label}_{red}_{steps}_partial.json"
            if not os.path.exists(path):
                print(f"  MISSING: {path}", file=sys.stderr)
                continue
            with open(path) as f:
                data = json.load(f)
            combo = data['combinations'].get(key)
            if combo is None:
                print(f"  BAD key in {path}", file=sys.stderr)
                continue
            found[key] = combo

    missing = expected - set(found)
    if missing:
        print(f"\nERROR: {len(missing)} combination(s) missing: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # Reconstruct full result
    # Infer algo/label from any partial file
    sample_path = f"{RESULTS_DIR}/{args.label}_{RED_AGENTS[0]}_{EVAL_STEPS[0]}_partial.json"
    with open(sample_path) as f:
        sample = json.load(f)

    result = {
        'label': args.label,
        'algo':  sample.get('algo', 'unknown'),
        'combinations': found,
        'total': sum(v['mean'] for v in found.values()),
    }

    out_path = args.out or f"{RESULTS_DIR}/{args.label}_eval.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Aggregated {len(found)} combinations  →  total = {result['total']:.2f}")
    print(f"Saved: {out_path}")

    # Per-combination summary
    print("\n  combination       mean    std")
    for red in RED_AGENTS:
        for steps in EVAL_STEPS:
            key = f"{red}_{steps}"
            v = found[key]
            print(f"  {key:<16}  {v['mean']:7.2f}  {v['std']:7.2f}")


if __name__ == '__main__':
    main()
