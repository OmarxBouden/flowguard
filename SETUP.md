# Setup

End-to-end install for a fresh clone.

## 1. Clone with submodules

```
git clone <repo-url> flowguard
cd flowguard
git submodule update --init --recursive
```

The simulator (`CybORG_plus_plus/`) is a git submodule; the recursive flag also pulls its inner `Debugged_CybORG/` tree.

## 2. Create the environment

Either conda or pip works. `environment.yml` and `requirements.txt` mirror each other — keep both in sync if you change deps.

### Conda (recommended)

```
conda env create -f environment.yml
conda activate cyborg
```

`environment.yml` pins Python 3.9 and a CPU-only `torch==2.3.1`. The env name in the file is `cyborg`; rename it with `-n <yourname>` on creation if you prefer.

### Pip + venv

```
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Install CybORG editable

The simulator is a Python package living inside the submodule. Install it in editable mode so `from CybORG import CybORG` works:

```
pip install -e CybORG_plus_plus/Debugged_CybORG/CybORG
```

## 4. Apply local patches to the submodule

We keep small bug fixes for CybORG++ as patch files under `patches/` rather than maintaining a fork. Apply them:

```
./patches/apply.sh
```

Idempotent — re-running is a no-op if patches are already applied. See `patches/README.md` for what each patch does.

## 5. Verify

Quick smoke test that everything wires up:

```
python -m scripts.evaluate.evaluate \
  --model models/m1_ppo_baseline_mix --algo ppo
```

You should see 9 combinations evaluated and a `results/m1_ppo_baseline_mix_eval.json` written. For an M2 (green-on) eval, append `--green behavioral`.

## Cluster / slurm

`scripts/evaluate/slurm_eval_array.sh` is the array-job entry point for distributed eval. Each array index runs one `(red, max_steps)` cell and writes a `*_<red>_<steps>_partial.json`; `scripts/evaluate/aggregate.py` stitches them into a single `*_eval.json`.
