# QICAS NOON Convergence Scan

Validates QICAS active space selection for transition metal complexes
by plotting Natural Orbital Occupation Numbers (NOONs) as a function
of active space size.

## Quick Start (on Noctua2)

```bash
git clone https://github.com/arsalanali24/qicas-noon-scan.git
cd qicas-noon-scan
source ~/.block2_fix/block2_env.sh

# Verify setup with test case first
python setup_scan.py --test

# New system
python setup_scan.py
```

## What It Produces

For each system: a convergence plot showing NOON profiles at increasing
active space sizes. When the profile stops changing, that is the minimal
sufficient active space — which should match the QICAS recommendation.

## Files

| File | Purpose |
|---|---|
| `setup_scan.py` | Interactive setup — run this to start |
| `noon_convergence_scan.py` | Main calculation script |
| `CONTEXT.md` | Upload to new Claude chat for instant context |

## Starting a New Claude Chat

Upload `CONTEXT.md` and say what system you want to run.
Claude will generate correct parameters immediately.

## Critical Parameters

M=100 always. Window: 26 (HIGH, 2S≥4), 24 (MEDIUM, 2S=2-3), 22 (LOW, 2S=0-1).
Never change these.
