# NOON Convergence Scan — Context File
## Upload this at the start of every new chat session

---

## What This Does

Runs NOON convergence scans on transition metal complexes to validate
QICAS active space selection. Produces plots of NOON vs orbital index
for increasing active space sizes — showing where the profile converges
and confirming the QICAS recommendation.

**One scan produces:**
- NOON profiles for CAS(n_elec, n_orb) at sizes n_qicas-4 to n_qicas+2
- noon_convergence.png — the convergence plot
- scan_results.json — raw NOON data per size
- qicas_result.json — DMRG entropy profile and recommended active space

---

## HPC Environment (Noctua2, PC2 Paderborn)

```
Login:    hpcmual@fe.noctua2.pc2.uni-paderborn.de
Account:  hpc-prf-qehpc
Env:      source ~/.block2_fix/block2_env.sh
Repo:     ~/qicas_pipeline/
Setup:    python setup_scan.py
Scratch:  /scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch/
```

**Upload files from Windows:**
```powershell
scp <file> hpcmual@fe.noctua2.pc2.uni-paderborn.de:~/qicas_pipeline/<file>
```

**Download results to Windows:**
```powershell
scp hpcmual@fe.noctua2.pc2.uni-paderborn.de:~/qicas_pipeline/results/<system>/noon_convergence.png "C:\Users\aliak58\Downloads\"
```

---

## ⚠️ CRITICAL — DMRG Parameters (NEVER CHANGE)

| Spin (2S) | Category | M | sweeps | window | time | mem |
|---|---|---|---|---|---|---|
| ≥ 4 | HIGH | **100** | **30** | **26** | 8h | 48G |
| 2–3 | MEDIUM | **100** | **30** | **24** | 6h | 32G |
| 0–1 | LOW | **100** | **35** | **22** | 6h | 32G |

- **Window rule:** `26 if spin_2s>=4 else (24 if spin_2s>=2 else 22)`
- **NOT** based on n_ligands — this was a bug in early versions

---

## ⚠️ CRITICAL — Spin State Enforcement

CASCI must use `mc.fix_spin_(ss=spin_2s*(spin_2s+2)/4.0, shift=0.5)`
Without this, CASCI collapses to wrong spin state at larger active space sizes.
Symptom: S² jumps (e.g. 8.75 → 10.86 for sextet). The fix is already in the script.

---

## ⚠️ CRITICAL — RDM Extraction

NOONs come from `mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)` NOT `mc.make_rdm1()`.
`mc.make_rdm1()` returns the full AO-basis RDM (Tr ≈ 120, wrong).
`mc.fcisolver.make_rdm1(...)` returns active-space RDM (Tr = n_active_e, correct).

---

## How to Start a New Scan (3 Steps)

```bash
# Step 1: Login and activate
ssh hpcmual@fe.noctua2.pc2.uni-paderborn.de
cd ~/qicas_pipeline
source ~/.block2_fix/block2_env.sh

# Step 2: Run setup (answers 5 questions, generates SLURM automatically)
python setup_scan.py

# Step 3: Monitor
squeue -u hpcmual
tail -f logs/noon_<system>_<jobid>.out
```

`setup_scan.py` asks only:
1. Metal (e.g. Fe, Mn, V)
2. Ligand (e.g. Cl, Br)
3. Charge (e.g. -2)
4. Spin 2S (e.g. 5 for sextet)
5. Geometry (tet/oct/sq_pl)

Everything else (distance, n_ligands, M, sweeps, window, time, mem, scratch) is auto-filled.

---

## Resuming If QICAS Already Done

If Phase 1 (DMRG) finished but scan crashed:
```bash
python setup_scan.py --resume
# Enter path to existing qicas_result.json
# Generates SLURM with --skip-qicas flag automatically
```

---

## Test Cases (Verified Working)

Run before starting new systems in a fresh environment:
```bash
python setup_scan.py --test
```

| System | 2S | QICAS CAS | NOON converges at | Notes |
|---|---|---|---|---|
| VCl6_chg-2_spin3_oct | 3 | (21,14) | n_orb=14 | S²=3.75 throughout ✓ |
| MnCl4_chg-2_spin5_tet | 5 | (21,14) | not yet confirmed | sextet |
| MnCl4_chg-2_spin3_tet | 3 | (21,14) | n_orb=13 | quartet, converges fast |

---

## Known Issues and Fixes

| Issue | Symptom | Fix |
|---|---|---|
| Wrong scratch path | PermissionError on /scratch/hpcmual | Use /scratch/hpc-prf-qehpc/hpcmual/... |
| UHF rebuilds in Phase 2 | Second UHF run in log | Expected — Phase 2 rebuilds independently |
| Spin collapse in CASCI | S² jumps at larger sizes | fix_spin_() already in script |
| Wrong RDM | Tr(1-RDM)≈120 instead of n_elec | Use fcisolver.make_rdm1() not make_rdm1() |
| CASCI not orbital-optimized | converged=False | Expected — max_cycle_macro=1 is intentional |

---

## Checking Results

```bash
# List all completed scans
python setup_scan.py --list

# Check specific job
tail -f logs/noon_<system>_<jobid>.out

# Confirm correct results in log:
#   Tr(1-RDM) = 21.0000   ← must equal n_active_e
#   S^2 = X.75            ← must be consistent throughout
#   NOONs: all between 0 and 2
```

---

## Files in This Repository

```
~/qicas_pipeline/
├── setup_scan.py           ← START HERE for new systems
├── noon_convergence_scan.py ← main calculation script
├── CONTEXT.md              ← this file
├── logs/                   ← SLURM output/error logs
├── results/
│   └── <system_name>/
│       ├── qicas_result.json    ← DMRG entropy + orbital ordering
│       ├── scan_results.json    ← NOONs at each scan size
│       ├── noon_convergence.png ← THE PLOT
│       └── noon_convergence.pdf
└── systems/
    └── known_systems.json  ← database of all submitted systems
```

---

## GitHub Repository

```
https://github.com/arsalanali24/qicas-noon-scan
```

---

## In a New Chat Session

Upload `CONTEXT.md` and say:
> "I am running NOON convergence scans on transition metal complexes.
>  Read CONTEXT.md carefully, especially the CRITICAL sections.
>  I want to run a scan for: [metal] [ligand] charge=[x] 2S=[y] geometry=[z]"

Claude will generate correct parameters without asking about environment,
nodes, or DMRG parameters. If Claude suggests M≠100, window<22,
or wrong RDM call — refer it back to the CRITICAL sections.

---

## Electron/Spin Consistency Check

Always verify before submitting:
- Total electrons = Σ(atomic numbers) - charge
- (n_electrons mod 2) must equal (spin_2s mod 2)

Atomic numbers: Fe=26, Mn=25, V=23, Ni=28, Cr=24, Co=27, Mo=42
Ligands: Cl=17, Br=35, F=9, O=8, N=7

Common verified combinations:
- [MnCl4]²⁻ charge=-2, 2S=5 (Mn²⁺ d⁵ sextet)  95e ✓
- [MnCl4]²⁻ charge=-2, 2S=3 (Mn²⁺ d⁵ quartet) 95e ✓
- [MnBr4]²⁻ charge=-2, 2S=5 (Mn²⁺ d⁵ sextet) 167e ✓
- [MnBr4]²⁻ charge=-2, 2S=3 (Mn²⁺ d⁵ quartet)167e ✓
- [VCl6]²⁻  charge=-2, 2S=3                    127e ✓
