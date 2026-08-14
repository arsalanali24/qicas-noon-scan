#!/usr/bin/env python3
"""
setup_scan.py — NOON Convergence Scan Setup
============================================
Run this on the LOGIN NODE to prepare and submit a NOON convergence scan.

Usage:
    python setup_scan.py              # interactive setup for new system
    python setup_scan.py --test       # run test case (V_Cl6) to verify setup
    python setup_scan.py --list       # show all completed scans
    python setup_scan.py --resume     # resume from existing qicas_result.json

What it does:
    1. Asks only: metal, ligand, charge, spin_2s, geometry
    2. Auto-fills: distance, n_ligands, DMRG params, window size, scratch, time, mem
    3. Validates environment (block2, pyscf)
    4. Generates SLURM script and submits
    5. Never asks about nodes, partitions, or environment

CRITICAL parameters (NEVER changed):
    M=100, sweeps=30 (HIGH/MEDIUM), sweeps=35 (LOW)
    window: HIGH(2S≥4)=26, MEDIUM(2S=2-3)=24, LOW(2S=0-1)=22
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR   = Path(__file__).parent.resolve()
RESULTS    = REPO_DIR / "results"
SYSTEMS_DB = REPO_DIR / "systems" / "known_systems.json"
SCAN_SCRIPT = REPO_DIR / "noon_convergence_scan.py"
LOGS_DIR   = REPO_DIR / "logs"

SCRATCH_BASE = "/scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch"
ENV_ACTIVATE = "source ~/.block2_fix/block2_env.sh"

# ── CRITICAL: DMRG parameters per spin category ───────────────────────────────
SPIN_CONFIG = {
    "HIGH":   {"spin_min": 4, "window": 26, "M": 100, "sweeps": 30, "time": "08:00:00", "mem": "48G"},
    "MEDIUM": {"spin_min": 2, "window": 24, "M": 100, "sweeps": 30, "time": "06:00:00", "mem": "32G"},
    "LOW":    {"spin_min": 0, "window": 22, "M": 100, "sweeps": 35, "time": "06:00:00", "mem": "32G"},
}

# ── Bond distance defaults (Å) ─────────────────────────────────────────────────
DIST_DEFAULTS = {
    ("Mn", "Cl", "tet"): 2.35, ("Mn", "Cl", "oct"): 2.56,
    ("Mn", "Br", "tet"): 2.50, ("Mn", "Br", "oct"): 2.63,
    ("Fe", "Cl", "tet"): 2.19, ("Fe", "Cl", "oct"): 2.35,
    ("Fe", "Br", "oct"): 2.50,
    ("V",  "Br", "oct"): 2.318,("V",  "Cl", "oct"): 2.394,
    ("Ni", "Br", "oct"): 2.53,
    ("Cr", "Cl", "tet"): 2.24,
    ("Mo", "Br", "tet"): 2.451,
    ("Ti", "Cl", "oct"): 2.115,
}

# n_ligands per geometry
N_LIGANDS = {"tet": 4, "oct": 6, "sq_pl": 4}

# Metals needing ECP (4d, 5d)
ECP_METALS = {"Mo", "Ru", "Rh", "Pd", "Ag", "W", "Re", "Os", "Ir", "Pt", "Au"}

# Test case — V_Cl6 (already verified working)
TEST_CASE = {
    "metal": "V", "ligand": "Cl", "charge": -2, "spin_2s": 3,
    "geometry": "oct", "dist_ang": 2.394, "n_ligands": 6,
    "name": "VCl6_chg-2_spin3_oct",
    "_note": "Known result: QICAS → CAS(21,14), NOONs converge at 14"
}


def get_spin_category(spin_2s: int) -> str:
    if spin_2s >= 4: return "HIGH"
    if spin_2s >= 2: return "MEDIUM"
    return "LOW"


def check_environment() -> bool:
    """Check block2 and pyscf are available. Return True if OK."""
    try:
        result = subprocess.run(
            ["python3", "-c", "import pyscf; import block2; print('OK')"],
            capture_output=True, text=True, timeout=10
        )
        if "OK" in result.stdout:
            print("  ✓ Environment OK (pyscf + block2 available)")
            return True
    except Exception:
        pass
    print("  ✗ Environment not activated")
    print(f"  Run: {ENV_ACTIVATE}")
    return False


def load_systems_db() -> dict:
    if SYSTEMS_DB.exists():
        with open(SYSTEMS_DB) as f:
            return json.load(f)
    return {}


def save_to_db(name: str, system: dict):
    db = load_systems_db()
    db[name] = system
    SYSTEMS_DB.parent.mkdir(exist_ok=True)
    with open(SYSTEMS_DB, "w") as f:
        json.dump(db, f, indent=2)


def ask(prompt: str, default=None, choices=None) -> str:
    """Ask user a question with optional default and choices."""
    if default is not None:
        prompt = f"{prompt} [{default}]"
    if choices:
        prompt = f"{prompt} ({'/'.join(choices)})"
    prompt += ": "
    while True:
        val = input(prompt).strip()
        if not val and default is not None:
            return str(default)
        if choices and val.lower() not in [c.lower() for c in choices]:
            print(f"  Please choose from: {', '.join(choices)}")
            continue
        if val:
            return val
        print("  Required — please enter a value")


def collect_system_info() -> dict:
    """Ask only the 5 required fields. Auto-fill everything else."""
    print("\n" + "="*50)
    print("  New System Setup")
    print("="*50)
    print("Required: metal, ligand, charge, 2S (spin), geometry")
    print("Everything else is auto-filled.\n")

    metal    = ask("Metal (e.g. Fe, Mn, V, Cr, Mo)").capitalize()
    ligand   = ask("Ligand (e.g. Cl, Br, F, O, N)").capitalize()
    charge   = int(ask("Total charge (e.g. -2, -1, 0)"))
    spin_2s  = int(ask("Spin 2S (e.g. 5 for sextet, 3 for quartet, 1 for doublet)"))
    geometry = ask("Geometry", choices=["tet", "oct", "sq_pl"]).lower()

    # Auto-fill distance
    dist_key = (metal, ligand, geometry)
    dist_default = DIST_DEFAULTS.get(dist_key)
    if dist_default:
        print(f"  → Auto-filled bond distance: {dist_default} Å")
        dist_ang = float(ask("Bond distance (Å)", default=dist_default))
    else:
        dist_ang = float(ask("Bond distance (Å) [no default for this combination]"))

    # Auto-fill n_ligands
    n_ligands = N_LIGANDS.get(geometry, 6)
    print(f"  → Auto-filled n_ligands: {n_ligands} (from {geometry})")

    # Electron count check
    atomic_Z = {"Mn":25,"Fe":26,"V":23,"Cr":24,"Ni":28,"Co":27,"Cu":29,
                 "Mo":42,"Ru":44,"Rh":45,"Pd":46,"Ti":22,"Ir":77,"Pt":78,
                 "Cl":17,"Br":35,"F":9,"O":8,"N":7,"H":1,"S":16}
    metal_Z  = atomic_Z.get(metal, 0)
    ligand_Z = atomic_Z.get(ligand, 0)
    n_elec   = metal_Z + n_ligands * ligand_Z - charge
    if (n_elec % 2) != (spin_2s % 2):
        print(f"\n  ⚠ WARNING: Electron count ({n_elec}) and spin_2s ({spin_2s}) parity mismatch!")
        print(f"     (n_elec mod 2 = {n_elec%2}, spin_2s mod 2 = {spin_2s%2})")
        if ask("Continue anyway?", choices=["y","n"]) == "n":
            sys.exit(1)
    else:
        print(f"  ✓ Electron check: {n_elec} electrons, spin_2s={spin_2s} — consistent")

    # Build system name
    name = f"{metal}{ligand}{n_ligands}_chg{charge:+d}_spin{spin_2s}_{geometry}"
    needs_ecp = metal in ECP_METALS

    system = {
        "name": name, "metal": metal, "ligand": ligand,
        "charge": charge, "spin_2s": spin_2s,
        "geometry": geometry, "dist_ang": dist_ang,
        "n_ligands": n_ligands, "needs_ecp": needs_ecp,
        "n_electrons": n_elec,
    }

    cat = get_spin_category(spin_2s)
    cfg = SPIN_CONFIG[cat]
    print(f"\n  → Spin category: {cat}")
    print(f"  → DMRG: M={cfg['M']}, sweeps={cfg['sweeps']}, window={cfg['window']}")
    print(f"  → SLURM: time={cfg['time']}, mem={cfg['mem']}")
    print(f"  → System name: {name}")

    return system


def generate_slurm(system: dict, result_dir: Path,
                   skip_qicas: bool = False) -> Path:
    """Generate SLURM submission script."""
    name     = system["name"]
    cat      = get_spin_category(system["spin_2s"])
    cfg      = SPIN_CONFIG[cat]
    scratch  = f"{SCRATCH_BASE}/{name}"

    qicas_json = result_dir / "qicas_result.json"
    scan_json  = result_dir / "scan_results.json"
    plot_pfx   = result_dir / "noon_convergence"

    skip_flag = "--skip-qicas \\" if skip_qicas else "\\"

    slurm = f"""#!/bin/bash
#SBATCH --job-name=noon_{name[:12]}
#SBATCH --account=hpc-prf-qehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time={cfg['time']}
#SBATCH --mem={cfg['mem']}
#SBATCH --output={LOGS_DIR}/noon_{name}_%j.out
#SBATCH --error={LOGS_DIR}/noon_{name}_%j.err
#SBATCH -p normal

{ENV_ACTIVATE}
cd {REPO_DIR}

# System: {name}
# Spin category: {cat} (2S={system['spin_2s']})
# DMRG: M={cfg['M']}, sweeps={cfg['sweeps']}, window={cfg['window']}

python {SCAN_SCRIPT} \\
    --system_dict '{json.dumps(system)}' \\
    {skip_flag}
    --M {cfg['M']} \\
    --n-below 4 \\
    --n-above 2 \\
    --scratch {scratch} \\
    --qicas-json {qicas_json} \\
    --scan-json  {scan_json} \\
    --plot-prefix {plot_pfx}
"""
    slurm_path = REPO_DIR / f"submit_{name}.slurm"
    with open(slurm_path, "w") as f:
        f.write(slurm)
    return slurm_path


def run_test_case() -> bool:
    """Submit the V_Cl6 test case and print expected output."""
    print("\n" + "="*50)
    print("  Test Case: VCl6_chg-2_spin3_oct")
    print("="*50)
    print("  Expected: QICAS → CAS(21,14)")
    print("  Expected: NOONs converge at n_orb=14, S²=3.75 throughout")
    print()

    if not check_environment():
        return False

    result_dir = RESULTS / TEST_CASE["name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # Check if already run
    if (result_dir / "noon_convergence.png").exists():
        print("  ✓ Test case already completed — results in:")
        print(f"    {result_dir}/noon_convergence.png")
        return True

    slurm = generate_slurm(TEST_CASE, result_dir)
    print(f"  Generated: {slurm}")

    ans = ask("Submit test job now?", default="y", choices=["y","n"])
    if ans == "y":
        result = subprocess.run(["sbatch", str(slurm)],
                                capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
        print("  Monitor: squeue -u hpcmual")
        print(f"  Log:     tail -f {LOGS_DIR}/noon_{TEST_CASE['name']}_*.out")
    return True


def list_completed():
    """Show all completed scans."""
    print("\n  Completed NOON scans:")
    print(f"  {'System':<35} {'Status':<12} {'QICAS CAS':<12}")
    print("  " + "-"*60)

    if not RESULTS.exists():
        print("  (none yet)")
        return

    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir(): continue
        has_plot  = any(d.glob("*.png"))
        has_scan  = (d / "scan_results.json").exists()
        has_qicas = (d / "qicas_result.json").exists()

        status = "✓ complete" if has_plot else ("scanning" if has_scan else ("QICAS done" if has_qicas else "running"))

        cas = ""
        if has_qicas:
            try:
                with open(d / "qicas_result.json") as f:
                    q = json.load(f)
                cas = f"({q['n_elec_qicas']},{q['n_active_qicas']})"
            except Exception:
                pass
        print(f"  {d.name:<35} {status:<12} {cas}")


def resume_from_json():
    """Use existing qicas_result.json to skip Phase 1."""
    print("\n  Resuming from existing QICAS result")
    json_path = ask("Path to qicas_result.json")
    if not Path(json_path).exists():
        print(f"  ERROR: {json_path} not found")
        sys.exit(1)

    with open(json_path) as f:
        qr = json.load(f)

    name = qr.get("name", "unknown_system")
    print(f"  Loaded: {name}")
    print(f"  QICAS recommendation: CAS({qr['n_elec_qicas']}, {qr['n_active_qicas']})")

    # Reconstruct system dict from qicas result
    system = {"name": name, "spin_2s": qr["spin_2s"],
              "_from_json": json_path}

    result_dir = RESULTS / name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Copy JSON to results dir if not already there
    import shutil
    target = result_dir / "qicas_result.json"
    if str(Path(json_path).resolve()) != str(target.resolve()):
        shutil.copy(json_path, target)

    slurm = generate_slurm(system, result_dir, skip_qicas=True)
    print(f"\n  Generated: {slurm}")
    ans = ask("Submit now?", default="y", choices=["y","n"])
    if ans == "y":
        result = subprocess.run(["sbatch", str(slurm)],
                                capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description="NOON Convergence Scan Setup")
    parser.add_argument("--test",   action="store_true", help="Run test case (VCl6)")
    parser.add_argument("--list",   action="store_true", help="List completed scans")
    parser.add_argument("--resume", action="store_true", help="Resume from existing qicas JSON")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  NOON Convergence Scan — Setup Tool")
    print("="*50)

    # Always check environment first
    print("\n[1] Checking environment ...")
    env_ok = check_environment()
    if not env_ok:
        print("\n  Activate with:")
        print(f"    {ENV_ACTIVATE}")
        sys.exit(1)

    RESULTS.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    if args.list:
        list_completed()
        return

    if args.test:
        run_test_case()
        return

    if args.resume:
        resume_from_json()
        return

    # Default: new system
    # Ask if they want to run test case first
    print("\n[2] Test case check ...")
    test_done = (RESULTS / TEST_CASE["name"] / "noon_convergence.png").exists()
    if test_done:
        print(f"  ✓ Test case (VCl6) already verified")
    else:
        ans = ask("Run test case first to verify setup?", default="n", choices=["y","n"])
        if ans == "y":
            run_test_case()
            print("\n  Wait for test job to complete, then rerun setup_scan.py for your new system.")
            return

    print("\n[3] New system setup ...")
    system = collect_system_info()

    # Save to database
    save_to_db(system["name"], system)

    # Create result directory
    result_dir = RESULTS / system["name"]
    result_dir.mkdir(parents=True, exist_ok=True)

    # Generate SLURM
    slurm = generate_slurm(system, result_dir)
    print(f"\n  Generated: {slurm}")
    print(f"  Results will go to: {result_dir}/")

    ans = ask("\nSubmit job now?", default="y", choices=["y","n"])
    if ans == "y":
        result = subprocess.run(["sbatch", str(slurm)],
                                capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
        print(f"\n  Monitor:  squeue -u hpcmual")
        print(f"  Log:      tail -f {LOGS_DIR}/noon_{system['name']}_*.out")
        print(f"  Results:  ls {result_dir}/")
    else:
        print(f"\n  To submit later:")
        print(f"    sbatch {slurm}")


if __name__ == "__main__":
    main()
