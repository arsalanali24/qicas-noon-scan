#!/usr/bin/env python3
"""
noon_convergence_scan.py
========================
Runs QICAS on V_Cl6_chg-2_spin3_oct_d2p394, then performs a CASSCF
active-space size scan to produce NOON convergence plots equivalent to
the autoCAS-style figures.

Two-phase workflow
------------------
Phase 1 — QICAS (DMRG, M=100)
    UHF → frontier window → DMRG → entropy-ranked orbital ordering
    Saves:  qicas_result.json   (orbital ordering + DMRG NOONs)

Phase 2 — CASSCF size scan
    For each n_orb in [n_min … n_qicas + 2]:
        Take top-n_orb entropy-ranked orbitals
        Run CASSCF(n_elec_fixed, n_orb)
        Extract NOONs from 1-RDM eigenvalues
    Saves:  scan_results.json

Phase 3 — Plot
    NOON vs orbital index, one curve per active space size
    Saves:  noon_convergence.pdf  +  noon_convergence.png

Usage
-----
# Full run (both phases + plot):
    source ~/.block2_fix/block2_env.sh
    python noon_convergence_scan.py

# Skip QICAS if already done (re-use saved JSON):
    python noon_convergence_scan.py --skip-qicas

# Skip scan too, just replot:
    python noon_convergence_scan.py --only-plot

# Different system (must be in SYSTEMS dict in qicas_canonical.py):
    python noon_convergence_scan.py --system Mo_Br4_chg0_spin4_tet_d2p451

System chosen for demo
----------------------
V_Cl6_chg-2_spin3_oct_d2p394
  spin_2s = 3  (S = 3/2, high-spin)
  No ECP required  (pure 3d metal)
  CI dim at QICAS recommendation ≈ 120–52 k  (plain CASSCF, seconds each)
  Scan range: ~5 sizes, each < 1 min on a single node
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
try:
    from pyscf import dft as _dft_mod
except ImportError:
    _dft_mod = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Path setup: allow running from repo root or scripts/ ──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.join(_HERE, "..")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── PySCF imports (only needed for phases 1 & 2) ─────────────────────────────
def _import_pyscf():
    try:
        from pyscf import gto, scf, mcscf, dft
        from pyscf.dmrgscf import dmrgci
        return gto, scf, mcscf, dmrgci
    except ImportError as e:
        print(f"[ERROR] PySCF / block2 not available: {e}")
        print("        Activate: source ~/.block2_fix/block2_env.sh")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# System definition  (matches qicas_canonical.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

# Geometry builders
def _oct(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0),(0,0,d),(0,0,-d)]

def _tet(d):
    s = d / math.sqrt(3)
    return [(s,s,s),(s,-s,-s),(-s,s,-s),(-s,-s,s)]

def _sqpl(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0)]

GEOM_BUILDERS = {"oct": _oct, "tet": _tet, "sq_pl": _sqpl}

LIGAND_ATOMS = {
    "Cl": [("Cl", 1)], "Br": [("Br", 1)], "F": [("F", 1)],
    "I":  [("I",  1)], "H":  [("H",  1)], "O": [("O", 1)],
    "N":  [("N",  1)], "S":  [("S",  1)],
}

# All high-spin systems available for --system override
SYSTEMS = {
    "V_Cl6_chg-2_spin3_oct_d2p394":  dict(spin_2s=3, metal="V",  ligand_raw="Cl",
                                           n_ligands=6, geometry="oct", dist_ang=2.394,
                                           charge=-2, metal_row="3d", needs_ecp=False,
                                           ref_no=3, ref_ne=9),
    "V_H4_chg-2_spin3_tet_d1p73":    dict(spin_2s=3, metal="V",  ligand_raw="H",
                                           n_ligands=4, geometry="tet", dist_ang=1.73,
                                           charge=-2, metal_row="3d", needs_ecp=False,
                                           ref_no=3, ref_ne=9),
    "Mo_Br4_chg0_spin4_tet_d2p451":  dict(spin_2s=4, metal="Mo", ligand_raw="Br",
                                           n_ligands=4, geometry="tet", dist_ang=2.451,
                                           charge=0,  metal_row="4d", needs_ecp=True,
                                           ref_no=6, ref_ne=10),

    "MnCl4_chg-2_spin5_tet":  dict(spin_2s=5, metal="Mn", ligand_raw="Cl",
                                    n_ligands=4, geometry="tet", dist_ang=2.35,
                                    charge=-2, metal_row="3d", needs_ecp=False,
                                    ref_no=12, ref_ne=19),
    "MnCl4_chg-2_spin3_tet":  dict(spin_2s=3, metal="Mn", ligand_raw="Cl",
                                    n_ligands=4, geometry="tet", dist_ang=2.35,
                                    charge=-2, metal_row="3d", needs_ecp=False,
                                    ref_no=12, ref_ne=19),
    "MnBr4_chg-2_spin5_tet":  dict(spin_2s=5, metal="Mn", ligand_raw="Br",
                                    n_ligands=4, geometry="tet", dist_ang=2.63,
                                    charge=-2, metal_row="3d", needs_ecp=False,
                                    ref_no=12, ref_ne=19),
    "MnBr4_chg-2_spin3_tet":  dict(spin_2s=3, metal="Mn", ligand_raw="Br",
                                    n_ligands=4, geometry="tet", dist_ang=2.63,
                                    charge=-2, metal_row="3d", needs_ecp=False,
                                    ref_no=12, ref_ne=19),

    "VBr6_chg-3_spin2_oct":   dict(spin_2s=2, metal="V",  ligand_raw="Br",
                                    n_ligands=6, geometry="oct", dist_ang=2.318,
                                    charge=-3, metal_row="3d", needs_ecp=False,
                                    ref_no=6,  ref_ne=10),
    "NiBr6_chg-4_spin2_oct":  dict(spin_2s=2, metal="Ni", ligand_raw="Br",
                                    n_ligands=6, geometry="oct", dist_ang=2.53,
                                    charge=-4, metal_row="3d", needs_ecp=False,
                                    ref_no=5,  ref_ne=10),
    "VBr6_chg-2_spin1_oct":   dict(spin_2s=1, metal="V",  ligand_raw="Br",
                                    n_ligands=6, geometry="oct", dist_ang=2.318,
                                    charge=-2, metal_row="3d", needs_ecp=False,
                                    ref_no=3,  ref_ne=4),
    "MnBr4_chg-1_spin0_tet":  dict(spin_2s=0, metal="Mn", ligand_raw="Br",
                                    n_ligands=4, geometry="tet", dist_ang=2.63,
                                    charge=-1, metal_row="3d", needs_ecp=False,
                                    ref_no=13, ref_ne=18),
    "Ti_Cl6_chg0_spin4_oct_d2p115":  dict(spin_2s=4, metal="Ti", ligand_raw="Cl",
                                           n_ligands=6, geometry="oct", dist_ang=2.115,
                                           charge=0,  metal_row="3d", needs_ecp=False,
                                           ref_no=4, ref_ne=10),
}


def build_mol(name, s, gto):
    """Build a PySCF Mole object from the system dict."""
    dist    = s["dist_ang"] or 2.10
    builder = GEOM_BUILDERS.get(s["geometry"], _oct)
    lig_pos = builder(dist)[: s["n_ligands"]]
    atoms   = [(s["metal"], (0., 0., 0.))]
    tmpl    = LIGAND_ATOMS.get(s["ligand_raw"], [("Cl", 1)])
    for pos in lig_pos:
        r  = math.sqrt(sum(x**2 for x in pos))
        uv = tuple(x / r for x in pos) if r > 0 else (1, 0, 0)
        off = 0.0
        for sym, nrep in tmpl:
            for _ in range(nrep):
                coord = tuple(pos[i] + uv[i] * off for i in range(3))
                atoms.append((sym, coord))
                off += 1.10
    mol = gto.Mole()
    mol.atom     = atoms
    mol.charge   = s["charge"]
    mol.spin     = s["spin_2s"]
    mol.basis    = "def2-svp"
    mol.unit     = "angstrom"
    mol.symmetry = False
    if s["needs_ecp"]:
        mol.ecp = {s["metal"]: "def2-svp"}
    mol.verbose = 4
    mol.build()
    return mol


def run_uhf(mol, scf, s):
    """DFT/PBE0 with multiple fallback strategies."""
    from pyscf import dft
    if s["spin_2s"] == 0:
        mf = dft.RKS(mol)
        mf.xc = 'pbe0'
        print("  [DFT] Using RKS/PBE0 (singlet)")
    else:
        mf = dft.UKS(mol)
        mf.xc = 'pbe0'
        print("  [DFT] Using UKS/PBE0")
    mf.max_cycle = 500
    mf.conv_tol  = 1e-10
    for init, ls, damp in [("atom", 0.2, 0.0),
                             ("minao", 0.5, 0.0),
                             ("minao", 0.3, 0.5),
                             ("1e",   0.5, 0.3)]:
        mf.init_guess  = init
        mf.level_shift = ls
        mf.damp        = damp
        mf.kernel()
        if mf.converged:
            print(f"  [UHF] Converged (init={init}, ls={ls})")
            return mf
    raise RuntimeError("UHF did not converge with any strategy")


def select_window(mf, s, scf):
    """Frontier orbital window — mirrors qicas_canonical.py exactly."""
    # Handle both RHF (singlet) and UHF
    if s["spin_2s"] == 0:
        # RKS: mo_coeff is (nao, nmo), mo_occ is (nmo,)
        n_mo = mf.mo_coeff.shape[1]
        n_a  = int((mf.mo_occ > 0).sum())
        n_b  = int((mf.mo_occ > 1).sum())
    else:
        # UKS: mo_coeff is (2, nao, nmo)
        n_mo = mf.mo_coeff[0].shape[1]
        n_a  = int(mf.mo_occ[0].sum())
        n_b  = int(mf.mo_occ[1].sum())
    win  = 26 if s["spin_2s"] >= 4 else (24 if s["spin_2s"] >= 2 else 22)
    half = win // 2
    start = max(0, n_a - half)
    end   = min(n_mo, start + win)
    if end - start < win:
        start = max(0, end - win)
    window = list(range(start, end))
    for idx in range(n_b, n_a):
        if idx not in window:
            window = list(range(min(window[0], idx),
                                max(window[-1] + 1, idx + 1)))
    return window


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — QICAS (DMRG)
# ─────────────────────────────────────────────────────────────────────────────

def run_qicas(name, s, out_json="qicas_result.json",
              M=100, nsweeps=30, scratch="/tmp/noon_scan", args=None):
    """
    Run UHF + DMRG on the orbital window.
    Returns a dict saved to out_json with:
        window_indices   — absolute MO indices in the window
        entropy_ranked   — window-relative indices sorted by entropy (desc)
        noon_window      — DMRG 1-RDM diagonal (NOONs for all window orbs)
        n_active_qicas   — QICAS-recommended number of active orbitals
        n_elec_qicas     — corresponding electron count
        e_dmrg           — DMRG total energy
    """
    gto, scf, mcscf, dmrgci = _import_pyscf()
    os.makedirs(scratch, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Phase 1 — QICAS on {name}")
    print(f"{'='*60}")

    # ── Step 1: Build molecule ────────────────────────────────────────────────
    print("\n[Step 1] Building molecule ...")
    mol = build_mol(name, s, gto)
    print(f"  Atoms: {mol.natm},  charge: {mol.charge},  spin_2s: {mol.spin}")

    # ── Step 2: UHF ──────────────────────────────────────────────────────────
    print("\n[Step 2] UHF ...")
    t0 = time.time()
    mf = run_uhf(mol, scf, s)
    print(f"  E(UHF) = {mf.e_tot:.8f}  ({time.time()-t0:.1f}s)")

    # ── Step 3: Orbital window ────────────────────────────────────────────────
    window = select_window(mf, s, scf)
    n_win  = len(window)
    print(f"\n[Step 3] Window: {n_win} orbitals  (indices {window[0]}–{window[-1]})")

    # ── Step 4: DMRG on window ───────────────────────────────────────────────
    print(f"\n[Step 4] DMRG  M={M}, sweeps={nsweeps} ...")
    # Handle RHF (singlet) vs UHF
    if mol.spin == 0:
        occ = mf.mo_occ / 2.0   # RHF: mo_occ counts electrons per orbital (0 or 2)
    else:
        occ = (mf.mo_occ[0] + mf.mo_occ[1]) / 2.0  # UHF: alpha + beta
    n_e   = int(round(2 * occ[window].sum()))
    if (n_e - mol.spin) % 2 != 0: n_e += 1
    if (n_e - mol.spin) % 2 != 0: n_e -= 2
    n_e   = int(np.clip(n_e, mol.spin, 2 * n_win))

    # Handle RHF vs UHF mo_coeff
    if mol.spin == 0:
        mo_alpha = mf.mo_coeff          # RHF: (nao, nmo)
    else:
        mo_alpha = mf.mo_coeff[0]       # UHF: alpha spin (nao, nmo)
    n_mo    = mo_alpha.shape[1]
    all_i   = list(range(n_mo))
    core    = sorted([i for i in all_i if i not in window and occ[i] > 1.5])
    virt    = sorted([i for i in all_i if i not in window and occ[i] < 0.5])
    other   = sorted([i for i in all_i if i not in window
                       and i not in core and i not in virt])
    mo_ord  = mo_alpha[:, core + window + other + virt]

    mc_dmrg = mcscf.CASSCF(mf, n_win, n_e)
    mc_dmrg.fcisolver = dmrgci.DMRGCI(mol, maxM=M, tol=1e-8)
    mc_dmrg.fcisolver.scratchDirectory  = scratch
    mc_dmrg.fcisolver.runtimeDir        = scratch
    mc_dmrg.fcisolver.maxIter           = nsweeps
    mc_dmrg.fcisolver.block_extra_keyword = ["num_thrds 8"]
    mc_dmrg.max_cycle_macro = 1          # single-shot DMRG, no orbital opt

    t0 = time.time()
    e_dmrg = mc_dmrg.kernel(mo_ord)[0]
    t_dmrg = time.time() - t0
    print(f"  E(DMRG) = {e_dmrg:.8f}  ({t_dmrg:.1f}s)")

    # ── Step 5: Extract 1-RDM → NOONs and entropies ─────────────────────────
    print("\n[Step 5] Extracting RDMs ...")
    dm1, dm2 = mc_dmrg.fcisolver.make_rdm12(
        mc_dmrg.ci, mc_dmrg.ncas, mc_dmrg.nelecas)
    # dm1[i,i] = total occupation of window orbital i  (0 ≤ n_i ≤ 2)
    noon_window = [float(dm1[i, i]) for i in range(n_win)]

    # Two-orbital entropy from the diagonal 2-RDM elements
    def _S2(ni, Gii):
        lam = np.clip([1 - 2*ni + Gii, ni - Gii, ni - Gii, Gii], 1e-14, 1.0)
        lam /= lam.sum()
        return -float(np.dot(lam, np.log(lam)))

    dm2_phys = dm2.transpose((0, 2, 3, 1)).copy()
    Gamma     = (2 * dm2_phys + dm2_phys.transpose(0, 1, 3, 2)) / 6.0
    entropies = np.array([_S2(dm1[i,i]/2, Gamma[i,i,i,i]/4) for i in range(n_win)])

    # ── Step 6: QICAS active space from entropy plateau ───────────────────────
    print("\n[Step 6] Entropy plateau → active space size ...")
    # Sort by entropy descending; find largest gap below floor
    entropy_ranked = np.argsort(entropies)[::-1].tolist()
    ss             = entropies[entropy_ranked]
    floor          = 0.02
    sig_mask       = ss > floor
    n_sig          = int(sig_mask.sum())

    # Largest gap in the significant orbitals → D_CAS
    if n_sig >= 2:
        gaps    = ss[:-1] - ss[1:]
        d_cas   = int(np.argmax(gaps[:n_sig]) + 1)
    else:
        d_cas   = max(2, n_sig)
    d_cas = max(d_cas, s.get("ref_no", 2))     # at least reference size
    print(f"  Entropy-ranked top entropies: {ss[:d_cas+2].round(4).tolist()}")
    print(f"  D_CAS (QICAS recommendation) = {d_cas}")

    # Electron count for QICAS active space
    active_abs  = [window[r] for r in entropy_ranked[:d_cas]]
    n_elec_act  = int(round(2 * occ[active_abs].sum()))
    if (n_elec_act - mol.spin) % 2 != 0: n_elec_act += 1
    if (n_elec_act - mol.spin) % 2 != 0: n_elec_act -= 2
    n_elec_act  = int(np.clip(n_elec_act, mol.spin, 2 * d_cas))
    print(f"  QICAS:  CAS({n_elec_act}, {d_cas})")
    # Override from --qicas-n-active if provided (matches paper FQI result)
    if getattr(args, "qicas_n_active", None):
        print(f"  [OVERRIDE] n_active {d_cas} → {args.qicas_n_active} (from --qicas-n-active)")
        d_cas = args.qicas_n_active

    # ── Save result ───────────────────────────────────────────────────────────
    result = {
        "name":              name,
        "spin_2s":           s["spin_2s"],
        "window_indices":    window,        # absolute MO indices
        "entropy_ranked":    entropy_ranked, # window-relative, desc entropy
        "entropies":         entropies.tolist(),
        "noon_window":       noon_window,    # DMRG NOONs for all window orbs
        "n_active_qicas":    d_cas,
        "n_elec_qicas":      n_elec_act,
        "e_dmrg":            float(e_dmrg),
        "t_dmrg_s":          float(t_dmrg),
        # Store MO info needed for Phase 2
        "n_mo":              n_mo,
        "core_indices":      core,
        "window_mo_order":   (core + window + other + virt),
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_json}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — CASSCF size scan
# ─────────────────────────────────────────────────────────────────────────────

def run_casscf_scan(name, s, qicas_result, out_json="scan_results.json",
                    n_below=4, n_above=2, scratch="/tmp/noon_scan", max_orbs=16):
    """
    Run CASSCF at nested active space sizes centred on QICAS recommendation.

    Scan:  n_orb = n_qicas - n_below  …  n_qicas + n_above
    Electrons fixed at n_elec_qicas throughout
      (adding orbitals = adding virtuals → electron count unchanged)

    For each size:
        Take top-n_orb orbitals by QICAS entropy ranking
        Build MO ordering:  core | active (top-n_orb) | rest of window | virtual
        Run CASSCF(n_elec_qicas, n_orb)
        Diagonalise 1-RDM → NOONs (eigenvalues, sorted descending)
    """
    gto, scf, mcscf, _ = _import_pyscf()
    os.makedirs(scratch, exist_ok=True)

    n_qicas       = qicas_result["n_active_qicas"]
    n_elec_fixed  = qicas_result["n_elec_qicas"]
    spin_2s       = qicas_result["spin_2s"]
    entropy_ranked = qicas_result["entropy_ranked"]   # window-relative
    window         = qicas_result["window_indices"]   # absolute
    mo_order_abs   = qicas_result["window_mo_order"]  # full absolute ordering

    # Scan range
    n_min = max(int(math.ceil(n_elec_fixed / 2)), n_qicas - n_below, 2)
    n_max = n_qicas + n_above
    scan_sizes = [n for n in range(n_min, n_max + 1) if n <= max_orbs]
    print(f"\n{'='*60}")
    print(f"  Phase 2 — CASSCF scan on {name}")
    print(f"  QICAS recommendation: CAS({n_elec_fixed}, {n_qicas})")
    print(f"  Scan: n_orb = {scan_sizes}  (n_elec fixed = {n_elec_fixed})")
    print(f"{'='*60}")

    # Rebuild molecule and UHF (needed for CASSCF)
    print("\n[Rebuild mol + UHF for CASSCF] ...")
    mol = build_mol(name, s, gto)
    mf  = run_uhf(mol, scf, s)
    occ = (mf.mo_occ[0] + mf.mo_occ[1]) / 2.0

    # Reconstruct the MO coefficient matrix in QICAS ordering
    # Handle RHF vs UHF
    if mol.spin == 0:
        mo_alpha = mf.mo_coeff
    else:
        mo_alpha = mf.mo_coeff[0]
    n_mo_scan  = mo_alpha.shape[1]
    mo_ordered = mo_alpha[:, mo_order_abs]   # shape (nao, n_mo)

    # Relative indices of window within mo_ordered
    # In mo_order_abs: core | window | other | virt
    n_core     = len(qicas_result["core_indices"])
    # window starts at index n_core in mo_ordered
    win_start  = n_core                        # relative index of window[0]

    scan_data = []

    for n_orb in scan_sizes:
        print(f"\n--- CAS({n_elec_fixed}, {n_orb}) ---")

        # Top-n_orb window orbitals by entropy (window-relative indices)
        active_win_rel = entropy_ranked[:n_orb]    # e.g. [13,12,0,1,...]
        # Convert to position in mo_ordered
        active_mo_rel  = [win_start + r for r in active_win_rel]
        # Remaining window orbitals (not in active)
        rest_win_rel   = [r for r in range(len(window)) if r not in set(active_win_rel)]
        rest_mo_rel    = [win_start + r for r in rest_win_rel]
        # Core and virtual positions in mo_ordered
        core_mo_rel    = list(range(n_core))
        virt_mo_rel    = list(range(win_start + len(window), mo_ordered.shape[1]))

        # Build MO ordering for CASSCF:
        # core (frozen) | active (QICAS top-n_orb) | inactive (rest of window + virt)
        mo_cas_order   = core_mo_rel + active_mo_rel + rest_mo_rel + virt_mo_rel
        mo_cas         = mo_ordered[:, mo_cas_order]

        # Verify electron count consistency
        n_alpha = (n_elec_fixed + spin_2s) // 2
        n_beta  = (n_elec_fixed - spin_2s) // 2
        if n_orb < n_alpha:
            print(f"  [SKIP] n_orb={n_orb} < n_alpha={n_alpha}, skipping")
            continue

        # Run CASSCF
        t0  = time.time()
        mc  = mcscf.CASSCF(mf, n_orb, n_elec_fixed)
        mc.max_cycle_macro = 1
        mc.conv_tol        = 1e-8
        mc.verbose         = 3
        target_ss = spin_2s * (spin_2s + 2) / 4.0
        mc.fix_spin_(ss=target_ss, shift=0.5)
        # Enforce correct spin state — prevents AutoCAS-style spin collapse
        target_ss = spin_2s * (spin_2s + 2) / 4.0
        mc.fix_spin_(ss=target_ss, shift=0.5)
        try:
            e_cas = mc.kernel(mo_cas)[0]
            converged = mc.converged
        except Exception as exc:
            print(f"  [ERROR] CASSCF failed: {exc}")
            scan_data.append({"n_orb": n_orb, "n_elec": n_elec_fixed,
                               "status": "FAILED", "error": str(exc)})
            continue
        t_cas = time.time() - t0

        # Extract NOONs from 1-RDM eigenvalues
        dm1_cas = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)            # shape (n_orb, n_orb), active-MO basis
        noons_raw = np.linalg.eigvalsh(dm1_cas)  # eigenvalues
        noons     = np.sort(noons_raw)[::-1]      # descending

        print(f"  E(CASSCF) = {e_cas:.8f}  converged={converged}  ({t_cas:.1f}s)")
        print(f"  NOONs: {noons.round(4).tolist()}")
        print(f"  Tr(1-RDM) = {noons.sum():.4f}  (should be {n_elec_fixed})")

        scan_data.append({
            "n_orb":     n_orb,
            "n_elec":    n_elec_fixed,
            "label":     f"({n_elec_fixed},{n_orb})",
            "e_casscf":  float(e_cas),
            "converged": bool(converged),
            "noons":     noons.tolist(),
            "t_s":       float(t_cas),
            "status":    "OK",
        })

    with open(out_json, "w") as f:
        json.dump({"name": name, "n_elec_fixed": n_elec_fixed,
                   "n_qicas": n_qicas, "scan": scan_data}, f, indent=2)
    print(f"\n  Saved → {out_json}")
    return scan_data


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Plot
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(scan_json="scan_results.json", qicas_json="qicas_result.json",
              out_prefix="noon_convergence"):
    """
    Reproduce the autoCAS-style NOON convergence figure.

    Top panel  : one curve per active space size, NOON vs orbital index
    Bottom panel: entropy profile from QICAS (reference)
    """
    with open(scan_json) as f:
        scan = json.load(f)
    with open(qicas_json) as f:
        qr  = json.load(f)

    name      = scan["name"]
    n_qicas   = scan.get("n_qicas") or scan.get("n_active_qicas") or scan.get("n_active") or 14
    n_elec    = scan["n_elec_fixed"]
    scan_data = [d for d in scan["scan"] if d["status"] == "OK"]

    if not scan_data:
        print("[WARN] No successful scan points to plot.")
        return

    n_sizes = len(scan_data)

    # Colour map: red for smallest → green for largest (matches original figures)
    colours   = [cm.RdYlGn(i / max(n_sizes - 1, 1)) for i in range(n_sizes)]
    markers   = ["o", "s", "^", "D", "v", "P", "*", "X"][:n_sizes]
    linestyle = "dotted"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9),
                                    gridspec_kw={"height_ratios": [3, 1.5]})
    fig.subplots_adjust(hspace=0.35)

    # ── Top panel: NOON convergence ──────────────────────────────────────────
    ax1.set_title(f"NOON convergence — {name}", fontsize=11, pad=8)
    for i, d in enumerate(scan_data):
        noons = d["noons"]
        xs    = np.arange(1, len(noons) + 1)
        lbl   = d["label"]
        if d["n_orb"] == n_qicas:
            lbl += "  ← QICAS"
        ax1.plot(xs, noons,
                 linestyle=linestyle,
                 marker=markers[i % len(markers)],
                 color=colours[i],
                 label=lbl,
                 linewidth=1.4,
                 markersize=6,
                 zorder=3 + i)

    # Mark the QICAS recommendation size with a vertical dashed line
    ax1.axvline(x=n_qicas, color="grey", linestyle="--", linewidth=0.8,
                label=f"QICAS n_orb={n_qicas}", zorder=1)
    # Mark occupation = 1.0 (SOMO plateau for high-spin)
    ax1.axhline(y=1.0, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
    ax1.axhline(y=0.0, color="black", linestyle=":", linewidth=0.6, alpha=0.5)

    ax1.set_xlabel("Active space orbital index (sorted by NOON, desc)", fontsize=10)
    ax1.set_ylabel("NOON value", fontsize=10)
    ax1.set_ylim(-0.05, 2.05)
    ax1.set_xlim(0.5, max(d["n_orb"] for d in scan_data) + 0.5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.legend(fontsize=8, framealpha=0.85, loc="upper right")
    ax1.grid(True, alpha=0.25)

    # ── Bottom panel: QICAS entropy profile ──────────────────────────────────
    ax2.set_title("QICAS single-orbital entropy profile", fontsize=10)
    entropies = np.array(qr["entropies"])
    ranked    = np.array(qr["entropy_ranked"])
    ent_sorted = entropies[ranked]               # sorted descending
    xs2       = np.arange(1, len(ent_sorted) + 1)

    ax2.bar(xs2, ent_sorted, color="steelblue", alpha=0.75, width=0.7)
    ax2.axvline(x=n_qicas + 0.5, color="red", linestyle="--", linewidth=1.2,
                label=f"QICAS cut (n={n_qicas})")
    ax2.set_xlabel("Window orbital (entropy rank)", fontsize=10)
    ax2.set_ylabel("S(ρ_i)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    # ── Save ─────────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fname = f"{out_prefix}.{ext}"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved → {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system",      default="V_Cl6_chg-2_spin3_oct_d2p394",
                   help="System key in SYSTEMS dict (default: V_Cl6)")
    p.add_argument("--skip-qicas",  action="store_true",
                   help="Skip Phase 1, load existing qicas_result.json")
    p.add_argument("--only-plot",   action="store_true",
                   help="Skip Phases 1+2, replot from existing JSON files")
    p.add_argument("--M",           type=int, default=100,
                   help="DMRG bond dimension for QICAS (default: 100)")
    p.add_argument("--n-below",     type=int, default=4,
                   help="Scan n_qicas - n_below sizes (default: 4)")
    p.add_argument("--n-above",     type=int, default=2,
                   help="Scan n_qicas + n_above sizes (default: 2)")
    p.add_argument("--max-orbs",    type=int, default=16,
                   help="Hard cap on n_orb to avoid memory wall (default: 16)")
    p.add_argument("--qicas-n-active", type=int, default=None,
                   help="Override QICAS n_active from paper (shifts dashed line to match paper)")
    p.add_argument("--scratch",     default="/tmp/noon_scan",
                   help="DMRG scratch directory")
    p.add_argument("--qicas-json",  default="qicas_result.json")
    p.add_argument("--scan-json",   default="scan_results.json")
    p.add_argument("--plot-prefix", default="noon_convergence")
    return p.parse_args()


def main():
    args = parse_args()
    name = args.system

    if name not in SYSTEMS:
        print(f"[ERROR] Unknown system '{name}'. Available: {list(SYSTEMS)}")
        sys.exit(1)
    s = SYSTEMS[name]

    t_total = time.time()

    # Phase 1
    if args.only_plot:
        print("[Skip] Phases 1+2 — loading existing JSON files")
        with open(args.qicas_json) as f:
            qr = json.load(f)
    elif args.skip_qicas:
        print("[Skip] Phase 1 — loading existing QICAS JSON")
        with open(args.qicas_json) as f:
            qr = json.load(f)
    else:
        qr = run_qicas(name, s,
                       out_json=args.qicas_json,
                       M=args.M,
                       scratch=args.scratch)

    # Phase 2
    if not args.only_plot:
        run_casscf_scan(name, s, qr,
                        out_json=args.scan_json,
                        n_below=args.n_below,
                        n_above=args.n_above,
                        scratch=args.scratch,
                        max_orbs=args.max_orbs)

    # Phase 3
    print(f"\n{'='*60}")
    print("  Phase 3 — Plotting")
    print(f"{'='*60}")
    make_plot(scan_json=args.scan_json,
              qicas_json=args.qicas_json,
              out_prefix=args.plot_prefix)

    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()