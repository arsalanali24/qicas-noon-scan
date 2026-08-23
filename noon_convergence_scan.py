#!/usr/bin/env python3
"""
noon_convergence_scan.py — NOON Convergence Scan for QICAS Validation
=====================================================================
Reconstructed from:
  - setup_scan.py        (CLI contract, SLURM generation, system dict shape)
  - qicas_canonical.py   (QICAS core: build_mol, run_hf, run_dmrg, entropies,
                          plateau detection, QICAS rotation)
  - CONTEXT.md           (CRITICAL parameters, corrected window rule, RDM rule,
                          fix_spin_ enforcement, known issues)
  - 10 screenshots       (plot styling, scan ranges, color scheme, axis labels)
  - .gitignore           (output file names: qicas_result.json, scan_results.json,
                          noon_convergence.png/pdf)

Pipeline:
  Phase 1 (QICAS):  UHF → frontier window → low-M DMRG → entropy profile
                    → plateau detection → recommended CAS(n_elec, n_orb)
  Phase 2 (SCAN):   CASCI at sizes n_qicas-N_BELOW to n_qicas+N_ABOVE
                    → extract NOONs from active-space 1-RDM
  Phase 3 (PLOT):   Two-panel convergence plot

CLI contract (from setup_scan.py generate_slurm, lines 231-240):
  python noon_convergence_scan.py \\
      --system_dict '{...}' \\
      [--skip-qicas] \\
      --M 100 --n-below 4 --n-above 2 \\
      --scratch <path> \\
      --qicas-json <path> --scan-json <path> --plot-prefix <path>

CRITICAL (from CONTEXT.md — NEVER CHANGE):
  M = 100 always
  Window: 26 (2S>=4), 24 (2S=2-3), 22 (2S=0-1) — based on spin, NOT n_ligands
  CASCI: fix_spin_(ss=spin_2s*(spin_2s+2)/4.0, shift=0.5)
  NOONs: mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)  NOT mc.make_rdm1()
"""

import os
import sys
import json
import time
import math
import argparse
import traceback
import numpy as np

# ── Source: qicas_canonical.py lines 32-35 ────────────────────────────
from scipy.linalg import expm as scipy_expm
from scipy.optimize import minimize as scipy_minimize
from pyscf import gto, scf, mcscf
from pyscf import dmrgscf
from pyscf.dmrgscf import dmrgci


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

# Source: qicas_canonical.py line 38
METALS_ECP = {'Mo', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt', 'Os', 'Re', 'W', 'Au'}

# Source: CONTEXT.md "CRITICAL — DMRG Parameters" table
#   Window rule: 26 if spin_2s>=4 else (24 if spin_2s>=2 else 22)
#   "NOT based on n_ligands — this was a bug in early versions"
# Contrast: qicas_canonical.py line 167 has the BUGGY rule:
#   win=26 if s['spin_2s']>=5 else (24 if s['n_ligands']>=6 else 20)
DMRG_PARAMS = {
    'HIGH':   {'spin_min': 4, 'window': 26, 'M': 100, 'sweeps': 30},
    'MEDIUM': {'spin_min': 2, 'window': 24, 'M': 100, 'sweeps': 30},
    'LOW':    {'spin_min': 0, 'window': 22, 'M': 100, 'sweeps': 35},
}


def spin_category(spin_2s):
    """Source: setup_scan.py get_spin_category (lines 67-69)"""
    if spin_2s >= 4:
        return 'HIGH'
    if spin_2s >= 2:
        return 'MEDIUM'
    return 'LOW'


# ═══════════════════════════════════════════════════════════════════════
#  Geometry builders — Source: qicas_canonical.py lines 40-59
# ═══════════════════════════════════════════════════════════════════════

def _oct(d):
    return [(d, 0, 0), (-d, 0, 0), (0, d, 0), (0, -d, 0), (0, 0, d), (0, 0, -d)]

def _tet(d):
    c = d / math.sqrt(3)
    return [(c, c, c), (-c, -c, c), (-c, c, -c), (c, -c, -c)]

def _sqpl(d):
    return [(d, 0, 0), (-d, 0, 0), (0, d, 0), (0, -d, 0)]

GEOM_BUILDERS = {'oct': _oct, 'tet': _tet, 'sq_pl': _sqpl, 'sqpl': _sqpl, 'sq': _sqpl}

# Source: qicas_canonical.py lines 49-59
# NOTE: setup_scan.py only uses simple ligands (Cl, Br, F, O, N)
# but we keep the full table from qicas_canonical.py for compatibility
LIGAND_ATOMS = {
    'Cl': [('Cl', 1)], 'Br': [('Br', 1)], 'F': [('F', 1)], 'I': [('I', 1)],
    'H': [('H', 1)], 'O': [('O', 1)], 'N': [('N', 1)], 'S': [('S', 1)],
    'C': [('C', 1)],
    'CN': [('C', 1), ('N', 1)],
    'NH3': [('N', 1), ('H', 3)],
    'H2O': [('O', 1), ('H', 2)],
    'PH3': [('P', 1), ('H', 3)],
}


# ═══════════════════════════════════════════════════════════════════════
#  Molecule and HF — Source: qicas_canonical.py lines 111-158
# ═══════════════════════════════════════════════════════════════════════

def build_mol(s):
    """Build PySCF molecule from system dictionary.

    Source: qicas_canonical.py build_mol (lines 111-131)
    Adaptation: setup_scan.py uses key 'ligand' not 'ligand_raw'
    """
    dist = s.get('dist_ang') or 2.10
    geometry = s.get('geometry', 'oct')
    builder = GEOM_BUILDERS.get(geometry, _oct)
    n_ligands = s.get('n_ligands', 6)
    lig_pos = builder(dist)[:n_ligands]

    atoms = [(s['metal'], (0., 0., 0.))]
    # setup_scan.py uses 'ligand'; qicas_canonical.py uses 'ligand_raw'
    ligand_key = s.get('ligand_raw', s.get('ligand', 'Cl'))
    tmpl = LIGAND_ATOMS.get(ligand_key, [('Cl', 1)])

    for pos in lig_pos:
        r = math.sqrt(sum(x**2 for x in pos))
        uv = tuple(x / r for x in pos) if r > 0 else (1, 0, 0)
        off = 0.0
        for sym, nrep in tmpl:
            for _ in range(nrep):
                coord = tuple(pos[i] + uv[i] * off for i in range(3))
                atoms.append((sym, coord))
                off += 1.10

    mol = gto.Mole()
    mol.atom = atoms
    mol.charge = s['charge']
    mol.spin = s['spin_2s']
    mol.basis = 'def2-svp'
    mol.unit = 'angstrom'
    mol.symmetry = False
    if s.get('needs_ecp', s['metal'] in METALS_ECP):
        mol.ecp = {s['metal']: 'def2-svp'}
    mol.verbose = 4
    mol.build()
    return mol


def run_hf(mol):
    """Run HF — RHF for singlets, UHF otherwise.

    Source: qicas_canonical.py run_hf (lines 134-158)
    Note: "UHF rebuilds in Phase 2" (CONTEXT.md Known Issues) is expected.
    """
    if mol.spin == 0:
        mf = scf.RHF(mol)
        print("  [HF] Using RHF (singlet — prevents broken-symmetry UHF)")
    else:
        mf = scf.UHF(mol)
        print(f"  [HF] Using UHF (spin_2s={mol.spin})")

    mf.max_cycle = 500
    mf.conv_tol = 1e-10
    mf.init_guess = 'atom'
    mf.level_shift = 0.2

    try:
        mf.kernel()
    except Exception:
        mf.init_guess = 'minao'
        mf.kernel()
    if not mf.converged:
        mf.level_shift = 0.5
        mf.kernel()
    if not mf.converged:
        mf.init_guess = 'minao'
        mf.level_shift = 0.3
        mf.damp = 0.5
        mf.kernel()
    if not mf.converged:
        mf.init_guess = '1e'
        mf.level_shift = 0.5
        mf.damp = 0.3
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("HF did not converge after all attempts")
    return mf


# ═══════════════════════════════════════════════════════════════════════
#  Window selection — CORRECTED rule from CONTEXT.md
# ═══════════════════════════════════════════════════════════════════════

def select_window(mf, spin_2s):
    """Select frontier orbital window for DMRG.

    CORRECTED rule (source: CONTEXT.md "CRITICAL — DMRG Parameters"):
      window = 26 if spin_2s >= 4 else (24 if spin_2s >= 2 else 22)

    BUGGY rule in qicas_canonical.py line 167 was:
      win = 26 if s['spin_2s'] >= 5 else (24 if s['n_ligands'] >= 6 else 20)
    Differences: threshold 4 vs 5, spin-based vs n_ligands-based, 22 vs 20.

    Centering and singly-occupied inclusion logic from qicas_canonical.py L163-174.
    """
    cat = spin_category(spin_2s)
    win = DMRG_PARAMS[cat]['window']

    if isinstance(mf, scf.uhf.UHF):
        n_mo = mf.mo_coeff[0].shape[1]
        n_a = int(mf.mo_occ[0].sum())
        n_b = int(mf.mo_occ[1].sum())
    else:
        n_mo = mf.mo_coeff.shape[1]
        n_a = int((mf.mo_occ > 0).sum())
        n_b = n_a

    # Center window on HOMO (same logic as qicas_canonical.py L168-170)
    half = win // 2
    start = max(0, n_a - half)
    end = min(n_mo, start + win)
    if end - start < win:
        start = max(0, end - win)
    window = list(range(start, end))

    # Ensure all singly-occupied orbitals are included (L171-173)
    for idx in range(n_b, n_a):
        if idx not in window:
            window = list(range(min(window[0], idx), max(window[-1] + 1, idx + 1)))

    return window


# ═══════════════════════════════════════════════════════════════════════
#  DMRG and entropy — Source: qicas_canonical.py lines 177-264
# ═══════════════════════════════════════════════════════════════════════

def run_dmrg(mol, mf, window, M=100, nsweeps=30, scratch='/tmp/dmrg'):
    """Source: qicas_canonical.py run_dmrg (lines 177-202)"""
    os.makedirs(scratch, exist_ok=True)

    if isinstance(mf, scf.uhf.UHF):
        occ = (mf.mo_occ[0] + mf.mo_occ[1]) / 2.0
        mo_alpha = mf.mo_coeff[0]
    else:
        occ = mf.mo_occ / 2.0
        mo_alpha = mf.mo_coeff

    # Electron count in window (with spin-parity enforcement)
    n_e = int(round(2 * occ[window].sum()))
    if (n_e - mol.spin) % 2 != 0:
        n_e += 1
    if (n_e - mol.spin) % 2 != 0:
        n_e -= 2
    n_e = int(np.clip(n_e, mol.spin, 2 * len(window)))

    # Reorder MOs: core | window | other | virtual
    n_mo = mo_alpha.shape[1]
    all_i = list(range(n_mo))
    core = sorted([i for i in all_i if i not in window and occ[i] > 1.5])
    virt = sorted([i for i in all_i if i not in window and occ[i] < 0.5])
    other = sorted([i for i in all_i if i not in window
                     and i not in core and i not in virt])
    mo_ord = mo_alpha[:, core + window + other + virt]

    mc = mcscf.CASSCF(mf.to_rhf(), len(window), n_e)
    mc.fcisolver = dmrgci.DMRGCI(mol, maxM=M, tol=1e-8)
    mc.fcisolver.scratchDirectory = scratch
    mc.fcisolver.runtimeDir = scratch
    mc.fcisolver.maxIter = nsweeps
    mc.fcisolver.block_extra_keyword = ['num_thrds 8']
    # CONTEXT.md: "max_cycle_macro=1 is intentional"
    # "converged=False is Expected"
    mc.max_cycle_macro = 1
    e = mc.kernel(mo_ord)[0]

    return mc, float(e), mo_ord, n_e


def get_rdms(mc):
    """Extract 1- and 2-RDMs from the DMRG solver.

    CRITICAL (source: CONTEXT.md "CRITICAL — RDM Extraction"):
      Use mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
      NOT mc.make_rdm12() which returns full AO-basis RDM.

    Source: qicas_canonical.py get_rdms (lines 210-219)
    """
    try:
        dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
        print(f"  [RDM] make_rdm12 OK  Tr(1-RDM) = {np.trace(dm1):.4f}")
        return dm1, dm2
    except Exception as e:
        print(f"  [RDM] MF approximation for 2-RDM ({e})")
        dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        n = mc.ncas
        dm2 = np.zeros((n, n, n, n))
        for i in range(n):
            dm2[i, i, i, i] = (dm1[i, i] / 2)**2 * 4
        return dm1, dm2


def _single_orbital_entropy(n_i, G_ii):
    """Source: qicas_canonical.py _S (lines 205-207)"""
    lam = np.clip([1 - 2*n_i + G_ii, n_i - G_ii, n_i - G_ii, G_ii], 1e-14, 1.0)
    lam /= lam.sum()
    return -float(np.dot(lam, np.log(lam)))


def entropies_from_rdms(gamma, Gamma, n):
    """Source: qicas_canonical.py entropies_from_rdms (lines 222-223)"""
    return np.array([_single_orbital_entropy(gamma[i, i] / 2, Gamma[i, i, i, i] / 4)
                     for i in range(n)])


def entropy_plateau_cas_size(ent, spin_2s, floor=0.02, min_act=2):
    """Find CAS size from largest gap in the entropy profile.

    Source: qicas_canonical.py entropy_plateau_cas_size (lines 226-240)
    """
    ss = np.sort(ent)[::-1]
    n_sig = int(np.sum(ss > floor))
    if n_sig <= max(spin_2s, min_act):
        return max(spin_2s, min_act)
    gaps = ss[:-1] - ss[1:]
    gaps_restricted = gaps[:n_sig]
    d = int(np.argmax(gaps_restricted)) + 1
    return int(np.clip(max(d, spin_2s, min_act), 2, n_sig))


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: Run QICAS — build mol → HF → DMRG → entropies → plateau
# ═══════════════════════════════════════════════════════════════════════

def run_qicas_phase(system, M=100, sweeps=30, scratch='/tmp/dmrg'):
    """Phase 1: Determine the recommended active space via QICAS entropy
    plateau detection.

    Returns a dict with:
      - n_active_qicas, n_elec_qicas  (consumed by setup_scan.py L306/325)
      - entropy_profile               (needed for bottom panel plot)
      - _internal: mol, mf, mo_ord, window, ent_rank  (for same-session Phase 2)
    """
    name = system.get('name', 'unknown')
    spin_2s = system['spin_2s']
    cat = spin_category(spin_2s)
    params = DMRG_PARAMS[cat]

    print(f"\n{'='*60}")
    print(f"  Phase 1: QICAS — {name}")
    print(f"  Spin: 2S={spin_2s}  Category: {cat}")
    print(f"  DMRG: M={M}, sweeps={sweeps}, window={params['window']}")
    print(f"{'='*60}")

    t0 = time.time()

    # Step 1: Build molecule
    print("\n[Step 1] Building molecule...")
    mol = build_mol(system)
    n_electrons = int(mol.nelectron)
    n_basis = int(mol.nao_nr())
    print(f"  {n_electrons} electrons, {n_basis} basis functions")

    # Step 2: Hartree-Fock
    print("\n[Step 2] Hartree-Fock...")
    t1 = time.time()
    mf = run_hf(mol)
    e_hf = float(mf.e_tot)
    print(f"  E(HF) = {e_hf:.8f}  ({time.time()-t1:.1f}s)")

    # Step 3: Select frontier window (CORRECTED rule)
    print("\n[Step 3] Selecting frontier window...")
    window = select_window(mf, spin_2s)
    print(f"  Window: {len(window)} orbitals, indices [{window[0]}..{window[-1]}]")

    # Step 4: DMRG
    print(f"\n[Step 4] DMRG (M={M}, {sweeps} sweeps)...")
    t1 = time.time()
    mc_dmrg, e_dmrg, mo_ord, n_elec_win = run_dmrg(
        mol, mf, window, M=M, nsweeps=sweeps, scratch=scratch)
    t_dmrg = time.time() - t1
    print(f"  E(DMRG) = {e_dmrg:.8f}  ({t_dmrg:.1f}s)")
    print(f"  Electrons in window: {n_elec_win}")

    # Step 5: Extract RDMs and compute single-orbital entropies
    print("\n[Step 5] RDMs and single-orbital entropies...")
    gamma, Gamma = get_rdms(mc_dmrg)
    n_win = len(window)
    ent = entropies_from_rdms(gamma, Gamma, n_win)
    ent_rank = np.argsort(ent)[::-1]           # indices sorted by entropy desc
    ent_sorted = ent[ent_rank]                  # entropy values sorted desc
    print(f"  Entropy range: [{ent.min():.4f}, {ent.max():.4f}]")

    # Step 6: Find CAS size from entropy plateau
    d_cas = entropy_plateau_cas_size(ent, spin_2s)
    print(f"\n[Step 6] Entropy plateau → D_CAS = {d_cas}")
    print(f"  QICAS recommendation: CAS({n_elec_win}, {d_cas})")

    t_total = time.time() - t0

    result = {
        # Required by setup_scan.py list_completed (L306) and resume_from_json (L325)
        'name': name,
        'n_elec_qicas': int(n_elec_win),
        'n_active_qicas': int(d_cas),
        'spin_2s': spin_2s,
        # Metadata for cross-session resume
        'metal': system.get('metal'),
        'ligand': system.get('ligand_raw', system.get('ligand')),
        'charge': system.get('charge'),
        'geometry': system.get('geometry'),
        'n_ligands': system.get('n_ligands'),
        'dist_ang': system.get('dist_ang'),
        'needs_ecp': system.get('needs_ecp', system.get('metal', '') in METALS_ECP),
        # Computational details
        'n_electrons': n_electrons,
        'n_basis': n_basis,
        'e_hf': e_hf,
        'e_dmrg': e_dmrg,
        'spin_category': cat,
        'window_size': n_win,
        'window_indices': window,
        'n_elec_window': int(n_elec_win),
        # Entropy profile (needed for bottom panel plot)
        'entropy_profile': {
            'entropies_sorted_desc': ent_sorted.tolist(),
            'entropies_raw': ent.tolist(),
            'entropy_rank': ent_rank.tolist(),
        },
        't_dmrg_s': t_dmrg,
        't_total_s': t_total,
        'status': 'OK',
    }

    # Internal objects for same-session Phase 2 (NOT serialized)
    result['_internal'] = {
        'mol': mol, 'mf': mf, 'mo_ord': mo_ord,
        'window': window, 'ent_rank': ent_rank,
    }

    print(f"\n  Phase 1 complete in {t_total:.0f}s")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: NOON Scan — CASCI at multiple CAS sizes
# ═══════════════════════════════════════════════════════════════════════

def build_scan_mo(mo_ord, window, active_indices, n_mo, mf):
    """Reorder MO matrix placing selected active orbitals in CAS position.

    Source: qicas_canonical.py build_cas_mo (lines 267-281)
    active_indices: window-relative indices, sorted by entropy descending
    """
    if isinstance(mf, scf.uhf.UHF):
        occ = (mf.mo_occ[0] + mf.mo_occ[1]) / 2.0
    else:
        occ = mf.mo_occ / 2.0

    all_i = list(range(n_mo))
    abs_act = [window[r] for r in active_indices]
    nonact_win = [window[r] for r in range(len(window)) if r not in active_indices]

    core_nw = sorted([i for i in all_i if i not in window and occ[i] > 1.5])
    virt_nw = sorted([i for i in all_i if i not in window and occ[i] < 0.5])
    core_win = sorted([i for i in nonact_win if occ[i] > 1.5])
    virt_win = sorted([i for i in nonact_win if occ[i] < 0.5])
    other = [i for i in all_i
             if i not in sorted(core_nw + virt_nw + core_win + virt_win + abs_act)]

    ordered = sorted(core_nw + core_win) + abs_act + other + sorted(virt_nw + virt_win)
    return mo_ord[:, np.argsort(ordered)]


def run_casci_noon(mf, mo_coeffs, n_cas, n_elec, spin_2s):
    """Run CASCI for one CAS(n_elec, n_cas) and extract NOONs.

    CRITICAL — Spin enforcement (source: CONTEXT.md):
      mc.fix_spin_(ss=spin_2s*(spin_2s+2)/4.0, shift=0.5)
      "Without this, CASCI collapses to wrong spin state at larger active
       space sizes. Symptom: S² jumps (e.g. 8.75 → 10.86 for sextet)."

    CRITICAL — RDM extraction (source: CONTEXT.md):
      NOONs from mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
      NOT mc.make_rdm1() — that returns full AO-basis RDM (Tr ≈ N_total).

    Expected log line (source: CONTEXT.md "Checking Results"):
      Tr(1-RDM) = 21.0000   ← must equal n_active_e

    Returns result dict or None on failure.
    """
    try:
        mc = mcscf.CASCI(mf.to_rhf(), n_cas, n_elec)
        mc.verbose = 4

        # CRITICAL: enforce spin state
        ss_target = spin_2s * (spin_2s + 2) / 4.0
        mc.fix_spin_(ss=ss_target, shift=0.5)

        # CONTEXT.md: "max_cycle_macro=1 is intentional"
        # "converged=False — Expected"
        mc.max_cycle_macro = 1
        e = mc.kernel(mo_coeffs)[0]

        # CRITICAL: active-space RDM, NOT full AO RDM
        rdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        tr_rdm1 = float(np.trace(rdm1))

        # Verify trace (CONTEXT.md: "Tr(1-RDM) = 21.0000 ← must equal n_active_e")
        if abs(tr_rdm1 - n_elec) > 0.5:
            print(f"    ⚠ Tr(1-RDM) = {tr_rdm1:.4f}, expected {n_elec}")

        # NOONs = eigenvalues of 1-RDM, sorted descending
        noons = np.sort(np.linalg.eigvalsh(rdm1))[::-1]

        # S² (CONTEXT.md: "S^2 = X.75 ← must be consistent throughout")
        try:
            s2 = float(mc.spin_square()[0])
        except Exception:
            s2 = None

        return {
            'n_cas': n_cas,
            'n_elec': n_elec,
            'energy': float(e),
            'noons': noons.tolist(),
            'tr_rdm1': tr_rdm1,
            's2': s2,
        }

    except Exception as exc:
        print(f"    ✗ CAS({n_elec},{n_cas}) FAILED: {exc}")
        traceback.print_exc()
        return None


def run_noon_scan(qicas_result, n_below=4, n_above=2):
    """Phase 2: CASCI at multiple CAS sizes around the QICAS recommendation.

    Scan range (verified against all 5 screenshot systems):
      lower = max(n_qicas - n_below,  (n_elec + spin_2s + 1) // 2)
      upper = min(n_qicas + n_above,  window_size)

    The lower bound ensures enough orbitals to hold all alpha electrons:
      n_alpha = (n_elec + spin_2s) / 2

    Evidence from screenshots:
      VCl6     2S=3 QICAS=14  → sizes 12-16  (min=12=(21+3)/2)
      MnCl4_s3 2S=3 QICAS=13  → sizes 12-15  (min=12)
      MnCl4_s5 2S=5 QICAS=14  → sizes 13-16  (min=13=(21+5)/2)
      MnBr4_s3 2S=3 QICAS=16  → sizes 12-16  (17,18 failed — skipped)
      MnBr4_s5 2S=5 QICAS=16  → sizes 13-17  (18 failed — skipped)

    Source: CONTEXT.md "UHF rebuilds in Phase 2 — Expected"
    """
    name = qicas_result['name']
    spin_2s = qicas_result['spin_2s']
    n_elec = qicas_result['n_elec_qicas']
    n_qicas = qicas_result['n_active_qicas']
    internal = qicas_result.get('_internal')

    print(f"\n{'='*60}")
    print(f"  Phase 2: NOON Scan — {name}")
    print(f"  QICAS: CAS({n_elec}, {n_qicas})")
    print(f"  Range: n_qicas-{n_below} to n_qicas+{n_above}")
    print(f"{'='*60}")

    # ── Get mol / mf / mo_ord ────────────────────────────────────────
    if internal:
        mol = internal['mol']
        mf = internal['mf']
        mo_ord = internal['mo_ord']
        window = internal['window']
        ent_rank = internal['ent_rank']
        print("  Using Phase 1 objects (same session)")
    else:
        # Cross-session resume: rebuild molecule and HF independently
        # CONTEXT.md: "UHF rebuilds in Phase 2 — Expected"
        print("  Rebuilding molecule and HF (cross-session resume)")
        system = {
            'name': name,
            'metal': qicas_result.get('metal'),
            'ligand': qicas_result.get('ligand'),
            'charge': qicas_result.get('charge'),
            'spin_2s': spin_2s,
            'geometry': qicas_result.get('geometry'),
            'n_ligands': qicas_result.get('n_ligands'),
            'dist_ang': qicas_result.get('dist_ang'),
            'needs_ecp': qicas_result.get('needs_ecp', False),
        }
        mol = build_mol(system)
        mf = run_hf(mol)
        window = select_window(mf, spin_2s)
        ent_rank = np.array(qicas_result['entropy_profile']['entropy_rank'])

        # Rebuild mo_ord from HF (same reorder as run_dmrg)
        if isinstance(mf, scf.uhf.UHF):
            mo_alpha = mf.mo_coeff[0]
            occ = (mf.mo_occ[0] + mf.mo_occ[1]) / 2.0
        else:
            mo_alpha = mf.mo_coeff
            occ = mf.mo_occ / 2.0
        n_mo = mo_alpha.shape[1]
        all_i = list(range(n_mo))
        core = sorted([i for i in all_i if i not in window and occ[i] > 1.5])
        virt = sorted([i for i in all_i if i not in window and occ[i] < 0.5])
        other = sorted([i for i in all_i if i not in window
                         and i not in core and i not in virt])
        mo_ord = mo_alpha[:, core + window + other + virt]

    n_mo = mo_ord.shape[1]
    window_size = len(window)

    # ── Compute scan range ───────────────────────────────────────────
    # Minimum: must fit all alpha electrons
    n_alpha = (n_elec + spin_2s + 1) // 2
    min_n_orb = max(n_alpha, 2)
    max_n_orb = window_size

    scan_start = max(n_qicas - n_below, min_n_orb)
    scan_end = min(n_qicas + n_above, max_n_orb)
    scan_sizes = list(range(scan_start, scan_end + 1))

    print(f"  n_alpha = {n_alpha}  →  valid range [{min_n_orb}, {max_n_orb}]")
    print(f"  Scanning sizes: {scan_sizes}")

    # ── Run CASCI at each size ───────────────────────────────────────
    scan_results = {
        'name': name,
        'spin_2s': spin_2s,
        'n_elec': n_elec,
        'n_qicas': n_qicas,
        'scan_sizes_attempted': scan_sizes,
        'results': {},
    }

    for n_orb in scan_sizes:
        print(f"\n  ── CAS({n_elec}, {n_orb}) "
              f"{'← QICAS' if n_orb == n_qicas else ''} ──")

        # Top n_orb orbitals by entropy
        active_rel = ent_rank[:n_orb].tolist()

        # Build MO matrix with these as active
        mo_scan = build_scan_mo(mo_ord, window, active_rel, n_mo, mf)

        # CASCI with spin enforcement
        t1 = time.time()
        res = run_casci_noon(mf, mo_scan, n_orb, n_elec, spin_2s)
        dt = time.time() - t1

        if res is not None:
            res['time_s'] = dt
            scan_results['results'][str(n_orb)] = res
            noons = res['noons']
            s2_str = f"  S²={res['s2']:.4f}" if res['s2'] is not None else ""
            print(f"    ✓ E={res['energy']:.8f}  Tr(1-RDM)={res['tr_rdm1']:.4f}"
                  f"{s2_str}  ({dt:.1f}s)")
            # CONTEXT.md: "NOONs: all between 0 and 2"
            noon_str = ' '.join(f'{x:.3f}' for x in noons[:min(8, len(noons))])
            if len(noons) > 8:
                noon_str += '...'
            print(f"    NOONs: {noon_str}")
        else:
            print(f"    ✗ Skipping ({dt:.1f}s)")

    n_ok = len(scan_results['results'])
    scan_results['scan_sizes_completed'] = sorted(int(k) for k in scan_results['results'])
    scan_results['n_completed'] = n_ok
    print(f"\n  Phase 2 complete: {n_ok}/{len(scan_sizes)} sizes converged")

    return scan_results


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3: Plotting — Style matched to 10 screenshots
# ═══════════════════════════════════════════════════════════════════════

def plot_convergence(scan_results, qicas_result, plot_prefix):
    """Generate two-panel NOON convergence plot.

    Plot format verified against 10 screenshots (5 systems × 2 copies):

    Top panel:
      Title:  "NOON convergence — {name}"
      X-axis: "Active space orbital index (sorted by NOON, desc)"
      Y-axis: "NOON value"
      Colors: Sequential red→yellow→green (RdYlGn), one per CAS size
      Markers: cycle through o, s, ^, D, v
      Lines:  dotted for all sizes
      QICAS:  gray dashed vertical line at n_qicas
      Legend: "(n_elec, n_orb)  ← QICAS" for the QICAS size

    Bottom panel:
      Title:  "QICAS single-orbital entropy profile"
      X-axis: "Window orbital (entropy rank)"
      Y-axis: "S(ρ_i)"
      Bars:   steelblue
      Cut:    red dashed vertical line, labeled "QICAS cut (n=X)"
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    name = scan_results['name']
    n_elec = scan_results['n_elec']
    n_qicas = scan_results['n_qicas']
    ent_sorted = qicas_result['entropy_profile']['entropies_sorted_desc']

    # Collect completed sizes in order
    completed = {}
    for n_orb_str, res in scan_results['results'].items():
        completed[int(n_orb_str)] = res['noons']
    sizes = sorted(completed.keys())

    if not sizes:
        print("  ✗ No completed sizes to plot")
        return

    n = len(sizes)

    # ── Color scheme: sample RdYlGn colormap ─────────────────────────
    # Verified from screenshots: smallest=dark red, middle=yellow, largest=dark green
    cmap = plt.cm.RdYlGn
    if n == 1:
        colors = [cmap(0.5)]
    else:
        # Sample from 0.05 to 0.95 to avoid extreme ends
        colors = [cmap(0.05 + 0.9 * i / (n - 1)) for i in range(n)]

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
    linestyles = [':'] * n  # all dotted (verified from screenshots)

    # ── Figure layout (verified from screenshots: ~2:1 height ratio) ─
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={'height_ratios': [2, 1]},
    )
    fig.subplots_adjust(hspace=0.35)

    # ── Top panel: NOON convergence ──────────────────────────────────
    for idx, n_orb in enumerate(sizes):
        noons = completed[n_orb]
        x = np.arange(1, len(noons) + 1)

        label = f'({n_elec},{n_orb})'
        if n_orb == n_qicas:
            label += '  \u2190 QICAS'    # ← QICAS

        ax1.plot(x, noons,
                 linestyle=linestyles[idx],
                 marker=markers[idx % len(markers)],
                 color=colors[idx],
                 markersize=7, linewidth=1.5,
                 label=label, zorder=3)

    # QICAS vertical reference line
    ax1.axvline(x=n_qicas, color='gray', linestyle='--', linewidth=1,
                label=f'QICAS n_orb={n_qicas}', zorder=1)

    # Horizontal reference at NOON = 1.0
    ax1.axhline(y=1.0, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

    ax1.set_xlabel('Active space orbital index (sorted by NOON, desc)', fontsize=11)
    ax1.set_ylabel('NOON value', fontsize=11)
    ax1.set_title(f'NOON convergence \u2014 {name}', fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.05, 2.10)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)

    # ── Bottom panel: QICAS entropy profile ──────────────────────────
    n_ent = len(ent_sorted)
    x_ent = np.arange(n_ent)

    ax2.bar(x_ent, ent_sorted, color='steelblue', alpha=0.8, width=0.8)
    ax2.axvline(x=n_qicas - 0.5, color='red', linestyle='--', linewidth=2,
                label=f'QICAS cut (n={n_qicas})')

    ax2.set_xlabel('Window orbital (entropy rank)', fontsize=11)
    ax2.set_ylabel(r'$S(\rho_i)$', fontsize=11)
    ax2.set_title('QICAS single-orbital entropy profile', fontsize=12)
    ax2.set_ylim(0, max(ent_sorted) * 1.15 if ent_sorted else 1.0)
    ax2.legend(loc='upper right', fontsize=10)

    # ── Save ─────────────────────────────────────────────────────────
    # Output files verified from .gitignore: noon_convergence.png, .pdf
    png_path = f'{plot_prefix}.png'
    pdf_path = f'{plot_prefix}.pdf'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {png_path}")
    print(f"  Plot saved: {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _json_safe(obj):
    """Make numpy types JSON-serializable.
    Source: qicas_canonical.py _j (lines 299-305)"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════
#  Main — CLI contract from setup_scan.py generate_slurm (lines 231-240)
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='NOON Convergence Scan for QICAS Active Space Validation')

    # Arguments exactly as generated by setup_scan.py generate_slurm
    p.add_argument('--system_dict', type=str, required=True,
                   help='JSON string with system definition')
    p.add_argument('--skip-qicas', action='store_true',
                   help='Skip Phase 1, load existing qicas-json')
    p.add_argument('--M', type=int, default=100,
                   help='DMRG bond dimension (ALWAYS 100)')
    p.add_argument('--n-below', type=int, default=4,
                   help='Sizes below QICAS to scan')
    p.add_argument('--n-above', type=int, default=2,
                   help='Sizes above QICAS to scan')
    p.add_argument('--scratch', type=str,
                   default='/scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch/default',
                   help='DMRG scratch directory')
    p.add_argument('--qicas-json', type=str, required=True,
                   help='Path for QICAS result JSON')
    p.add_argument('--scan-json', type=str, required=True,
                   help='Path for scan results JSON')
    p.add_argument('--plot-prefix', type=str, required=True,
                   help='Prefix for plot files (.png/.pdf appended)')

    args = p.parse_args()

    # Parse the system dict passed as JSON string
    system = json.loads(args.system_dict)
    name = system.get('name', 'unknown')
    spin_2s = system['spin_2s']
    cat = spin_category(spin_2s)
    params = DMRG_PARAMS[cat]

    print(f"\n{'='*60}")
    print(f"  NOON Convergence Scan")
    print(f"  System:  {name}")
    print(f"  Spin:    2S={spin_2s} ({cat})")
    print(f"  DMRG:    M={args.M}, window={params['window']}")
    print(f"  Scan:    n_below={args.n_below}, n_above={args.n_above}")
    print(f"  Scratch: {args.scratch}")
    print(f"{'='*60}")

    # Ensure output directories exist
    for path in [args.qicas_json, args.scan_json, args.plot_prefix]:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)

    # ── Phase 1: QICAS ───────────────────────────────────────────────
    if args.skip_qicas:
        print(f"\n  --skip-qicas: Loading {args.qicas_json}")
        with open(args.qicas_json) as f:
            qicas_result = json.load(f)
        print(f"  Loaded: CAS({qicas_result['n_elec_qicas']}, "
              f"{qicas_result['n_active_qicas']})")
    else:
        sweeps = params['sweeps']
        qicas_result = run_qicas_phase(
            system, M=args.M, sweeps=sweeps, scratch=args.scratch)

        # Save (strip internal PySCF objects)
        qicas_save = {k: v for k, v in qicas_result.items()
                      if not k.startswith('_')}
        with open(args.qicas_json, 'w') as f:
            json.dump(_json_safe(qicas_save), f, indent=2)
        print(f"\n  Saved: {args.qicas_json}")

    # ── Phase 2: NOON Scan ───────────────────────────────────────────
    scan_results = run_noon_scan(
        qicas_result, n_below=args.n_below, n_above=args.n_above)

    with open(args.scan_json, 'w') as f:
        json.dump(_json_safe(scan_results), f, indent=2)
    print(f"\n  Saved: {args.scan_json}")

    # ── Phase 3: Plot ────────────────────────────────────────────────
    print("\n[Phase 3] Generating convergence plot...")
    try:
        plot_convergence(scan_results, qicas_result, args.plot_prefix)
    except Exception as exc:
        print(f"  ⚠ Plot generation failed: {exc}")
        print("  (JSON results saved — plot can be regenerated)")
        traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────
    n_ok = scan_results.get('n_completed', 0)
    n_att = len(scan_results.get('scan_sizes_attempted', []))
    print(f"\n{'='*60}")
    print(f"  Done — {name}")
    print(f"  QICAS:  CAS({qicas_result['n_elec_qicas']}, "
          f"{qicas_result['n_active_qicas']})")
    print(f"  Scanned: {n_ok}/{n_att} sizes converged")
    if n_ok < n_att:
        ok = set(scan_results.get('scan_sizes_completed', []))
        failed = [s for s in scan_results.get('scan_sizes_attempted', [])
                  if s not in ok]
        print(f"  Failed:  {failed}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
