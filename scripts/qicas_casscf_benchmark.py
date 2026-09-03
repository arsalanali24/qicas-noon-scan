"""
qicas_casscf_benchmark.py
=========================
QICAS + CASCI and QICAS + CASSCF pipeline — BENCHMARK VERSION.

Extends qicas_casscf_16systems.py with:
  --xyz   : read optimized geometry from XYZ file (PBE0/def2-TZVP)
            instead of ideal parametric geometry.
  --save_mo: save QICAS-rotated MO matrix as .npy for use in spin-gap
             benchmark calculations.

This version is used for the spin-state gap benchmark where geometry
must match the DFT-optimized structure for comparison with experiment.

QICAS + CASCI and QICAS + CASSCF pipeline for 16 transition-metal complexes.

Two goals, run in sequence for each system:

  Goal 1 — Active space proposal (QICAS → CASCI)
      DMRG on frontier orbital window → entropy-guided orbital rotation →
      active space determined from entropy plateau → CASCI validation
      (QICAS-rotated orbitals vs plain HF orbitals at the same CAS size).

  Goal 2 — Warm-start efficiency (QICAS → CASSCF)
      Feed QICAS-rotated orbitals into CASSCF at the QICAS-determined CAS
      size.  Run a second CASSCF from plain HF orbitals at the same CAS.
      Both must converge to the same energy; the comparison reveals whether
      QICAS orbitals accelerate convergence.

autoCAS results are NOT fed in — QICAS runs fully blind.
autoCAS numbers are stored in the output JSON for post-hoc comparison only.

Usage (single system, for testing):
    python qicas_casscf_16systems.py --system CSD_CrCl4_2m_tet_spin4

Usage (all systems, via SLURM array):
    python qicas_casscf_16systems.py --system_index $SLURM_ARRAY_TASK_ID

Requires:
    PySCF with dmrgscf / block2 backend.
    Activate environment before running:
        source ~/.block2_fix/block2_env.sh

Output:
    results/qicas_casscf_<system_name>.json   (one file per system)
"""

import os
import sys
import json
import math
import time
import argparse
import traceback
import numpy as np

# ── PySCF imports ─────────────────────────────────────────────────────────

from pyscf import gto, scf, mcscf
from pyscf import dmrgscf
from pyscf.dmrgscf import dmrgci

# ── System definitions ────────────────────────────────────────────────────

def _tet(d):
    c = d / math.sqrt(3)
    return [(c,c,c), (-c,-c,c), (-c,c,-c), (c,-c,-c)]

def _oct(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0),(0,0,d),(0,0,-d)]

def _sqpl(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0)]

GEOMETRY_BUILDERS = {'tet':_tet, 'oct':_oct, 'sqpl':_sqpl, 'sq_pl':_sqpl}

DIST_TABLE = {
    ('Cr','Cl','tet'):2.24, ('Cr','Cl','oct'):2.34,
    ('Mn','Cl','tet'):2.35, ('Mn','Cl','oct'):2.48,
    ('Mn','Br','tet'):2.50, ('Mn','F','oct'): 1.98,
    ('Fe','Cl','tet'):2.19, ('Fe','Cl','oct'):2.38,
    ('Co','Cl','oct'):2.44,
    ('Ni','Cl','sqpl'):2.20,('Ni','Cl','oct'):2.40,
}

# Each system: (metal, ligand, n_ligands, charge, spin_2s, geometry,
#               dist_ang_or_None, metal_row, autocas_cas_ne, autocas_cas_no)
# autocas_* are stored for post-hoc comparison ONLY — not used by QICAS.
SYSTEMS = {
    # ── 3d ──────────────────────────────────────────────────────────────
    "CSD_CrCl4_2m_tet_spin4": dict(
        metal='Cr', ligand='Cl', n_ligands=4, charge=-2, spin_2s=4,
        geometry='tet', dist_ang=None, metal_row='3d',
        autocas_ne=12, autocas_no=8,
    ),
    "CSD_MnCl4_2m_tet_spin5": dict(
        metal='Mn', ligand='Cl', n_ligands=4, charge=-2, spin_2s=5,
        geometry='tet', dist_ang=None, metal_row='3d',
        autocas_ne=17, autocas_no=10,
    ),
    "CSD_MnBr4_2m_tet_spin5": dict(
        metal='Mn', ligand='Br', n_ligands=4, charge=-2, spin_2s=5,
        geometry='tet', dist_ang=None, metal_row='3d',
        autocas_ne=17, autocas_no=10,
    ),
    "CSD_MnF6_4m_oct_spin5": dict(
        metal='Mn', ligand='F',  n_ligands=6, charge=-4, spin_2s=5,
        geometry='oct', dist_ang=None, metal_row='3d',
        autocas_ne=15, autocas_no=8,
    ),
    "CSD_CrCl4_2m_tet_spin2": dict(
        metal='Cr', ligand='Cl', n_ligands=4, charge=-2, spin_2s=2,
        geometry='tet', dist_ang=None, metal_row='3d',
        autocas_ne=6, autocas_no=7,
    ),
    "CSD_FeCl4_2m_tet_spin0": dict(
        metal='Fe', ligand='Cl', n_ligands=4, charge=-2, spin_2s=0,
        geometry='tet', dist_ang=None, metal_row='3d',
        autocas_ne=4, autocas_no=4,
    ),
    "CSD_NiCl4_2m_sqpl_spin0": dict(
        metal='Ni', ligand='Cl', n_ligands=4, charge=-2, spin_2s=0,
        geometry='sqpl', dist_ang=None, metal_row='3d',
        autocas_ne=6, autocas_no=6,
    ),
    "CSD_NiCl6_4m_oct_spin2": dict(
        metal='Ni', ligand='Cl', n_ligands=6, charge=-4, spin_2s=2,
        geometry='oct', dist_ang=None, metal_row='3d',
        autocas_ne=12, autocas_no=6,
    ),
    "CSD_CoCl6_4m_oct_spin1": dict(
        metal='Co', ligand='Cl', n_ligands=6, charge=-4, spin_2s=1,
        geometry='oct', dist_ang=None, metal_row='3d',
        autocas_ne=3, autocas_no=2,
    ),
    # ── 4d ──────────────────────────────────────────────────────────────
    "Mo_Cl6_chg-3_spin3_oct_d2p299": dict(
        metal='Mo', ligand='Cl', n_ligands=6, charge=-3, spin_2s=3,
        geometry='oct', dist_ang=2.299, metal_row='4d',
        autocas_ne=25, autocas_no=20,
    ),
    "Mo_Cl6_chg-3_spin1_oct_d2p299": dict(
        metal='Mo', ligand='Cl', n_ligands=6, charge=-3, spin_2s=1,
        geometry='oct', dist_ang=2.299, metal_row='4d',
        autocas_ne=7, autocas_no=7,
    ),
    "Rh_Cl6_chg-3_spin0_oct_d2p32": dict(
        metal='Rh', ligand='Cl', n_ligands=6, charge=-3, spin_2s=0,
        geometry='oct', dist_ang=2.320, metal_row='4d',
        autocas_ne=2, autocas_no=4,
    ),
    "Ru_Cl6_chg-3_spin1_oct_d2p232": dict(
        metal='Ru', ligand='Cl', n_ligands=6, charge=-3, spin_2s=1,
        geometry='oct', dist_ang=2.232, metal_row='4d',
        autocas_ne=11, autocas_no=12,
    ),
    "Pd_Cl4_chg-2_spin0_sq_pl_d2p3": dict(
        metal='Pd', ligand='Cl', n_ligands=4, charge=-2, spin_2s=0,
        geometry='sq_pl', dist_ang=2.300, metal_row='4d',
        autocas_ne=22, autocas_no=16,
    ),
    "Rh_Cl6_chg-3_spin2_oct_d2p32": dict(
        metal='Rh', ligand='Cl', n_ligands=6, charge=-3, spin_2s=2,
        geometry='oct', dist_ang=2.320, metal_row='4d',
        autocas_ne=8, autocas_no=10,
    ),
    "Pd_Cl6_chg-2_spin0_oct_d2p3": dict(
        metal='Pd', ligand='Cl', n_ligands=6, charge=-2, spin_2s=0,
        geometry='oct', dist_ang=2.300, metal_row='4d',
        autocas_ne=4, autocas_no=5,
    ),
    # ── Br-ligand medium and low spin systems ─────────────────────────────
    # AutoCAS active spaces are REAL M=250 results from backup benchmark
    # dataset. These systems enable validation across all three spin
    # categories (high/medium/low) with consistent ligand (Br) and
    # 3d metal row.
    "V_Br6_chg-3_spin2_oct_d2p318": dict(
        metal='V',  ligand='Br', n_ligands=6, charge=-3, spin_2s=2,
        geometry='oct', dist_ang=2.318, metal_row='3d',
        # Real AutoCAS M=250 result, parity_ok=True
        autocas_ne=10, autocas_no=6,
    ),
    "Ni_Br6_chg-4_spin2": dict(
        metal='Ni', ligand='Br', n_ligands=6, charge=-4, spin_2s=2,
        geometry='oct', dist_ang=2.53, metal_row='3d',
        # Real AutoCAS M=250 result, parity_ok=True
        autocas_ne=10, autocas_no=5,
    ),
    "Mn_Br4_chg-1_spin0": dict(
        metal='Mn', ligand='Br', n_ligands=6, charge=-1, spin_2s=0,
        geometry='oct', dist_ang=2.63, metal_row='3d',
        # Real AutoCAS M=250 result, parity_ok=True
        autocas_ne=18, autocas_no=13,
    ),
    "V_Br6_chg-2_spin1_oct_d2p318": dict(
        metal='V',  ligand='Br', n_ligands=6, charge=-2, spin_2s=1,
        geometry='oct', dist_ang=2.318, metal_row='3d',
        # Real AutoCAS M=250 result, parity_ok=False (use with caution)
        autocas_ne=4, autocas_no=3,
    ),
}

ALL_SYSTEM_NAMES = list(SYSTEMS.keys())

# ── XYZ reader ────────────────────────────────────────────────────────────

def read_xyz(xyz_path):
    """
    Read an XYZ file and return (atom_list, comment).

    atom_list format: [(symbol, (x, y, z)), ...]  -- coordinates in Angstrom.
    This is the format PySCF mol.atom expects with mol.unit = 'angstrom'.

    The XYZ file format is:
        Line 1: number of atoms (integer)
        Line 2: comment (ignored, but stored)
        Lines 3+: symbol  x  y  z
    """
    with open(xyz_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    n_atoms = int(lines[0])
    comment = lines[1]
    atom_list = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        sym   = parts[0]
        xyz   = (float(parts[1]), float(parts[2]), float(parts[3]))
        atom_list.append((sym, xyz))

    if len(atom_list) != n_atoms:
        raise ValueError(
            f"XYZ file {xyz_path}: header says {n_atoms} atoms "
            f"but found {len(atom_list)}"
        )

    print(f"  [XYZ] Loaded {n_atoms} atoms from {xyz_path}")
    print(f"  [XYZ] Comment: {comment}")
    return atom_list, comment


# ── Geometry builder ──────────────────────────────────────────────────────

def build_mol(name, s, xyz_path=None):
    """
    Build and return a PySCF Mole object for system s.

    If xyz_path is given, geometry is read from that XYZ file
    (e.g. PBE0/def2-TZVP optimized geometry from geom_opt_MnX4.py).
    Otherwise falls back to ideal parametric geometry as before.
    """
    if xyz_path is not None:
        # ── Use DFT-optimized geometry from XYZ file ──────────────────
        atom_list, comment = read_xyz(xyz_path)
        print(f"  [build_mol] Using DFT-optimized geometry: {xyz_path}")
    else:
        # ── Use ideal parametric geometry (original behaviour) ─────────
        dist = s['dist_ang'] or DIST_TABLE.get(
            (s['metal'], s['ligand'], s['geometry']), 2.30)
        builder = GEOMETRY_BUILDERS[s['geometry']]
        lig_coords = builder(dist)[:s['n_ligands']]
        atom_list = [(s['metal'], (0., 0., 0.))]
        for xyz in lig_coords:
            atom_list.append((s['ligand'], xyz))
        print(f"  [build_mol] Using ideal parametric geometry "
              f"(M-L = {dist:.4f} Ang)")

    mol = gto.Mole()
    mol.atom     = atom_list
    mol.charge   = s['charge']
    mol.spin     = s['spin_2s']   # PySCF spin = 2S = n_alpha - n_beta
    mol.basis    = 'def2-svp'
    mol.unit     = 'angstrom'
    mol.symmetry = False          # avoid symmetry complications

    # ECP for 4d/5d metals — critical: without this PySCF runs all-electron
    _4D = {'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd'}
    _5D = {'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg'}
    if s['metal_row'] in ('4d','5d') or s.get('metal') in (_4D | _5D):
        mol.ecp = {s['metal']: 'def2-svp'}   # correct — metal only

    mol.verbose = 4
    mol.build()
    return mol


# ── HF reference ─────────────────────────────────────────────────────────

def run_hf(mol, s):
    """Run UHF with robust convergence fallbacks. Returns mf object."""
    mf = scf.UHF(mol)
    mf.init_guess = "atom"
    mf.max_cycle  = 500
    mf.conv_tol   = 1e-10
    mf.kernel()

    if not mf.converged:
        print("  HF not converged, trying level_shift=0.3 + minao guess...")
        mf2 = scf.UHF(mol)
        mf2.max_cycle    = 500
        mf2.conv_tol     = 1e-10
        mf2.level_shift  = 0.3
        mf2.init_guess   = "minao"
        mf2.kernel()
        if mf2.converged:
            return mf2
        mf = mf2

    if not mf.converged:
        print("  HF not converged, trying level_shift=0.6...")
        mf3 = scf.UHF(mol)
        mf3.max_cycle    = 500
        mf3.conv_tol     = 1e-10
        mf3.level_shift  = 0.6
        mf3.init_guess   = "minao"
        mf3.diis_start_cycle = 10
        mf3.kernel(mf.make_rdm1())
        if mf3.converged:
            return mf3
        mf = mf3

    if not mf.converged:
        print("  HF not converged, trying Newton solver...")
        mf4 = mf.newton()
        mf4.max_cycle = 100
        mf4.kernel(mf.mo_coeff)
        if mf4.converged:
            return mf4
        mf = mf4

    if not mf.converged:
        print("  WARNING: HF did not converge after all attempts.")
        print("  Proceeding with best available orbitals.")
    return mf


# ── Window selection ──────────────────────────────────────────────────────

def select_window(mf, s, window_size):
    """
    Return (window_indices, n_elec_in_window).

    Strategy: frontier-centred window of `window_size` MOs around the
    HOMO/LUMO gap, guaranteed to include all singly-occupied orbitals.
    Uses alpha occupation from UHF to locate SOCCs.
    """
    n_mo    = mf.mo_occ[0].shape[0]   # alpha MOs
    n_occ_a = int(mf.mo_occ[0].sum())
    n_occ_b = int(mf.mo_occ[1].sum())
    n_socc  = n_occ_a - n_occ_b       # number of singly-occupied

    # Half-half split around HOMO of alpha
    half = window_size // 2
    start = max(0,      n_occ_a - half)
    end   = min(n_mo,   start   + window_size)
    # Adjust if we hit the end boundary
    if end - start < window_size:
        start = max(0, end - window_size)

    window = list(range(start, end))

    # Guarantee all SOCCs are inside the window
    # SOCCs in UHF are the n_socc highest occupied alpha orbitals
    socc_indices = list(range(n_occ_b, n_occ_a))
    for idx in socc_indices:
        if idx not in window:
            # Extend window to include this SOCC
            window = list(range(min(window[0], idx),
                                max(window[-1]+1, idx+1)))

    # Electron count in window (using average of alpha+beta occupations)
    occ_avg = (mf.mo_occ[0] + mf.mo_occ[1]) / 2.0
    n_elec_in_window = int(round(2 * occ_avg[window].sum()))

    return window, n_elec_in_window


# ── DMRG step ─────────────────────────────────────────────────────────────

def run_dmrg(mol, mf, window, n_elec_in_window, M=350, n_sweeps=50,
             scratch_dir=None, name='system'):
    """
    Run low-bond-dimension DMRG on the orbital window.
    Returns (mc_dmrg, e_dmrg).
    """
    n_act = len(window)

    # Build MO coefficients with window orbitals as active
    mo_coeff = mf.mo_coeff  # UHF returns (alpha, beta); use alpha for CASSCF
    # For UHF -> CASSCF we need a single set of MOs.
    # Natural orbitals from UHF density are the standard approach.
    dm_a, dm_b = mf.make_rdm1()
    dm_total = dm_a + dm_b
    # Diagonalise density in MO basis
    n_a, v_a = np.linalg.eigh(dm_a)   # alpha natural occupations
    # Use alpha MO coefficients as the reference orbital set
    mo_ref = mf.mo_coeff[0]  # alpha coefficients

    # Reorder window orbitals to be contiguous as active space
    n_mo  = mo_ref.shape[1]
    n_occ_a = int(mf.mo_occ[0].sum())

    # Build CASSCF object at window size
    mc = mcscf.CASSCF(mf.to_rhf(), n_act, n_elec_in_window)

    # Tell PySCF to sort the active space from window
    # mo_coeff ordering: keep window orbitals as active block
    # Build a reordered MO set: [core | active_window | virtual]
    all_idx   = list(range(n_mo))
    core_idx  = [i for i in all_idx if i not in window and
                 mf.mo_occ[0][i] + mf.mo_occ[1][i] > 1.5]
    virt_idx  = [i for i in all_idx if i not in window and
                 mf.mo_occ[0][i] + mf.mo_occ[1][i] < 0.5]
    other_idx = [i for i in all_idx if i not in window and
                 i not in core_idx and i not in virt_idx]
    # Compose ordered index: core | window | (other singly-occ) | virtual
    ordered = core_idx + window + other_idx + virt_idx
    mo_ordered = mo_ref[:, ordered]

    # Set up DMRG solver via block2
    if scratch_dir is None:
        _base = os.environ.get('PYSCF_TMPDIR',
                os.environ.get('SCRATCH',
                '/scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch'))
        scratch_dir = os.path.join(_base, f'qicas_{name}')
    os.makedirs(scratch_dir, exist_ok=True)

    mc.fcisolver = dmrgci.DMRGCI(mol, maxM=M, tol=1e-8)
    mc.fcisolver.scratchDirectory  = scratch_dir
    mc.fcisolver.runtimeDir        = scratch_dir
    mc.fcisolver.maxIter           = n_sweeps
    mc.fcisolver.block_extra_keyword = ['num_thrds 64']  # adjust for cluster

    # Explicitly set twodot_to_onedot to avoid block2 AssertionError:
    # block2 requires: len(schedule) > twodot_to_onedot
    # PySCF generates schedule with ceil(n_sweeps * 0.75) entries approximately,
    # so set tto = max(2, n_sweeps // 2) which is always safely below n_sweeps.
    # At n_sweeps=20: tto=10  (was incorrectly 22 before this fix)
    # At n_sweeps=50: tto=25
    mc.fcisolver.twodot_to_onedot = max(2, n_sweeps // 2)

    # Run DMRG-CASCI (no orbital optimisation here — that is QICAS's job)
    mc.max_cycle_macro = 1   # single pass: CASCI not CASSCF
    e_dmrg = mc.kernel(mo_ordered)[0]

    return mc, e_dmrg, mo_ordered


# ── Single-orbital entropies from 1-RDM and 2-RDM ────────────────────────

def _orbital_entropy_from_lam(n_i, G_ii):
    """S(rho_i) from spin-averaged occupation n_i and double-occ G_ii."""
    lam = np.array([1.0 - 2*n_i + G_ii, n_i - G_ii, n_i - G_ii, G_ii])
    lam = np.clip(lam, 1e-14, 1.0)
    lam /= lam.sum()
    return -float(np.sum(lam * np.log(lam)))


def get_rdms(mc):
    """
    Get spin-summed 1-RDM (gamma) and 2-RDM (Gamma) from the DMRG object.
    Corresponds to "Calculate 1- and 2-RDM gamma and Gamma" in Figure 2.
    """
    try:
        dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
        print("  [RDM] obtained via make_rdm12()")
        return dm1, dm2
    except Exception as e:
        print(f"  [RDM] make_rdm12 failed ({e}), using make_rdm1 + MF approx")
        dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        n = mc.ncas
        dm2 = np.zeros((n, n, n, n))
        for i in range(n):
            dm2[i, i, i, i] = (dm1[i, i] / 2.0) ** 2 * 4.0
        return dm1, dm2


def entropies_from_rdms(gamma, Gamma, n):
    """Single-orbital entropies from spin-summed 1-RDM and 2-RDM."""
    ent = np.zeros(n)
    for i in range(n):
        ent[i] = _orbital_entropy_from_lam(gamma[i,i]/2.0, Gamma[i,i,i,i]/4.0)
    return ent


def entropy_plateau_cas_size(entropies, spin_2s, entropy_floor=0.05, min_active=2):
    """
    Determine D_CAS from entropy plateau (Stein-Reiher/AutoCAS method, ref 21
    in the QICAS paper). Finds the largest ABSOLUTE entropy gap in the sorted
    profile among significant orbitals (entropy > entropy_floor).

    More robust than ratio-jump: small tail values (0.01/0.005 = ratio 2)
    do not create large absolute gaps and do not pollute the cutoff.
    """
    sig_idx   = np.where(entropies > entropy_floor)[0]
    n_sig     = len(sig_idx)
    if n_sig <= max(spin_2s, min_active):
        return max(spin_2s, min_active)
    sig_sorted = np.sort(entropies[sig_idx])[::-1]
    gaps       = sig_sorted[:-1] - sig_sorted[1:]
    d_cas      = int(np.argmax(gaps)) + 1
    return int(np.clip(max(d_cas, spin_2s, min_active), 2, n_sig))


def qicas_orbital_rotation(gamma, Gamma, n_win, d_cas, max_iter=300, tol=1e-7):
    """
    QICAS core: minimize F_QI = sum_{i in N} S(rho_i) over unitary rotations.
    Faithful to "Minimize the entropy of non-active orbitals" in Figure 2.

    U = expm(X), X skew-symmetric.
    N = the (n_win - d_cas) orbitals with lowest initial entropy (fixed set).
    Gradient computed numerically via L-BFGS-B.

    Only the Gamma diagonal Gamma[i,i,i,i] for i in N is computed per step,
    making each evaluation O(d_non_active * n_win^4).
    """
    from scipy.linalg import expm as _expm
    from scipy.optimize import minimize as _minimize
    n = n_win

    ent_0      = entropies_from_rdms(gamma, Gamma, n)
    sorted_idx = np.argsort(ent_0)[::-1]
    nonact_idx = sorted_idx[d_cas:].tolist()   # fixed throughout
    fqi_0      = float(ent_0[nonact_idx].sum())
    print(f"  [QICAS rotation] D_CAS={d_cas}, |N|={len(nonact_idx)}, "
          f"F_QI(i)={fqi_0:.6f}")

    def _fqi(x_flat):
        X = x_flat.reshape(n, n); X = (X - X.T) / 2
        U = _expm(X)
        g = U @ gamma @ U.T
        val = 0.0
        for i in nonact_idx:
            u_i  = U[:, i]
            G_ii = np.einsum('p,q,r,s,pqrs->', u_i, u_i, u_i, u_i, Gamma) / 4.0
            val += _orbital_entropy_from_lam(g[i, i] / 2.0, G_ii)
        return val

    res = _minimize(_fqi, np.zeros(n * n), method='L-BFGS-B',
                    options={"maxiter": max_iter, "ftol": 1e-5, "gtol": 1e-6, "maxls": 50,
                             'maxfun': max_iter * 20})

    X_opt = res.x.reshape(n, n); X_opt = (X_opt - X_opt.T) / 2
    U_opt = _expm(X_opt)
    g_opt = U_opt @ gamma @ U_opt.T
    ent_opt = np.zeros(n)
    for i in range(n):
        u_i     = U_opt[:, i]
        G_ii    = np.einsum('p,q,r,s,pqrs->', u_i, u_i, u_i, u_i, Gamma) / 4.0
        ent_opt[i] = _orbital_entropy_from_lam(g_opt[i, i] / 2.0, G_ii)
    fqi_f = float(ent_opt[nonact_idx].sum())
    print(f"  [QICAS rotation] F_QI(f)={fqi_f:.6f}, "
          f"reduction={fqi_0-fqi_f:.6f}, ok={res.success}, nit={res.nit}")
    return U_opt, ent_opt, g_opt, fqi_0, fqi_f


def determine_active_space_from_qicas(ent_qicas, gamma_qicas, spin_2s, d_cas,
                                       window, mf):
    """
    Classify orbitals in the QICAS-rotated basis B*.

    The paper (p. 11024): "a nonactive orbital in B* is classified as closed
    (virtual) if its occupancy is larger (smaller) than 1."

    D_CAS was fixed BEFORE the rotation and does not change. After rotation:
    - Take the d_cas orbitals with highest entropy as active
    - Count active electrons from HF occupations of the selected window orbitals
      (NOT from DMRG NOONs, which are correlated within the window and give
       inflated values for near-doubly-occupied orbitals)
    - Adjust for parity with spin_2s
    """
    n_win      = len(ent_qicas)

    # Active = top d_cas orbitals by entropy in the rotated basis
    sorted_idx = np.argsort(ent_qicas)[::-1]
    active_rel = sorted(sorted_idx[:d_cas].tolist())

    # Active electron count from HF occupations (correct reference)
    # window[r] gives the absolute MO index in the full molecular orbital set
    abs_active = [window[r] for r in active_rel]
    occ_a = mf.mo_occ[0]
    occ_b = mf.mo_occ[1]
    n_active_e = int(round(sum(occ_a[i] + occ_b[i] for i in abs_active)))

    # Parity: (n_active_e - spin_2s) must be even
    if (n_active_e - spin_2s) % 2 != 0:
        n_active_e += 1
    if (n_active_e - spin_2s) % 2 != 0:
        n_active_e -= 2

    return active_rel, d_cas, n_active_e


def build_casci_mo(mo_ordered_alpha, window, active_rel, n_mo, mf):
    """
    Reorder MOs so that active orbitals form a contiguous active block:
        [core | active | virtual]

    Returns (mo_reordered, n_core) where n_core is the number of
    doubly-occupied core orbitals.
    """
    n_win = len(window)
    abs_active = [window[r] for r in active_rel]
    abs_nonact_win = [window[r] for r in range(n_win) if r not in active_rel]

    # All orbital indices
    all_idx = list(range(n_mo))

    # Classify non-window orbitals
    occ_avg = (mf.mo_occ[0] + mf.mo_occ[1]) / 2.0
    core_nonwin = [i for i in all_idx
                   if i not in window and occ_avg[i] > 0.9]
    virt_nonwin = [i for i in all_idx
                   if i not in window and occ_avg[i] < 0.1]
    other_nonwin = [i for i in all_idx
                    if i not in window and i not in core_nonwin
                    and i not in virt_nonwin]

    # Classify non-active window orbitals as core or virtual
    core_win = [i for i in abs_nonact_win if occ_avg[i] > 0.9]
    virt_win = [i for i in abs_nonact_win if occ_avg[i] < 0.1]
    other_win = [i for i in abs_nonact_win
                 if i not in core_win and i not in virt_win]

    core_all = sorted(core_nonwin + core_win)
    virt_all = sorted(virt_nonwin + virt_win)
    other_all = sorted(other_nonwin + other_win)

    ordered = core_all + abs_active + other_all + virt_all
    mo_reordered = mo_ordered_alpha[:, ordered]
    n_core = len(core_all)

    # Verify electron count sanity
    n_elec_core = int(round(2 * sum(occ_avg[i] for i in core_all)))
    assert n_elec_core % 2 == 0, (
        f"Core electron count {n_elec_core} is odd — "
        f"SOCC may have been left outside the active space.")

    return mo_reordered, n_core


# ── CASCI validation (Goal 1) ─────────────────────────────────────────────

def run_casci(mol, mf, mo_hf, mo_qicas, n_active_e, n_active, label):
    """
    Run CASCI with both HF-ordered and QICAS-rotated MOs at the same
    CAS size.  Returns (e_hf, e_qicas, delta_mha).
    """
    def _casci_energy(mo_ref):
        # Enforce minimum orbitals: n_active >= ceil(n_alpha) to avoid PySCF assertion
        import math
        n_alpha = (n_active_e + 1) // 2 if isinstance(n_active_e, int) else (n_active_e[0] + n_active_e[1])
        n_alpha = n_active_e[0] if isinstance(n_active_e, tuple) else (n_active_e + 1) // 2
        n_active_safe = max(n_active, n_alpha)
        mc = mcscf.CASCI(mf.to_rhf(), n_active_safe, n_active_e)
        mc.verbose = 3
        e = mc.kernel(mo_ref)[0]
        return float(e)

    e_hf    = _casci_energy(mo_hf)
    e_qicas = _casci_energy(mo_qicas)
    delta   = (e_qicas - e_hf) * 1000.0   # Hartree -> mHa, negative = better

    return e_hf, e_qicas, delta


# ── CASSCF from two starting points (Goal 2) ─────────────────────────────

def run_casscf_comparison(mol, mf, mo_hf, mo_qicas,
                           n_active_e, n_active, max_macro=200, name="system"):
    """
    Run CASSCF to convergence from both HF and QICAS starting orbitals.
    Returns dict with energies, iteration counts, and wall times.
    """
    results = {}

    for label, mo_start in [('from_hf', mo_hf), ('from_qicas', mo_qicas)]:
        t0 = time.time()
        mc = mcscf.CASSCF(mf.to_rhf(), n_active, n_active_e)
        mc.fcisolver = dmrgci.DMRGCI(mol, maxM=350, tol=1e-8)
        mc.fcisolver.scratchDirectory = os.path.join(os.environ.get("PYSCF_TMPDIR","/scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch"), f"casscf_{label}_{name}")
        mc.fcisolver.runtimeDir = mc.fcisolver.scratchDirectory
        mc.fcisolver.maxIter = 30
        mc.fcisolver.block_extra_keyword = ["num_thrds 64"]
        mc.fcisolver.twodot_to_onedot = 15
        os.makedirs(mc.fcisolver.scratchDirectory, exist_ok=True)
        mc.max_cycle_macro = max_macro
        mc.conv_tol        = 1e-9
        mc.conv_tol_grad   = 3e-4   # DMRG noise requires relaxed gradient tolerance
        mc.verbose         = 4

        # Count macro iterations by hooking into the callback
        macro_count = [0]
        orig_callback = mc.callback

        def count_callback(envs):
            macro_count[0] += 1
            if orig_callback:
                orig_callback(envs)
        mc.callback = count_callback

        e, _, _, _, _ = mc.kernel(mo_start)
        t1 = time.time()

        results[label] = {
            'e_casscf':    float(e),
            'converged':   bool(mc.converged),
            'n_macro_iter': macro_count[0],
            't_s':         t1 - t0,
        }

    # Orbital subspace overlap: QICAS active space vs CASSCF-from-HF active space
    # (measures whether QICAS found the same orbital subspace)
    try:
        mo_cas_hf  = results['from_hf']['mc_mo'] if 'mc_mo' in results['from_hf'] else None
        results['overlap_note'] = "Run qicas_overlap.py post-hoc for orbital SVD"
    except Exception:
        pass

    return results


# ── Orbital subspace overlap ──────────────────────────────────────────────

def orbital_overlap(mo_a, mo_b, mol):
    """
    Compute singular values of the overlap matrix between two active
    subspaces (columns of mo_a and mo_b are the active MOs).
    Mean singular value close to 1 means the subspaces are the same.
    """
    S = mol.intor('int1e_ovlp')
    M = mo_a.T @ S @ mo_b
    svd_vals = np.linalg.svd(M, compute_uv=False)
    return {
        'singular_values': svd_vals.tolist(),
        'min_sv':          float(svd_vals.min()),
        'mean_sv':         float(svd_vals.mean()),
    }


# ── Main per-system runner ────────────────────────────────────────────────

def run_one_system(name, s, M_dmrg=350, n_sweeps=50, out_dir='results',
                   xyz_path=None, save_mo=False):
    """
    Full QICAS + CASCI + CASSCF pipeline for one system.
    Returns a result dict and writes it to JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'qicas_casscf_{name}.json')

    result = {
        'name':       name,
        'metal':      s['metal'],
        'ligand':     s['ligand'],
        'charge':     s['charge'],
        'spin_2s':    s['spin_2s'],
        'metal_row':  s['metal_row'],
        'M_dmrg':     M_dmrg,
        'geometry_source': xyz_path if xyz_path else 'ideal_parametric',
        # autoCAS reference stored for post-hoc comparison — NOT used by QICAS
        'autocas_reference': {
            'cas_ne': s['autocas_ne'],
            'cas_no': s['autocas_no'],
            'note':   'autoCAS M=150 result, stored for comparison only'
        },
        'status': 'RUNNING',
    }

    t_total_start = time.time()

    try:
        # ── Step 1: Build molecule ──────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  metal={s['metal']} ligand={s['ligand']} "
              f"charge={s['charge']} 2S={s['spin_2s']} row={s['metal_row']}")
        print(f"{'='*60}")

        mol = build_mol(name, s, xyz_path=xyz_path)
        result['n_electrons'] = int(mol.nelectron)
        result['n_basis']     = mol.nao_nr()

        # ── Step 2: HF reference ────────────────────────────────────────
        print("\n[Step 2] Running UHF...")
        t0 = time.time()
        mf = run_hf(mol, s)
        result['e_hf']    = float(mf.e_tot)
        result['t_hf_s']  = time.time() - t0
        print(f"  E(HF) = {mf.e_tot:.10f} Ha  ({result['t_hf_s']:.1f} s)")

        # ── Step 3: Window selection ────────────────────────────────────
        window_size = 24 if s['n_ligands'] == 6 else 20
        if s['spin_2s'] >= 5:
            window_size = max(window_size, 26)

        window, n_elec_window = select_window(mf, s, window_size)
        result['window'] = {
            'size':          len(window),
            'abs_indices':   window,
            'n_elec':        n_elec_window,
        }
        print(f"\n[Step 3] Window: {len(window)} orbitals, "
              f"{n_elec_window} electrons")

        # ── Step 4: DMRG on window ──────────────────────────────────────
        print(f"\n[Step 4] DMRG (M={M_dmrg}, {n_sweeps} sweeps)...")
        t0 = time.time()
        # Use Noctua2 scratch filesystem — /tmp is too small for M>=250
        # PYSCF_TMPDIR is set by block2_env.sh to /scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch
        _base = os.environ.get('PYSCF_TMPDIR',
                os.environ.get('SCRATCH',
                '/scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch'))
        scratch = os.path.join(_base, f'qicas_{name}_M{M_dmrg}')
        os.makedirs(scratch, exist_ok=True)
        print(f"  DMRG scratch: {scratch}")
        mc_dmrg, e_dmrg, mo_ordered = run_dmrg(
            mol, mf, window, n_elec_window,
            M=M_dmrg, n_sweeps=n_sweeps,
            scratch_dir=scratch, name=name)
        result['e_dmrg']   = float(e_dmrg)
        result['t_dmrg_s'] = time.time() - t0
        print(f"  E(DMRG) = {e_dmrg:.10f} Ha  ({result['t_dmrg_s']:.1f} s)")

        # ── Step 5: Get 1-RDM and 2-RDM (γ and Γ) ──────────────────────
        # Paper Figure 2: "Calculate 1- and 2-RDM γ and Γ"
        print("\n[Step 5] Getting 1-RDM and 2-RDM from DMRG...")
        t0 = time.time()
        gamma, Gamma = get_rdms(mc_dmrg)
        entropies_init = entropies_from_rdms(gamma, Gamma, len(window))
        d_cas_init = entropy_plateau_cas_size(entropies_init, s['spin_2s'])
        result['entropies_initial'] = {
            'values':             entropies_init.tolist(),
            'window_abs_indices': window,
            'd_cas_from_plateau': d_cas_init,
        }
        print(f"  Entropy range (initial): [{entropies_init.min():.4f}, "
              f"{entropies_init.max():.4f}]")
        print(f"  D_CAS from initial plateau: {d_cas_init}")

        # ── Step 6: QICAS orbital rotation ─────────────────────────────
        # Paper Figure 2: "Minimize the entropy of non-active orbitals"
        print("\n[Step 6] QICAS orbital rotation (F_QI minimization)...")
        t0 = time.time()
        U_qicas, ent_qicas, gamma_qicas, fqi_initial, fqi_final = \
            qicas_orbital_rotation(gamma, Gamma, len(window), d_cas_init)
        t_rot = time.time() - t0

        active_rel, n_active, n_active_e = determine_active_space_from_qicas(
            ent_qicas, gamma_qicas, s['spin_2s'], d_cas_init, window, mf)

        result['qicas'] = {
            'n_active':        n_active,
            'n_active_e':      n_active_e,
            'active_rel':      active_rel,
            'active_abs':      [window[r] for r in active_rel],
            'fqi_initial':     fqi_initial,
            'fqi_final':       fqi_final,
            'fqi_reduction':   fqi_initial - fqi_final,
            't_rotation_s':    t_rot,
            'entropies_qicas': ent_qicas.tolist(),
        }
        print(f"  QICAS active space: CAS({n_active_e},{n_active})")
        print(f"  F_QI: {fqi_initial:.4f} → {fqi_final:.4f} "
              f"(reduction: {fqi_initial-fqi_final:.4f})")
        print(f"  Rotation wall time: {t_rot:.1f} s")

        # ── Save QICAS-rotated MO matrix (for spin-gap benchmark) ───────
        if save_mo:
            mo_save_path = os.path.join(out_dir, f'qicas_mo_{name}.npy')
            n_core_nw = sum(1 for i in range(mol.nao_nr())
                            if i not in window and
                            (mf.mo_occ[0][i] + mf.mo_occ[1][i]) > 1.5)
            n_win_save = len(window)
            mo_to_save = mo_ordered.copy()
            mo_to_save[:, n_core_nw:n_core_nw + n_win_save] = (
                mo_ordered[:, n_core_nw:n_core_nw + n_win_save] @ U_qicas.T
            )
            np.save(mo_save_path, mo_to_save)
            result['qicas']['mo_saved_path'] = mo_save_path
            print(f"  QICAS MO matrix saved to: {mo_save_path}")

        # ── Step 7: Build MO arrays (apply U_qicas to window block) ────
        # Paper: "QICAS optimized orbitals" = orbitals after rotation
        print("\n[Step 7] Building MO arrays...")
        mo_alpha = mf.mo_coeff[0]   # alpha MOs (UHF reference)

        # HF-ordered MOs at QICAS-determined CAS size (Goal 2 baseline)
        mo_hf_cas, n_core_hf = build_casci_mo(
            mo_alpha, window, active_rel, mol.nao_nr(), mf)

        # QICAS-rotated MOs: apply U_qicas within the window block
        # mo_ordered has window orbitals contiguous after core block
        n_core_nonwin = sum(1 for i in range(mol.nao_nr())
                            if i not in window and
                            (mf.mo_occ[0][i] + mf.mo_occ[1][i]) > 1.5)
        n_win = len(window)
        mo_qicas_full = mo_ordered.copy()
        # Columns [n_core_nonwin : n_core_nonwin+n_win] are the window orbitals
        mo_qicas_full[:, n_core_nonwin:n_core_nonwin + n_win] = (
            mo_ordered[:, n_core_nonwin:n_core_nonwin + n_win] @ U_qicas.T
        )
        mo_qicas_cas, n_core_qi = build_casci_mo(
            mo_qicas_full, window, active_rel, mol.nao_nr(), mf)
        # ── Step 8 (Goal 1): CASCI validation ──────────────────────────
        print(f"\n[Step 8 / Goal 1] CASCI at CAS({n_active_e},{n_active})...")
        t0 = time.time()
        e_casci_hf, e_casci_qi, delta_casci = run_casci(
            mol, mf, mo_hf_cas, mo_qicas_cas, n_active_e, n_active,
            label=name)
        t_casci = time.time() - t0

        result['goal1_casci'] = {
            'cas_ne':          n_active_e,
            'cas_no':          n_active,
            'e_casci_hf':      e_casci_hf,
            'e_casci_qicas':   e_casci_qi,
            'delta_mha':       delta_casci,
            'qicas_better':    delta_casci < 0,
            't_s':             t_casci,
        }
        print(f"  E(CASCI|HF)    = {e_casci_hf:.10f} Ha")
        print(f"  E(CASCI|QICAS) = {e_casci_qi:.10f} Ha")
        print(f"  Δ = {delta_casci:+.3f} mHa "
              f"({'QICAS better' if delta_casci < 0 else 'HF better'})")

        # ── Step 9 (Goal 2): CASSCF from both starting points ──────────
        print(f"\n[Step 9 / Goal 2] CASSCF from HF and QICAS orbitals "
              f"at CAS({n_active_e},{n_active})...")
        t0 = time.time()
        casscf_results = run_casscf_comparison(
            mol, mf, mo_hf_cas, mo_qicas_cas, n_active_e, n_active,
            max_macro=400, name=name)
        t_casscf = time.time() - t0

        e_hf_casscf = casscf_results['from_hf']['e_casscf']
        e_qi_casscf = casscf_results['from_qicas']['e_casscf']
        n_iter_hf   = casscf_results['from_hf']['n_macro_iter']
        n_iter_qi   = casscf_results['from_qicas']['n_macro_iter']
        iter_speedup = n_iter_hf - n_iter_qi   # positive = QICAS faster

        result['goal2_casscf'] = {
            'cas_ne':          n_active_e,
            'cas_no':          n_active,
            'from_hf':         casscf_results['from_hf'],
            'from_qicas':      casscf_results['from_qicas'],
            'iter_speedup':    iter_speedup,
            'qicas_faster':    iter_speedup > 0,
            'energy_diff_mha': (e_qi_casscf - e_hf_casscf) * 1000.0,
            't_total_s':       t_casscf,
        }

        # CASCI(QICAS orbs) → CASSCF gap: key metric for Goal 2
        gap_casci_to_casscf = (e_casci_qi - e_hf_casscf) * 1000.0
        result['goal2_casscf']['gap_casci_qicas_to_casscf_mha'] = gap_casci_to_casscf
        result['goal2_casscf']['within_chemical_accuracy'] = abs(gap_casci_to_casscf) < 1.6

        print(f"  E(CASSCF|HF)    = {e_hf_casscf:.10f} Ha  "
              f"({n_iter_hf} macro iters)")
        print(f"  E(CASSCF|QICAS) = {e_qi_casscf:.10f} Ha  "
              f"({n_iter_qi} macro iters)")
        print(f"  Iter speedup: {iter_speedup:+d} "
              f"({'QICAS faster' if iter_speedup > 0 else 'HF faster'})")
        print(f"  CASCI→CASSCF gap: {gap_casci_to_casscf:+.3f} mHa "
              f"({'✓ chem. acc.' if abs(gap_casci_to_casscf) < 1.6 else '✗'})")

        # ── Comparison with autoCAS (post-hoc only) ─────────────────────
        autocas_no = s['autocas_no']
        autocas_ne = s['autocas_ne']
        result['comparison_with_autocas'] = {
            'qicas_no':    n_active,
            'qicas_ne':    n_active_e,
            'autocas_no':  autocas_no,
            'autocas_ne':  autocas_ne,
            'delta_no':    n_active - autocas_no,   # positive = QICAS larger
            'delta_ne':    n_active_e - autocas_ne,
            'note': 'autoCAS used as post-hoc reference only — not fed to QICAS',
        }

        result['status']     = 'OK'
        result['wall_time_s'] = time.time() - t_total_start

        print(f"\n  ── DONE in {result['wall_time_s']:.0f} s ──")
        print(f"  autoCAS reference (post-hoc): CAS({autocas_ne},{autocas_no})")
        print(f"  QICAS result:                 CAS({n_active_e},{n_active})")

    except Exception as exc:
        result['status']    = 'ERROR'
        result['error']     = str(exc)
        result['traceback'] = traceback.format_exc()
        result['wall_time_s'] = time.time() - t_total_start
        print(f"\n  ERROR: {exc}")
        print(traceback.format_exc())

    # Write JSON
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result written to: {out_path}")

    return result


# ── SLURM-array entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='QICAS + CASCI + CASSCF for 16 TM benchmark systems')
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--system',       type=str,
                     help='System name (exact key in SYSTEMS dict)')
    grp.add_argument('--system_index', type=int,
                     help='0-based index into ALL_SYSTEM_NAMES (for SLURM array)')
    parser.add_argument('--M',         type=int, default=100,
                        help='DMRG bond dimension (default: 100, paper uses 70-100)')
    parser.add_argument('--sweeps',    type=int, default=50,
                        help='Number of DMRG sweeps (default: 50)')
    parser.add_argument('--out_dir',   type=str, default='results',
                        help='Output directory for JSON results')
    parser.add_argument('--xyz',       type=str, default=None,
                        help='Path to XYZ file with optimized geometry '
                             '(e.g. geom_results/geom_MnCl4_2m_opt.xyz). '
                             'If not given, uses ideal parametric geometry.')
    parser.add_argument('--save_mo',   action='store_true', default=False,
                        help='Save QICAS-rotated MO matrix as .npy file '
                             'in out_dir (needed for spin-gap benchmark).')
    args = parser.parse_args()

    if args.system_index is not None:
        if args.system_index < 0 or args.system_index >= len(ALL_SYSTEM_NAMES):
            print(f"ERROR: system_index must be 0-{len(ALL_SYSTEM_NAMES)-1}")
            sys.exit(1)
        name = ALL_SYSTEM_NAMES[args.system_index]
    else:
        name = args.system
        if name not in SYSTEMS:
            print(f"ERROR: unknown system '{name}'")
            print(f"Available: {ALL_SYSTEM_NAMES}")
            sys.exit(1)

    s = SYSTEMS[name]

    # Validate XYZ path if provided
    xyz_path = None
    if args.xyz:
        if not os.path.exists(args.xyz):
            print(f"ERROR: XYZ file not found: {args.xyz}")
            sys.exit(1)
        xyz_path = args.xyz
        print(f"Using DFT-optimized geometry from: {xyz_path}")
    else:
        print("Using ideal parametric geometry (no --xyz supplied)")

    run_one_system(name, s, M_dmrg=args.M, n_sweeps=args.sweeps,
                   out_dir=args.out_dir, xyz_path=xyz_path,
                   save_mo=args.save_mo)


if __name__ == '__main__':
    main()
