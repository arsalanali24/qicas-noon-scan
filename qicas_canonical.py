"""
qicas_40systems.py
==================
Full canonical QICAS pipeline for 40 transition-metal complexes:
  - 20 high-spin systems (2S >= 3)
  - 20 low-spin  systems (2S <= 1)

Pipeline (faithful to Figure 2 of Ding, Knecht, Schilling JPCL 2023):
  UHF -> frontier window -> Low-m DMRG -> gamma/Gamma -> entropy plateau (D_CAS)
  -> F_QI orbital rotation -> CASCI + CASSCF comparison

QICAS runs COMPLETELY BLIND. Dataset labels (n_active, n_active_e) are
stored for post-hoc comparison ONLY -- never used during calculation.

Key outputs per system:
  - Full per-orbital entropy profile (for plateau visualisation HS vs LS)
  - D_CAS from entropy plateau
  - F_QI reduction from orbital rotation
  - CASCI: QICAS vs HF orbitals
  - CASSCF: convergence from HF vs QICAS starting orbitals

Usage:
    python qicas_40systems.py --system_index $SLURM_ARRAY_TASK_ID
    python qicas_40systems.py --system V_H4_chg-2_spin3_tet_d1p73
    python qicas_40systems.py --list_systems
"""

import os, sys, json, time, math, argparse, traceback
import numpy as np
from scipy.linalg import expm as scipy_expm
from scipy.optimize import minimize as scipy_minimize
from pyscf import gto, scf, mcscf
from pyscf import dmrgscf
from pyscf.dmrgscf import dmrgci

# ── Constants ─────────────────────────────────────────────────────────────

METALS_ECP = {'Mo','Ru','Rh','Pd','Ir','Pt','Os','Re','W','Au'}

def _oct(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0),(0,0,d),(0,0,-d)]
def _tet(d):
    c = d/math.sqrt(3)
    return [(c,c,c),(-c,-c,c),(-c,c,-c),(c,-c,-c)]
def _sqpl(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0)]
GEOM_BUILDERS = {'oct':_oct,'tet':_tet,'sq_pl':_sqpl,'sqpl':_sqpl,'sq':_sqpl}

LIGAND_ATOMS = {
    'Cl':[('Cl',1)],'Br':[('Br',1)],'F':[('F',1)],'I':[('I',1)],
    'H':[('H',1)],'O':[('O',1)],'N':[('N',1)],'S':[('S',1)],'C':[('C',1)],
    'CN':[('C',1),('N',1)],
    'NH3':[('N',1),('H',3)],
    'H2O':[('O',1),('H',2)],
    'PH3':[('P',1),('H',3)],
    'Cl2F4':[('Cl',1),('F',1)],
    'Cl3N1':[('Cl',1),('N',1)],
    'Cl4O2':[('Cl',1),('O',1)],
}

# ── System definitions ────────────────────────────────────────────────────

SYSTEMS = {
    # HIGH-SPIN
    "Pd_Br6_chg-2_spin4_oct_d2p46":   dict(spin_2s=4,metal='Pd',ligand_raw='Br',n_ligands=6,geometry='oct',dist_ang=2.46, charge=-2,metal_row='4d',needs_ecp=True, ref_no=4, ref_ne=10),
    "Ti_Cl6_chg0_spin4_oct_d2p115":   dict(spin_2s=4,metal='Ti',ligand_raw='Cl',n_ligands=6,geometry='oct',dist_ang=2.115,charge=0, metal_row='3d',needs_ecp=False,ref_no=4, ref_ne=10),
    "Ni_Cl2F4_chg-2_spin6_f105":      dict(spin_2s=6,metal='Ni',ligand_raw='Cl2F4',n_ligands=6,geometry='oct',dist_ang=None,charge=-2,metal_row='3d',needs_ecp=False,ref_no=6, ref_ne=10),
    "Mo_N6_chg-2_spin6_oct_d2p18":    dict(spin_2s=6,metal='Mo',ligand_raw='N',  n_ligands=6,geometry='oct',dist_ang=2.18, charge=-2,metal_row='4d',needs_ecp=True, ref_no=10,ref_ne=10),
    "V_H4_chg-2_spin3_tet_d1p73":     dict(spin_2s=3,metal='V', ligand_raw='H',  n_ligands=4,geometry='tet',dist_ang=1.73, charge=-2,metal_row='3d',needs_ecp=False,ref_no=3, ref_ne=9),
    "Ru_O6_chg-3_spin3_oct_d1p947":   dict(spin_2s=3,metal='Ru',ligand_raw='O',  n_ligands=6,geometry='oct',dist_ang=1.947,charge=-3,metal_row='4d',needs_ecp=True, ref_no=9, ref_ne=9),
    "Mn_CN6_chg-2_spin3_oct_d1p95":   dict(spin_2s=3,metal='Mn',ligand_raw='CN', n_ligands=6,geometry='oct',dist_ang=1.95, charge=-2,metal_row='3d',needs_ecp=False,ref_no=7, ref_ne=9),
    "Cu_S4_chg-1_spin4":              dict(spin_2s=4,metal='Cu',ligand_raw='S',  n_ligands=4,geometry='tet',dist_ang=2.20, charge=-1,metal_row='3d',needs_ecp=False,ref_no=8, ref_ne=10),
    "Pt_NH36_chg-2_spin4_oct_d1p947": dict(spin_2s=4,metal='Pt',ligand_raw='NH3',n_ligands=6,geometry='oct',dist_ang=1.947,charge=-2,metal_row='5d',needs_ecp=True, ref_no=6, ref_ne=10),
    "Fe_I6_chg-3_spin5":              dict(spin_2s=5,metal='Fe',ligand_raw='I',  n_ligands=6,geometry='oct',dist_ang=2.70, charge=-3,metal_row='3d',needs_ecp=False,ref_no=8, ref_ne=9),
    "Ir_PH34_chg-2_spin3_tet_d2p415": dict(spin_2s=3,metal='Ir',ligand_raw='PH3',n_ligands=4,geometry='tet',dist_ang=2.415,charge=-2,metal_row='5d',needs_ecp=True, ref_no=9, ref_ne=9),
    "Co_F5sq_chg-5_spin3":            dict(spin_2s=3,metal='Co',ligand_raw='F',  n_ligands=5,geometry='oct',dist_ang=1.90, charge=-5,metal_row='3d',needs_ecp=False,ref_no=5, ref_ne=9),
    "Cr_Cl4O2_chg-3_spin5_f100":      dict(spin_2s=5,metal='Cr',ligand_raw='Cl4O2',n_ligands=6,geometry='oct',dist_ang=None,charge=-3,metal_row='3d',needs_ecp=False,ref_no=5, ref_ne=9),
    "Rh_H2O6_chg-2_spin3_oct_d1p976": dict(spin_2s=3,metal='Rh',ligand_raw='H2O',n_ligands=6,geometry='oct',dist_ang=1.976,charge=-2,metal_row='4d',needs_ecp=True, ref_no=8, ref_ne=9),
    "Pd_O6_chg-2_spin4_oct_d1p9":     dict(spin_2s=4,metal='Pd',ligand_raw='O',  n_ligands=6,geometry='oct',dist_ang=1.90, charge=-2,metal_row='4d',needs_ecp=True, ref_no=8, ref_ne=10),
    "Ti_H6_chg0_spin4_oct_d1p958":    dict(spin_2s=4,metal='Ti',ligand_raw='H',  n_ligands=6,geometry='oct',dist_ang=1.958,charge=0, metal_row='3d',needs_ecp=False,ref_no=10,ref_ne=10),
    "Ni_Cl3N1_chg-1_spin5_f95":       dict(spin_2s=5,metal='Ni',ligand_raw='Cl3N1',n_ligands=6,geometry='oct',dist_ang=None,charge=-1,metal_row='3d',needs_ecp=False,ref_no=5, ref_ne=9),
    "Mo_Br4_chg0_spin4_tet_d2p451":   dict(spin_2s=4,metal='Mo',ligand_raw='Br', n_ligands=4,geometry='tet',dist_ang=2.451,charge=0, metal_row='4d',needs_ecp=True, ref_no=6, ref_ne=10),
    "V_Cl6_chg-2_spin3_oct_d2p394":   dict(spin_2s=3,metal='V', ligand_raw='Cl', n_ligands=6,geometry='oct',dist_ang=2.394,charge=-2,metal_row='3d',needs_ecp=False,ref_no=3, ref_ne=9),
    "Ru_O4_chg-2_spin4_tet_d2p152":   dict(spin_2s=4,metal='Ru',ligand_raw='O',  n_ligands=4,geometry='tet',dist_ang=2.152,charge=-2,metal_row='4d',needs_ecp=True, ref_no=9, ref_ne=10),
    # LOW-SPIN
    "Pd_Br4_chg-2_spin0_sq_pl_d2p583":dict(spin_2s=0,metal='Pd',ligand_raw='Br', n_ligands=4,geometry='sq_pl',dist_ang=2.583,charge=-2,metal_row='4d',needs_ecp=True, ref_no=4, ref_ne=10),
    "V_Cl6_chg-2_spin1_oct_d2p166":   dict(spin_2s=1,metal='V', ligand_raw='Cl', n_ligands=6,geometry='oct',dist_ang=2.166,charge=-2,metal_row='3d',needs_ecp=False,ref_no=1, ref_ne=9),
    "Ni_Cl2F4_chg-3_spin1_f100":      dict(spin_2s=1,metal='Ni',ligand_raw='Cl2F4',n_ligands=6,geometry='oct',dist_ang=None,charge=-3,metal_row='3d',needs_ecp=False,ref_no=5, ref_ne=9),
    "Mo_H2O6_chg-2_spin0_oct_d2p09":  dict(spin_2s=0,metal='Mo',ligand_raw='H2O',n_ligands=6,geometry='oct',dist_ang=2.09, charge=-2,metal_row='4d',needs_ecp=True, ref_no=7, ref_ne=10),
    "Zn_NH34_chg-2_spin0_tet_d2p205": dict(spin_2s=0,metal='Zn',ligand_raw='NH3',n_ligands=4,geometry='tet',dist_ang=2.205,charge=-2,metal_row='3d',needs_ecp=False,ref_no=8, ref_ne=10),
    "Ti_S4_chg-2_spin0_tet_d2p28":    dict(spin_2s=0,metal='Ti',ligand_raw='S',  n_ligands=4,geometry='tet',dist_ang=2.28, charge=-2,metal_row='3d',needs_ecp=False,ref_no=10,ref_ne=10),
    "Mn_I6_chg-3_spin0":              dict(spin_2s=0,metal='Mn',ligand_raw='I',  n_ligands=6,geometry='oct',dist_ang=2.70, charge=-3,metal_row='3d',needs_ecp=False,ref_no=5, ref_ne=10),
    "Cu_C4_chg-4_spin1":              dict(spin_2s=1,metal='Cu',ligand_raw='C',  n_ligands=4,geometry='tet',dist_ang=1.90, charge=-4,metal_row='3d',needs_ecp=False,ref_no=10,ref_ne=9),
    "Pt_PH34_chg-2_spin0_tet_d2p166": dict(spin_2s=0,metal='Pt',ligand_raw='PH3',n_ligands=4,geometry='tet',dist_ang=2.166,charge=-2,metal_row='5d',needs_ecp=True, ref_no=6, ref_ne=10),
    "Fe_Cl4O2_chg-1_spin1_f105":      dict(spin_2s=1,metal='Fe',ligand_raw='Cl4O2',n_ligands=6,geometry='oct',dist_ang=None,charge=-1,metal_row='3d',needs_ecp=False,ref_no=4, ref_ne=9),
    "Ir_N6_chg-3_spin0_oct_d2p205":   dict(spin_2s=0,metal='Ir',ligand_raw='N',  n_ligands=6,geometry='oct',dist_ang=2.205,charge=-3,metal_row='5d',needs_ecp=True, ref_no=10,ref_ne=10),
    "Ru_O6_chg-2_spin0_oct_d2p152":   dict(spin_2s=0,metal='Ru',ligand_raw='O',  n_ligands=6,geometry='oct',dist_ang=2.152,charge=-2,metal_row='4d',needs_ecp=True, ref_no=10,ref_ne=10),
    "Co_H4_chg-3_spin0":              dict(spin_2s=0,metal='Co',ligand_raw='H',  n_ligands=4,geometry='tet',dist_ang=1.65, charge=-3,metal_row='3d',needs_ecp=False,ref_no=6, ref_ne=10),
    "Cr_F6_chg-4_spin0":              dict(spin_2s=0,metal='Cr',ligand_raw='F',  n_ligands=6,geometry='oct',dist_ang=1.95, charge=-4,metal_row='3d',needs_ecp=False,ref_no=4, ref_ne=10),
    "Rh_CN6_chg-3_spin0_oct_d1p909":  dict(spin_2s=0,metal='Rh',ligand_raw='CN', n_ligands=6,geometry='oct',dist_ang=1.909,charge=-3,metal_row='4d',needs_ecp=True, ref_no=5, ref_ne=10),
    "Pd_Cl4_chg-2_spin0_sq_pl_d2p185":dict(spin_2s=0,metal='Pd',ligand_raw='Cl', n_ligands=4,geometry='sq_pl',dist_ang=2.185,charge=-2,metal_row='4d',needs_ecp=True, ref_no=3, ref_ne=10),
    "V_O6_chg-2_spin1_oct_d1p995":    dict(spin_2s=1,metal='V', ligand_raw='O',  n_ligands=6,geometry='oct',dist_ang=1.995,charge=-2,metal_row='3d',needs_ecp=False,ref_no=9, ref_ne=9),
    "Ni_Cl3N1_chg-4_spin0_f100":      dict(spin_2s=0,metal='Ni',ligand_raw='Cl3N1',n_ligands=6,geometry='oct',dist_ang=None,charge=-4,metal_row='3d',needs_ecp=False,ref_no=5, ref_ne=10),
    "Mo_Cl4_chg0_spin0_tet_d2p541":   dict(spin_2s=0,metal='Mo',ligand_raw='Cl', n_ligands=4,geometry='tet',dist_ang=2.541,charge=0, metal_row='4d',needs_ecp=True, ref_no=4, ref_ne=10),
    "Zn_PH34_chg-4_spin0_tet_d2p3":   dict(spin_2s=0,metal='Zn',ligand_raw='PH3',n_ligands=4,geometry='tet',dist_ang=2.30, charge=-4,metal_row='3d',needs_ecp=False,ref_no=8, ref_ne=10),
}

ALL_SYSTEM_NAMES = list(SYSTEMS.keys())


def build_mol(name, s):
    dist    = s['dist_ang'] or 2.10
    builder = GEOM_BUILDERS.get(s['geometry'], _oct)
    lig_pos = builder(dist)[:s['n_ligands']]
    atoms   = [(s['metal'], (0.,0.,0.))]
    tmpl    = LIGAND_ATOMS.get(s['ligand_raw'], [('Cl',1)])
    for pos in lig_pos:
        r  = math.sqrt(sum(x**2 for x in pos))
        uv = tuple(x/r for x in pos) if r>0 else (1,0,0)
        off = 0.0
        for sym, nrep in tmpl:
            for _ in range(nrep):
                coord = tuple(pos[i]+uv[i]*off for i in range(3))
                atoms.append((sym, coord))
                off += 1.10
    mol = gto.Mole()
    mol.atom=atoms; mol.charge=s['charge']; mol.spin=s['spin_2s']
    mol.basis='def2-svp'; mol.unit='angstrom'; mol.symmetry=False
    if s['needs_ecp']: mol.ecp={s['metal']:'def2-svp'}
    mol.verbose=4; mol.build()
    return mol


def run_hf(mol):
    # Use RHF for singlets to prevent broken-symmetry UHF solutions
    # UHF for spin_2s=0 often converges to a high-spin broken-symmetry
    # solution with S=ln(3) entropy profile — physically incorrect
    if mol.spin == 0:
        mf = scf.RHF(mol)
        print("  [HF] Using RHF (singlet — prevents broken-symmetry UHF)")
    else:
        mf = scf.UHF(mol)
        print(f"  [HF] Using UHF (spin_2s={mol.spin})")
    mf.max_cycle=500; mf.conv_tol=1e-10
    mf.init_guess='atom'; mf.level_shift=0.2
    try:
        mf.kernel()
    except Exception:
        mf.init_guess='minao'; mf.kernel()
    if not mf.converged:
        mf.level_shift=0.5; mf.kernel()
    if not mf.converged:
        mf.init_guess='minao'; mf.level_shift=0.3; mf.damp=0.5; mf.kernel()
    if not mf.converged:
        mf.init_guess='1e'; mf.level_shift=0.5; mf.damp=0.3; mf.kernel()
    if not mf.converged:
        raise RuntimeError("UHF did not converge")
    return mf


def select_window(mf, s):
    # Handle both RHF (2D mo_coeff) and UHF (tuple of alpha/beta)
    if isinstance(mf, scf.uhf.UHF):
        n_mo=mf.mo_coeff[0].shape[1]; n_a=int(mf.mo_occ[0].sum()); n_b=int(mf.mo_occ[1].sum())
    else:
        n_mo=mf.mo_coeff.shape[1]; n_a=int((mf.mo_occ>0).sum()); n_b=n_a
    win=26 if s['spin_2s']>=5 else (24 if s['n_ligands']>=6 else 20)
    half=win//2; start=max(0,n_a-half); end=min(n_mo,start+win)
    if end-start<win: start=max(0,end-win)
    window=list(range(start,end))
    for idx in range(n_b,n_a):
        if idx not in window:
            window=list(range(min(window[0],idx),max(window[-1]+1,idx+1)))
    return window


def run_dmrg(mol, mf, window, M=100, nsweeps=30, scratch='/tmp/q40', name='s'):
    os.makedirs(scratch, exist_ok=True)
    if isinstance(mf, scf.uhf.UHF):
        occ=(mf.mo_occ[0]+mf.mo_occ[1])/2.0
    else:
        occ=mf.mo_occ/2.0
    n_e=int(round(2*occ[window].sum()))
    if (n_e-mol.spin)%2!=0: n_e+=1
    if (n_e-mol.spin)%2!=0: n_e-=2
    n_e=int(np.clip(n_e,mol.spin,2*len(window)))
    if isinstance(mf, scf.uhf.UHF):
        mo_alpha = mf.mo_coeff[0]
    else:
        mo_alpha = mf.mo_coeff
    n_mo=mo_alpha.shape[1]; all_i=list(range(n_mo))
    core=sorted([i for i in all_i if i not in window and occ[i]>1.5])
    virt=sorted([i for i in all_i if i not in window and occ[i]<0.5])
    other=sorted([i for i in all_i if i not in window and i not in core and i not in virt])
    mo_ord=mo_alpha[:,core+window+other+virt]
    mc=mcscf.CASSCF(mf.to_rhf(),len(window),n_e)
    mc.fcisolver=dmrgci.DMRGCI(mol,maxM=M,tol=1e-8)
    mc.fcisolver.scratchDirectory=scratch; mc.fcisolver.runtimeDir=scratch
    mc.fcisolver.maxIter=nsweeps; mc.fcisolver.block_extra_keyword=['num_thrds 8']
    mc.max_cycle_macro=1
    e=mc.kernel(mo_ord)[0]
    return mc, float(e), mo_ord, n_e


def _S(n_i, G_ii):
    lam=np.clip([1-2*n_i+G_ii,n_i-G_ii,n_i-G_ii,G_ii],1e-14,1.0)
    lam/=lam.sum(); return -float(np.dot(lam,np.log(lam)))


def get_rdms(mc):
    try:
        dm1,dm2=mc.fcisolver.make_rdm12(mc.ci,mc.ncas,mc.nelecas)
        print("  [RDM] make_rdm12 OK"); return dm1,dm2
    except Exception as e:
        print(f"  [RDM] MF approx ({e})")
        dm1=mc.fcisolver.make_rdm1(mc.ci,mc.ncas,mc.nelecas)
        n=mc.ncas; dm2=np.zeros((n,n,n,n))
        for i in range(n): dm2[i,i,i,i]=(dm1[i,i]/2)**2*4
        return dm1,dm2


def entropies_from_rdms(gamma,Gamma,n):
    return np.array([_S(gamma[i,i]/2,Gamma[i,i,i,i]/4) for i in range(n)])


def entropy_plateau_cas_size(ent,spin_2s,floor=0.02,min_act=2):
    # Sort all orbitals by entropy descending
    ss = np.sort(ent)[::-1]
    # Only consider orbitals above floor as potentially active
    n_sig = int(np.sum(ss > floor))
    if n_sig <= max(spin_2s, min_act):
        return max(spin_2s, min_act)
    # Find largest absolute gap among ALL orbitals (not just sig)
    # This finds the true boundary between correlated and inactive
    gaps = ss[:-1] - ss[1:]
    # But restrict the cutoff to be within [min_act, n_sig]
    # so we do not pick a gap in the near-zero tail
    gaps_restricted = gaps[:n_sig]  # only gaps within significant orbitals
    d = int(np.argmax(gaps_restricted)) + 1
    return int(np.clip(max(d, spin_2s, min_act), 2, n_sig))


def qicas_rotation(gamma,Gamma,n_win,d_cas,max_iter=300,tol=1e-7):
    n=n_win; ent0=entropies_from_rdms(gamma,Gamma,n)
    nonact=np.argsort(ent0)[:(n-d_cas)].tolist()
    fqi0=float(ent0[nonact].sum())
    print(f"  [QICAS] d_cas={d_cas} |N|={len(nonact)} F_QI(i)={fqi0:.6f}")
    def _fqi(x):
        X=x.reshape(n,n); X=(X-X.T)/2; U=scipy_expm(X); g=U@gamma@U.T
        v=0.0
        for i in nonact:
            u=U[:,i]; G=np.einsum('p,q,r,s,pqrs->',u,u,u,u,Gamma)/4
            v+=_S(g[i,i]/2,G)
        return v
    res=scipy_minimize(_fqi,np.zeros(n*n),method='L-BFGS-B',
                       options={'maxiter':max_iter,'ftol':tol,'gtol':tol*0.1,'maxfun':max_iter*20})
    X=res.x.reshape(n,n); X=(X-X.T)/2; U=scipy_expm(X); gopt=U@gamma@U.T
    eopt=np.zeros(n)
    for i in range(n):
        u=U[:,i]; G=np.einsum('p,q,r,s,pqrs->',u,u,u,u,Gamma)/4
        eopt[i]=_S(gopt[i,i]/2,G)
    fqif=float(eopt[nonact].sum())
    print(f"  [QICAS] F_QI(f)={fqif:.6f} red={fqi0-fqif:.6f} ok={res.success} nit={res.nit}")
    return U,eopt,fqi0,fqif


def build_cas_mo(mo_ord,window,active_rel,n_mo,mf):
    if isinstance(mf, scf.uhf.UHF):
        occ=(mf.mo_occ[0]+mf.mo_occ[1])/2.0
    else:
        occ=mf.mo_occ/2.0
    all_i=list(range(n_mo))
    abs_act=[window[r] for r in active_rel]
    nonact_win=[window[r] for r in range(len(window)) if r not in active_rel]
    core_nw=sorted([i for i in all_i if i not in window and occ[i]>1.5])
    virt_nw=sorted([i for i in all_i if i not in window and occ[i]<0.5])
    core_win=sorted([i for i in nonact_win if occ[i]>1.5])
    virt_win=sorted([i for i in nonact_win if occ[i]<0.5])
    other=[i for i in all_i if i not in sorted(core_nw+virt_nw+core_win+virt_win+abs_act)]
    ordered=sorted(core_nw+core_win)+abs_act+other+sorted(virt_nw+virt_win)
    return mo_ord[:,np.argsort(ordered)]


def casci_e(mf,mo,nc,ne):
    mc=mcscf.CASCI(mf.to_rhf(),nc,ne); mc.verbose=3
    return float(mc.kernel(mo)[0])


def casscf_run(mf,mo,nc,ne,max_macro=100):
    mc=mcscf.CASSCF(mf.to_rhf(),nc,ne)
    mc.max_cycle_macro=max_macro; mc.conv_tol=1e-9; mc.conv_tol_grad=1e-4; mc.verbose=4
    cnt=[0]
    def _cb(e): cnt[0]+=1
    mc.callback=_cb; t0=time.time()
    e,_,_,_,_=mc.kernel(mo)
    return dict(e_casscf=float(e),converged=bool(mc.converged),n_macro_iter=cnt[0],t_s=time.time()-t0)


def _j(obj):
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return None if np.isnan(obj) else float(obj)
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,dict): return {k:_j(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [_j(v) for v in obj]
    return obj


def run_one(name,s,M=100,sweeps=30,out_dir='results_40'):
    os.makedirs(out_dir,exist_ok=True)
    out=os.path.join(out_dir,f'qicas40_{name}.json')
    sc='HS' if s['spin_2s']>=3 else ('MS' if s['spin_2s']==2 else 'LS')
    res=dict(name=name,metal=s['metal'],ligand=s['ligand_raw'],
             charge=s['charge'],spin_2s=s['spin_2s'],spin_class=sc,
             metal_row=s['metal_row'],M_dmrg=M,
             dataset_reference=dict(n_active=s['ref_no'],n_active_e=s['ref_ne'],
                 note='Post-hoc comparison only — NOT fed into QICAS'),
             status='RUNNING')
    t0=time.time()
    try:
        print(f"\n{'='*60}\n  {name}\n  {s['metal']}/{s['ligand_raw']} chg={s['charge']} 2S={s['spin_2s']} [{sc}] {s['metal_row']}\n  Dataset ref (post-hoc): CAS({s['ref_ne']},{s['ref_no']})\n{'='*60}")
        mol=build_mol(name,s)
        res.update(n_electrons=int(mol.nelectron),n_basis=int(mol.nao_nr()))

        print("\n[Step 2] UHF..."); t1=time.time()
        mf=run_hf(mol)
        res['e_hf']=float(mf.e_tot); res['t_hf_s']=time.time()-t1
        print(f"  E(UHF)={mf.e_tot:.8f} ({res['t_hf_s']:.1f}s)")

        window=select_window(mf,s)
        res['window']=dict(size=len(window),indices=window)
        print(f"\n[Step 3] Window: {len(window)} orbitals")

        print(f"\n[Step 4] DMRG M={M}, {sweeps} sweeps..."); t1=time.time()
        scratch=f'/tmp/qicas40_{name}'
        mc_dmrg,e_dmrg,mo_ord,n_elec_win=run_dmrg(mol,mf,window,M=M,nsweeps=sweeps,scratch=scratch,name=name)
        res['e_dmrg']=float(e_dmrg); res['t_dmrg_s']=time.time()-t1; res['n_elec_window']=int(n_elec_win)
        print(f"  E(DMRG)={e_dmrg:.8f} ({res['t_dmrg_s']:.1f}s)")

        print("\n[Step 5] RDMs and entropies...")
        gamma,Gamma=get_rdms(mc_dmrg)
        n_win=len(window)
        ent=entropies_from_rdms(gamma,Gamma,n_win)
        # Store FULL entropy profile for plateau visualisation
        res['entropy_profile']=dict(
            entropies=ent.tolist(),
            noon_window=[float(gamma[i,i]) for i in range(n_win)],
            window_indices=window,
            entropy_min=float(ent.min()),entropy_max=float(ent.max()),
            entropy_sum=float(ent.sum()),spin_class=sc,spin_2s=s['spin_2s'])
        print(f"  Entropy range: [{ent.min():.4f}, {ent.max():.4f}]  sum={ent.sum():.4f}")

        d_cas=entropy_plateau_cas_size(ent,s['spin_2s'])
        print(f"\n[Step 6] Plateau: D_CAS={d_cas}  (dataset ref: {s['ref_no']})")
        res['plateau']=dict(d_cas=d_cas,dataset_ref_no=s['ref_no'],delta_vs_ref=d_cas-s['ref_no'])

        print(f"\n[Step 7] QICAS rotation (D_CAS={d_cas})..."); t1=time.time()
        U_q,ent_q,fqi_i,fqi_f=qicas_rotation(gamma,Gamma,n_win,d_cas)
        t_rot=time.time()-t1
        active_rel=np.argsort(ent_q)[::-1][:d_cas].tolist()
        abs_act=[window[r] for r in active_rel]
        if isinstance(mf, scf.uhf.UHF):
            occ_avg=(mf.mo_occ[0]+mf.mo_occ[1])/2.0
        else:
            occ_avg=mf.mo_occ/2.0
        n_active_e=int(round(sum(occ_avg[i]*2 for i in abs_act)))
        if (n_active_e-s['spin_2s'])%2!=0: n_active_e+=1
        if (n_active_e-s['spin_2s'])%2!=0: n_active_e-=2
        n_active_e=int(np.clip(n_active_e,s['spin_2s'],2*d_cas))
        res['qicas']=dict(n_active=d_cas,n_active_e=n_active_e,
            active_rel=active_rel,active_abs=abs_act,
            fqi_initial=fqi_i,fqi_final=fqi_f,fqi_reduction=fqi_i-fqi_f,
            entropies_qicas=ent_q.tolist(),t_rotation_s=t_rot)
        print(f"  QICAS: CAS({n_active_e},{d_cas})  F_QI: {fqi_i:.4f}->{fqi_f:.4f}")

        n_mo=mol.nao_nr()
        mo_hf=build_cas_mo(mo_ord,window,active_rel,n_mo,mf)
        if isinstance(mf, scf.uhf.UHF):
            occ_avg_full=(mf.mo_occ[0]+mf.mo_occ[1])/2.0
        else:
            occ_avg_full=mf.mo_occ/2.0
        n_core_nw=sum(1 for i in range(n_mo) if i not in window and occ_avg_full[i]>1.5)
        mo_qi=mo_ord.copy()
        mo_qi[:,n_core_nw:n_core_nw+n_win]=mo_ord[:,n_core_nw:n_core_nw+n_win]@U_q.T
        mo_qi=build_cas_mo(mo_qi,window,active_rel,n_mo,mf)

        print(f"\n[Step 9 / Goal 1] CASCI CAS({n_active_e},{d_cas})...")
        e_hf_ci=casci_e(mf,mo_hf,d_cas,n_active_e)
        e_qi_ci=casci_e(mf,mo_qi,d_cas,n_active_e)
        delta=(e_qi_ci-e_hf_ci)*1000
        res['goal1_casci']=dict(cas_ne=n_active_e,cas_no=d_cas,
            e_casci_hf=e_hf_ci,e_casci_qicas=e_qi_ci,delta_mha=delta,qicas_better=delta<0)
        print(f"  delta={delta:+.3f} mHa ({'QICAS better' if delta<0 else 'HF better'})")

        print(f"\n[Step 10 / Goal 2] CASSCF CAS({n_active_e},{d_cas})...")
        r_hf=casscf_run(mf,mo_hf,d_cas,n_active_e)
        r_qi=casscf_run(mf,mo_qi,d_cas,n_active_e)
        sp=r_hf['n_macro_iter']-r_qi['n_macro_iter']
        res['goal2_casscf']=dict(cas_ne=n_active_e,cas_no=d_cas,from_hf=r_hf,from_qicas=r_qi,
            iter_speedup=sp,qicas_faster=sp>0,energy_diff_mha=(r_qi['e_casscf']-r_hf['e_casscf'])*1000)
        print(f"  HF: {r_hf['n_macro_iter']} iters ({'Y' if r_hf['converged'] else 'N'})  "
              f"QICAS: {r_qi['n_macro_iter']} iters ({'Y' if r_qi['converged'] else 'N'})  speedup={sp:+d}")

        res['status']='OK'; res['wall_time_s']=time.time()-t0
        print(f"\n  DONE in {res['wall_time_s']:.0f}s")

    except Exception as exc:
        res['status']='ERROR'; res['error']=str(exc)
        res['traceback']=traceback.format_exc(); res['wall_time_s']=time.time()-t0
        print(f"\n  ERROR: {exc}"); print(traceback.format_exc())

    with open(out,'w') as f: json.dump(_j(res),f,indent=2)
    print(f"  Saved: {out}")
    return res


def main():
    p=argparse.ArgumentParser(description='Full canonical QICAS for 40 HS/LS systems')
    p.add_argument('--list_systems',action='store_true')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--system',type=str); g.add_argument('--system_index',type=int)
    p.add_argument('--M',type=int,default=100); p.add_argument('--sweeps',type=int,default=30)
    p.add_argument('--out_dir',type=str,default='results_40')
    args=p.parse_args()
    if args.list_systems:
        for i,n in enumerate(ALL_SYSTEM_NAMES):
            s=SYSTEMS[n]; sc='HS' if s['spin_2s']>=3 else 'LS'
            print(f"  [{i:2d}] [{sc}] {n}  2S={s['spin_2s']}  {s['metal_row']}  CAS({s['ref_ne']},{s['ref_no']})")
        return
    if args.system_index is not None:
        if not 0<=args.system_index<len(ALL_SYSTEM_NAMES): sys.exit("Bad index")
        name=ALL_SYSTEM_NAMES[args.system_index]
    elif args.system:
        if args.system not in SYSTEMS: sys.exit(f"Unknown: {args.system}")
        name=args.system
    else: p.print_help(); sys.exit(1)
    run_one(name,SYSTEMS[name],M=args.M,sweeps=args.sweeps,out_dir=args.out_dir)

if __name__=='__main__': main()
