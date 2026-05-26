# Elastic Constants of Binary Alloys — DFT → ML → FEM

A complete computational pipeline to compute elastic constants (C₁₁, C₁₂, C₄₄)
for Fe-Cr and Cu-Ni binary alloys using DFT, then build an ML surrogate model
for use as FEM material inputs across arbitrary compositions.

---

## Project Status

| Stage | System | Status |
|---|---|---|
| DFT — elastic constant sweep | Fe-Cr (BCC, 16-atom SQS) | ✅ Complete — 17/17 tags |
| DFT — elastic constant sweep | Cu-Ni (FCC, 32-atom SQS) | ⏳ Pending |
| Post-processing — extraction | Fe-Cr | ✅ Complete |
| ML surrogate | Fe-Cr | 🔄 Next |
| FEM/MD integration | — | ⏳ Pending |

---

## Repository Structure

```
fe-cr_cu-ni_DFT/
│
├── scripts/
│   ├── run_elastic_grid.sh          # Main DFT sweep — Fe-Cr tags 0–13 Cr
│   └── run_elastic_grid_afm.sh      # AFM variant — high-Cr tags (14–16 Cr)
│
├── dft_data/                        # Extracted stress data per tag
│   ├── fecr_fe16cr00_hydro.dat      # cols: EPS  P_kbar
│   ├── fecr_fe16cr00_tetra.dat      # cols: EPS  S11  S33  DS
│   ├── fecr_fe16cr00_shear.dat      # cols: EPS  S12
│   └── ...  (3 files × 17 tags)
│
├── outputs/                         # QE output files (.out) — all strain calcs
│   └── fecr_fe**cr**_*.out
│
├── fe02cr14/                        # Diagnostic files — convergence study
│   └── fecr_fe02cr14_*.out          # Pre-AFM failed runs + converged runs
│
├── analysis/                        # Post-processing results
│   ├── elastic_constants_fecr.csv   # C11, C12, C44, B, A for all 17 tags
│   ├── elastic_constants_summary.txt
│   ├── stress_comparison_summary.txt
│   ├── fe02cr14_convergence_study.txt
│   └── *.png                        # All plots
│
├── notebooks/
│   ├── elastic_extraction.ipynb             # C11/C12/C44 extraction — all tags
│   ├── stress_comparison_thr5_vs_thr7.ipynb # conv_thr benchmark — fe02cr14
│   └── fe02cr14_convergence_study.ipynb     # Magnetic convergence study
│
└── docs/
    ├── session_summary_may24_2026.md
    ├── session_summary_may25_2026.md
    └── session_handoff_may23.md
```

---

## Physical Background

The elastic tensor fully characterises the linear mechanical response of a
crystal. For cubic symmetry (BCC Fe-Cr, FCC Cu-Ni) only three independent
constants exist: C₁₁ (resistance to uniaxial strain), C₁₂ (transverse
response to uniaxial strain), and C₄₄ (resistance to shear). These feed
directly into FEM simulations as material inputs and into MD potentials as
validation targets.

### Why DFT and not experiment
Elastic constants for arbitrary binary alloy compositions are not tabulated
experimentally. DFT with the stress-strain method allows systematic
computation across the full composition range at a fraction of the cost of
growing and testing alloy samples for each composition.

### Special Quasirandom Structures (SQS)
A binary alloy at composition x is not a perfect crystal — atoms occupy sites
randomly. An SQS supercell mimics this disorder by constructing a small
periodic cell whose correlation functions best match those of a truly random
alloy. Here we use 16-atom BCC SQS cells for Fe-Cr and 32-atom FCC cells for
Cu-Ni, generated prior to this work. Each SQS cell represents one composition
(one "tag") in the sweep.

### Magnetism in Fe-Cr and why it complicated the calculations
BCC Fe is ferromagnetic. BCC Cr is the only elemental metal that is
antiferromagnetic at room temperature (Néel temperature ~311 K, spin-density
wave ground state). In Fe-Cr alloys the magnetic ground state transitions from
ferromagnetic at Fe-rich compositions toward antiferromagnetic at Cr-rich
compositions. The exact transition composition in a disordered SQS cell is not
known a priori.

For compositions up to ~81% Cr (13 Cr atoms out of 16) the SCF converged
cleanly with ferromagnetic initialisation. For the three highest-Cr tags
(14–16 Cr atoms, 87.5–100% Cr) the SCF stalled — the system was trapped
between competing spin configurations. The diagnostic signature was:
total magnetisation frozen at 3.16 µB (a QE two-Fermi-level artefact of
ferromagnetic initialisation) while absolute magnetisation oscillated 5–8 µB
for 300 iterations, indicating the spin texture was never stabilised.

The fix was AFM sublattice initialisation: Cr atoms were split into two
species (CrA, CrB) by BCC sublattice parity, with opposite starting
magnetisations (+0.5, −0.5). This provides a physically appropriate starting
point for Cr-rich BCC and allowed all three tags to converge in ~50 iterations.
See `notebooks/fe02cr14_convergence_study.ipynb` for the full diagnostic.

---

## DFT Setup

**Code:** Quantum ESPRESSO 7.3.1  
**GPU:** NVIDIA RTX 4090 (Vast.ai), NVHPC 24.7, CUDA 12.5, cc=89  
**Pseudopotentials:** ONCV/PAW PBE — Fe.pbe-spn-kjpaw_psl.1.0.0.UPF, Cr.pbe-spn-kjpaw_psl.1.0.0.UPF  
**k-points:** 6×6×6 Monkhorst-Pack  
**Cutoffs:** ecutwfc=60 Ry, ecutrho=480 Ry  
**Smearing:** Marzari-Vanderbilt, degauss=0.02 Ry  

### Strain modes and elastic constant extraction

Three independent strain modes per tag, ε = ±0.01:

| Mode | Deformation | Extracts |
|---|---|---|
| Hydrostatic | diag(1+ε, 1+ε, 1+ε) | C₁₁ + 2C₁₂ |
| Tetragonal | volume-conserving c/a distortion | C₁₁ − C₁₂ |
| Shear | symmetric off-diagonal a₁₂=a₂₁=a·ε | C₄₄ |

Finite difference formulas:

```
C11+2C12 = (P(-ε) - P(+ε)) / 2ε
C11-C12  = -(DS(+ε) - DS(-ε)) / 3ε     [DS = S33 - S11]
C44      = (S12(-ε) - S12(+ε)) / 4ε
C11      = (C11+2C12 + 2*(C11-C12)) / 3
C12      = (C11+2C12 - (C11-C12)) / 3
```

All stresses in kbar → divide by 10 for GPa.

---

## Fe-Cr Dataset Notes

### Convergence tiers

| Tags | conv_thr | Magnetic init | Notes |
|---|---|---|---|
| fe16cr00 → fe06cr10 | 1e-8 | FM | Clean convergence |
| fe06cr10 → fe03cr13 | 1e-7 | FM | Clean convergence |
| fe02cr14 → fe00cr16 | 1e-5 | AFM (CrA/CrB sublattice) | Near AFM phase boundary |

### High-Cr tags — ML flags

The three high-Cr tags must be treated with care in the ML surrogate:

- `afm_init = True` — AFM sublattice initialisation used for all strain calcs
- `conv_thr = 1e-5` — looser threshold due to magnetic frustration and budget

*the FM→AFM transition near 85% Cr may represent a genuine physical discontinuity in elastic response.*

### fe02cr14 convergence study

See `fe02cr14/` and `notebooks/fe02cr14_convergence_study.ipynb` for full
documentation of why FM initialisation failed and how AFM sublattice
splitting resolved it. Key finding: `tot_magnetization` appeared stable at
3.16 µB (QE two-Fermi-level artefact) while absolute magnetisation
oscillated 5–8 µB for 300 iterations — spin texture was never stabilised.

---

## Reproducing the DFT calculations

Pseudopotentials are not included in this repository. Download from
[PSlibrary](https://pseudopotentials.quantum-espresso.org/legacy_tables/ps-library)
and place in `pseudo/`.

QE compilation (Vast.ai RTX 4090, NVHPC 24.7):
```bash
./configure \
    --with-cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5 \
    --with-cuda-runtime=12.5 \
    --with-cuda-cc=89 \
    --enable-openmp
make -j8 pw
```

Run:
```bash
cd fe-cr_cu-ni_DFT
nohup bash scripts/run_elastic_grid.sh > elastic_grid.log 2>&1 &
```

---

## Next Steps

1. Cu-Ni DFT sweep (32-atom FCC SQS, 17 compositions)
2. ML surrogate — Gaussian Process or Random Forest on C₁₁/C₁₂/C₄₄ vs x_Cr
3. FEM/MD — use ML-predicted elastic constants as material inputs

---

## References

- Quantum ESPRESSO: Giannozzi et al., J. Phys.: Condens. Matter 21, 395502 (2009)
- SQS method: Zunger et al., Phys. Rev. Lett. 65, 353 (1990)
- BCC Cr antiferromagnetism: Fawcett, Rev. Mod. Phys. 60, 209 (1988)
