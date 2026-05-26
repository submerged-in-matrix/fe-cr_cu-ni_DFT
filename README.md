# Elastic Constants of Binary Alloys — DFT → ML → FEM

A complete computational pipeline to compute elastic constants (C₁₁, C₁₂, C₄₄)
for Fe-Cr and Cu-Ni binary alloys using DFT, then build an ML surrogate model
for use as FEM material inputs across arbitrary compositions.

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
| fe04cr12 → fe03cr13 | 1e-7 | FM | Clean convergence |
| fe02cr14 → fe00cr16 | 1e-5 | AFM (CrA/CrB sublattice) | Near AFM phase boundary |

### High-Cr tags — ML flags

three high-Cr tags must be treated with care in the ML surrogate:

- `afm_init = True` — AFM sublattice initialisation used
- `conv_thr = 1e-5` — fe02cr14 h_0.0 onwards - looser threshold due to magnetic frustration and budget


The FM→AFM transition near 85% Cr may represent a genuine physical discontinuity in elastic response.

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

1. ML surrogate — Gaussian Process or Random Forest on C₁₁/C₁₂/C₄₄ vs x_Cr
2. FEM/MD — use ML-predicted elastic constants as material inputs
3. Cu-Ni DFT sweep (32-atom FCC SQS, 17 compositions)
---

## References

- Quantum ESPRESSO: Giannozzi et al., J. Phys.: Condens. Matter 21, 395502 (2009)
- SQS method: Zunger et al., Phys. Rev. Lett. 65, 353 (1990)
- BCC Cr antiferromagnetism: Fawcett, Rev. Mod. Phys. 60, 209 (1988)
