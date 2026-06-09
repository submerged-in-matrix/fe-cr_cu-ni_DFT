# Elastic Constants of Binary Alloys — DFT → ML Surrogate → FEM

A full computational pipeline for Fe-Cr binary alloys: DFT elastic constant computation across 17 compositions, GP and MLP surrogate model training, and CalculiX FEM nanoindentation simulations to extract reduced modulus Eᵣ as a function of Cr content.

Cu-Ni (FCC, 32-atom supercell) is scaffolded in the codebase but deferred — all scripts contain commented-out Cu-Ni blocks.

---

## Repository Structure

```
.
├── scripts/
│   ├── run_elastic_grid.sh        # DFT sweep: Fe-Cr (BCC, 16-atom supercell)
│   └── run_elastic_grid_afm.sh    # AFM-initialised variant for high-Cr tags
├── inputs/                        # Generated QE input files per tag
├── dft_data/                      # Extracted stress tensors (hydro/tetra/shear .dat per tag)
├── outputs/                       # Raw QE .out files
├── pseudo/                        # UPF pseudopotential files
├── ml/
│   ├── gp_scratch.py              # GP implementation: RBF + Matérn-5/2, Cholesky, L-BFGS-B
│   ├── data_utils.py              # Shared loader, LOO-CV, metrics, tier/flag arrays
│   ├── enrich_csv.py              # Produces raw + corrected CSVs with per-constant noise
│   ├── 01_gp_surrogate.ipynb      # GP training across both datasets and all tiers
│   ├── 03_mlp.ipynb               # MLP training, numpy Adam, bootstrap uncertainty
│   ├── 04_model_comparison.ipynb  # GP vs MLP comparison, FEM material card generation
│   └── 05_ablation_fe02cr14.ipynb # fe02cr14 removal study
├── analysis/                      # All derived outputs: CSVs, PKLs, figures, material cards
├── notebooks/
│   └── fecr_nanoindentation_analysis.ipynb
├── fem/
│   └── conical/
│       ├── fe16cr00/              # CCX input, dat, frd files — Fe16Cr0
│       ├── fe08cr08/              # Fe8Cr8
│       ├── fe04cr12/              # Fe4Cr12
│       └── fe00cr16/              # Fe0Cr16
└── results/
```

---

## System and Compute

| Item | Details |
|---|---|
| DFT code | Quantum ESPRESSO 7.3.1 (`pw.x`), GPU-accelerated |
| DFT compute | Vast.ai RTX 4090 (NVHPC 24.7, CUDA 12.5, cc=89) |
| FEM code | CalculiX CCX 2.21 |
| FEM compute | CloudHPC free-tier |
| Local machine | Ubuntu WSL (Azog), Python 3.12 |

---

## Stage 1 — DFT Elastic Constants

**System:** Fe-Cr BCC, 16-atom supercell, 17 compositions (Fe16Cr0 → Fe0Cr16 in steps of 1 atom), 10 strain calculations per composition = 170 total QE calculations.

**Method:** Three independent strain types per composition — hydrostatic, volume-conserving tetragonal, and shear — each at ±ε amplitudes. Elastic constants extracted from the stress tensor:

- `C11 + 2·C12` from hydrostatic strain
- `C11 − C12` from tetragonal strain, denominator `3ε`
- `C44` from shear strain, sign convention `(σ[−ε] − σ[+ε]) / 4ε`

**Convergence settings:**

| Tags | conv_thr | Magnetic init |
|---|---|---|
| fe16cr00 – fe07cr09 | 1e-8 | FM |
| fe06cr10 – fe03cr13 | 1e-7 | FM |
| fe02cr14 | mixed 1e-7 / 1e-5 | mixed FM / AFM |
| fe01cr15, fe00cr16 | 1e-5 | AFM sublattice |

**AFM treatment:** High-Cr tags (fe02cr14, fe01cr15, fe00cr16) use `run_elastic_grid_afm.sh` with 3 atom types (Fe, CrA, CrB), BCC sublattice parity splitting, and `starting_magnetization` +0.5/+0.5/−0.5. FM initialisation is physically indefensible at ≥87% Cr — the system crosses the AFM phase boundary and FM SCF oscillates between competing spin states regardless of numerical settings.

**Known data quality issues:**

- `fe09cr07`: post-BFGS final SCF not converged (best accuracy 6e-10, 100-iteration budget); geometry uncertainty propagates to all strain calculations
- `fe02cr14`: mixed conv_thr tiers, mixed FM/AFM initialisation; subject of dedicated ablation study (`05_ablation_fe02cr14.ipynb`)
- `JOB DONE` in QE output is not a convergence indicator — the pipeline checks for stress tensor presence, not `JOB DONE`

---

## Stage 2 — ML Surrogate

**Models:** Gaussian Process (GP) and MLP. XGBoost was implemented and evaluated but excluded from final results — N=17 compositions is too small for reliable gradient boosting.

**Datasets:** Two variants —
- `raw`: uniform σ = 1.0 GPa per calculation (baseline)
- `corrected`: per-constant empirical noise derived from per-calculation uncertainty audit (`per_calc_uncertainty.csv`)

**5-tier data quality classification:**

| Tier | Tags | Basis |
|---|---|---|
| A | n_cr 0–9 | conv_thr 1e-8, FM, clean VCR |
| B | n_cr 10–13 | conv_thr 1e-7 |
| C | n_cr 14–16 | conv_thr 1e-5, AFM init |
| D | fe09cr07, fe02cr14 | post-BFGS SCF non-convergence |
| E | selected tags | cubic enforcement residual (b/a ≠ 1) |

**Best models (from `04_model_comparison.ipynb`):**
- C11, C12: MLP raw ablated (fe02cr14 removed)
- C44: MLP raw full
- Uncertainty bounds: GP posterior standard deviation

**Key outputs in `analysis/`:**
- `elastic_constants_fecr_raw.csv`, `elastic_constants_fecr_corrected.csv`
- `fem_material_inputs_best.csv` — MLP mean values with GP std, in GPa
- `calculix_material_cards_best.inp` — CalculiX `*ELASTIC TYPE=ANISO` cards in MPa
- `abaqus_material_cards_best.inp` — ABAQUS `*Elastic, type=ANISOTROPIC` cards in MPa
- All GP and MLP results as `.pkl` files (4 variants each: raw/corrected × full/no14)

---

## Stage 3 — FEM Nanoindentation

**Setup:**
- Code: CalculiX CCX 2.21, axisymmetric model
- Elements: CAX6
- Contact: SURFACE TO SURFACE, face identifier S3
- Indenter: 70.3° conical diamond, E = 1,141,000 MPa, ν = 0.07
- Units throughout: µm / µN / MPa
- Indentation depth: h_max = 1.01 µm, unloading to 90% of h_max

**Compositions simulated:** Fe16Cr0, Fe8Cr8, Fe4Cr12, Fe0Cr16

**v16 runs — completed, physically invalid:**
All four compositions ran to completion. A contact penalty stiffness of K = 10,000 MPa/µm was used. Post-run analysis revealed the mean contact pressure at peak load (770–793 MPa) sat at the penalty ceiling K × h_element, meaning the substrate elastic constants never entered the force response. The v16 P(h) curves and any Eᵣ values derived from them are physically invalid. Comparison of v16 against the partial v17 run confirmed the artefact is present from the first contact increment — there is no depth range where v16 is correct, and no post-processing correction is possible.

**v17 run — incomplete (budget exhausted):**
A corrected-K rerun (K = 150,000 MPa/µm) was launched for Fe16Cr0 only as a pipeline validation test. The run was stopped at h = 0.70 µm when CloudHPC free-tier hours (302 vCPU-hrs) were exhausted. No unloading segment was computed; Oliver-Pharr analysis was not possible. The partial P(h) curve is retained in `fem/conical/fe16cr00/` for reference.

**What the FEM produced that is valid:**
- Displacement field (U2) and mesh integrity confirmed via CGX visualisation for all four v16 compositions — these are geometrically correct even though the force magnitudes are not
- The K-saturation artefact is visually demonstrable from the v16 stress contours (SYY flat at ~770 MPa under the indenter tip)

**FEM is discontinued** at this stage due to compute budget. The analytical route below is the primary result.

---

## Primary Result — Analytical Eᵣ

Hill VRH averaging of DFT elastic constants → equivalent-isotropic indentation modulus, with indenter compliance correction:

`Eᵣ = 1 / ( (1 − ν²) / E_VRH + (1 − ν_i²) / E_i )`

Validated against pure Fe BCC: computed 229.5 GPa vs literature 230.4 GPa (0.4% error). The Barnett-Lothe/Stroh orientation-specific path was implemented and evaluated but failed validation (Cu: 33 GPa computed vs ~134 GPa expected) and is excluded.

| Composition | Cr (at%) | E_VRH (GPa) | ν_VRH | Eᵣ (GPa) |
|---|---|---|---|---|
| Fe16Cr0 | 0 | 255.0 | 0.297 | 224.9 |
| Fe8Cr8 | 50 | 252.3 | 0.300 | 223.2 |
| Fe4Cr12 | 75 | 295.7 | 0.302 | 253.5 |
| Fe0Cr16 | 100 | 315.1 | 0.224 | 257.3 |

The slight dip at Fe8Cr8 relative to Fe16Cr0 is attributed to the C44 contribution to G_VRH. The steep rise at high Cr is driven by the increase in C11 and the anomalously low C12 (66 GPa) in Fe0Cr16, which is a consequence of the AFM magnetic state at near-pure Cr compositions.

---

## Known Limitations

- DFT sweep covers 11 of 17 compositions with tier-A or tier-B quality data. Compositions fe05cr11 through fe00cr16 carry looser convergence thresholds and/or AFM magnetic complications — their elastic constants carry larger uncertainty and enter the ML training set with reduced effective weight.
- Fe0Cr16 elastic constants (particularly C12 = 66 GPa) reflect an AFM ground state that may differ from the paramagnetic or FM state relevant to room-temperature experimental measurements. This composition's Eᵣ value should be treated with caution.
- The FEM Oliver-Pharr route was not completed. The analytical Eᵣ values are derived from polycrystalline VRH averages and do not account for crystallographic texture or orientation-dependent indentation response.
- Cu-Ni system is not included in the current results.

---

## Reproducing the Results

**Dependencies:** Quantum ESPRESSO 7.3.1, Python 3.12, numpy, scipy, matplotlib, scikit-learn (GP baseline only), PyTorch (MLP), CalculiX CCX 2.21, CGX 2.21

**DFT sweep (Fe-Cr):**
```bash
nohup bash scripts/run_elastic_grid.sh > elastic_grid.log 2>&1 &
# High-Cr AFM tags:
nohup bash scripts/run_elastic_grid_afm.sh > elastic_grid_afm.log 2>&1 &
```

**Elastic constant extraction and ML:**
Run notebooks in order: `01_gp_surrogate.ipynb` → `03_mlp.ipynb` → `04_model_comparison.ipynb`

**FEM (if resuming):**
- Correct K value: 150,000 MPa/µm (as used in v17)
- Input files in `fem/conical/<comp>/`
- CloudHPC: upload via three-folder approach; select calculiX-2.21; download CALCULIX.tar.gz

---

## Notes on Key Implementation Decisions

- **`JOB DONE` is not a convergence gate.** QE prints it even after `convergence NOT achieved`. The pipeline checks for stress tensor presence in the output file.
- **Bash `sed -i` edits do not affect a running script process.** Changes only take effect on the next launch.
- **C11−C12 denominator is 3ε**, not 4ε, for the volume-conserving tetragonal strain as defined in `run_elastic_grid.sh`.
- **CCX face identifier S3** (not S1) is the correct top face for CAX6 SURFACE TO SURFACE contact in this mesh.
- **UNSYM solver is required** for all CCX steps when SURFACE TO SURFACE contact is active.
- **Penalty contact convergence does not guarantee physical correctness.** Always verify that observed contact pressure is below K × h_element.
