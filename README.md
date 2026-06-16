# Elastic Constants of Binary Alloys — DFT → ML Surrogate → FEM

A full computational pipeline for Fe-Cr binary alloys: DFT elastic constant computation across 17 compositions, Gaussian Process and MLP surrogate model training, and CalculiX FEM nanoindentation simulations to extract the reduced modulus Eᵣ as a function of Cr content.

Cu-Ni (FCC, 32-atom supercell) is scaffolded in the codebase but deferred — all scripts contain commented-out Cu-Ni blocks that are preserved for future use.

---

## Repository Structure

```
.
├── scripts/
│   ├── run_elastic_grid.sh         # DFT sweep: Fe-Cr (BCC, 16-atom supercell)
│   └── run_elastic_grid_afm.sh     # AFM-initialised variant for high-Cr tags
├── dft_data/                       # Extracted stress tensors (.dat per tag per strain)
├── outputs/                        # Raw QE .out files
├── pseudo/                         # UPF pseudopotential files
├── ml/
│   ├── gp_scratch.py               # GP: Matérn-5/2 kernel, Cholesky, L-BFGS-B
│   ├── data_utils.py               # Shared loader, LOO-CV, metrics, tier/flag arrays
│   ├── enrich_csv.py               # Produces raw + uncertainty-corrected CSVs
│   ├── 01_gp_surrogate.ipynb       # GP training across both datasets and all tiers
│   ├── 03_mlp.ipynb                # MLP training, numpy Adam, bootstrap uncertainty
│   ├── 04_model_comparison.ipynb   # GP vs MLP comparison, FEM material card generation
│   └── 05_ablation_fe02cr14.ipynb  # fe02cr14 removal study
├── analysis/                       # Derived outputs: CSVs, PKLs, figures, material cards
├── notebooks/
│   └── fecr_nanoindentation_analysis.ipynb   # Oliver-Pharr analysis, all compositions
├── fem/
│   └── conical/
│       ├── fe16cr00/               # CCX input, dat, frd — Fe16Cr0
│       ├── fe08cr08/               # Fe8Cr8
│       ├── fe04cr12/               # Fe4Cr12
│       └── fe00cr16/               # Fe0Cr16
├── results/
└── fe02cr14/                       # Isolated directory for the mixed-tier tag
```

---

## System and Compute

| Item | Details |
|---|---|
| DFT code | Quantum ESPRESSO 7.3.1 (`pw.x`), GPU-accelerated |
| DFT compute | Vast.ai RTX 4090 (NVHPC 24.7, CUDA 12.5, cc=89) |
| FEM code | CalculiX CCX 2.21 |
| FEM compute | CloudHPC (browser-based), local WSL for restarts |
| Local machine | Ubuntu WSL2 (Azog), Python 3.12 |

---

## Stage 1 — DFT Elastic Constants

**System:** Fe-Cr BCC, 16-atom supercell, 17 compositions (Fe16Cr0 → Fe0Cr16 in steps of 1 atom), 10 strain calculations per composition = 170 total QE calculations.

**Method:** Three independent strain types per composition — hydrostatic, volume-conserving tetragonal, and shear — each at ±ε amplitudes. Elastic constants extracted from the stress tensor:

- `C11 + 2·C12` from hydrostatic strain
- `C11 − C12` from volume-conserving tetragonal strain, denominator `3ε`
- `C44` from shear strain, `(σ[−ε] − σ[+ε]) / 4ε`

**Convergence settings by tier:**

| Tags | conv_thr | Magnetic initialisation |
|---|---|---|
| fe16cr00 – fe08cr08 | 1×10⁻⁸ | FM |
| fe07cr09 – fe03cr13 | 1×10⁻⁷ | FM |
| fe02cr14 | mixed 1×10⁻⁷ / 1×10⁻⁵ | mixed FM / AFM |
| fe01cr15, fe00cr16 | 1×10⁻⁵ | AFM sublattice |

**AFM treatment:** High-Cr tags (fe02cr14, fe01cr15, fe00cr16) use `run_elastic_grid_afm.sh` with 3 atom types (Fe, CrA, CrB), BCC sublattice parity splitting, and `starting_magnetization` values of +0.5 / +0.5 / −0.5. FM initialisation is physically indefensible at ≥87% Cr — the system crosses the AFM phase boundary and FM SCF oscillates between competing spin states regardless of numerical settings.

**Known data quality issues:**

- `fe09cr07`: post-vc-relax final SCF did not converge; geometry uncertainty propagates to all strain calculations for this tag.
- `fe02cr14`: mixed convergence tiers and mixed magnetic initialisation across its strain calculations. Subject of a dedicated ablation study (`05_ablation_fe02cr14.ipynb`).
- `JOB DONE` in QE output is not a convergence indicator — QE prints it even after `convergence NOT achieved`. The pipeline checks for stress tensor presence in the output file, not for `JOB DONE`.

---

## Stage 2 — ML Surrogate

**Models:** Gaussian Process (GP) and MLP. XGBoost was implemented and evaluated but excluded from final results — N=17 is insufficient for reliable gradient boosting.

**Datasets:** Two variants:

- `raw`: uniform noise σ = 1.0 GPa per calculation (baseline)
- `corrected`: per-constant empirical noise derived from a per-calculation uncertainty audit (`per_calc_uncertainty.csv`)

Uncertainty sources are combined via direct addition, not quadrature. Each elastic constant (C11+2C12, C11−C12, C44) has its own noise column.

**Five-tier data quality classification:**

| Tier | Tags | Basis |
|---|---|---|
| A | fe16cr00 – fe08cr08 | conv_thr 1×10⁻⁸, FM, clean vc-relax |
| B | fe07cr09 – fe03cr13 | conv_thr 1×10⁻⁷ |
| C | fe02cr14 – fe00cr16 | conv_thr 1×10⁻⁵, AFM initialisation |
| D | fe09cr07, fe02cr14 | post-vc-relax SCF non-convergence |
| E | selected tags | cubic enforcement residual (b/a ≠ 1) |

AFM-tier tags (C) enter the training set with lower sample weights and are flagged in the dataset. Non-converged FM outputs for fe02cr14, fe01cr15, and fe00cr16 are retained as diagnostics only and never enter training.

**Best models selected from `04_model_comparison.ipynb` (MAE as primary metric, R² as sanity check):**

- C11, C12: MLP raw, fe02cr14 ablated
- C44: MLP raw, full dataset
- Uncertainty bounds: GP posterior standard deviation

**Key outputs in `analysis/`:**

- `elastic_constants_fecr_raw.csv`, `elastic_constants_fecr_corrected.csv`
- `fem_material_inputs_best.csv` — MLP mean values with GP std, in GPa
- `calculix_material_cards_best.inp` — `*ELASTIC, TYPE=ANISO` cards in MPa
- `abaqus_material_cards_best.inp` — `*Elastic, type=ANISOTROPIC` cards in MPa
- All GP and MLP models as `.pkl` files (4 variants each: raw/corrected × full/ablated)

---

## Stage 3 — FEM Nanoindentation

**Setup:**

| Parameter | Value |
|---|---|
| Code | CalculiX CCX 2.21 |
| Element type | CAX6 (axisymmetric) |
| Contact formulation | SURFACE TO SURFACE, penalty |
| Indenter geometry | Conical, 70.3° half-included angle |
| Indenter material | Diamond: E = 1,141,000 MPa, ν = 0.07 |
| Units | µm / µN / MPa throughout |
| Target indentation depth | h_max ≈ 0.404 µm |
| Unloading | 90% of h_max |

**Compositions:** Fe16Cr0, Fe8Cr8, Fe4Cr12 (pending), Fe0Cr16

### Contact penalty stiffness — the central parameter

The penalty stiffness K controls whether the FEM produces physically meaningful forces. Two runs were performed across all compositions to demonstrate the K-dependency:

**v16 (K = 10,000 MPa/µm — K-saturated, physically invalid):** All four compositions ran to completion. Post-run analysis revealed that the mean contact pressure at peak load (~40 GPa) matched the penalty ceiling K × h_element, meaning the substrate elastic constants never entered the force response. The Eᵣ values were flat across all compositions at ~40 GPa regardless of material — a direct consequence of K-saturation, not material behaviour. The P-h curves converged cleanly with no solver warnings; the saturation is silent and can only be detected by checking the contact pressure against K × h_element.

**v19 (K = 5×C11 per composition — validated):** K is derived from CCX manual Formula 1 lower bound (K = 5–50 × C11). σ∞ = 0.25% × C11. K = 15×C11 was attempted first but caused contact oscillation and was abandoned. v19 ran to completion for three compositions and produced physically valid Oliver-Pharr results.

| Composition | C11 (MPa) | K = 5×C11 (MPa/µm) | σ∞ (MPa) |
|---|---|---|---|
| Fe16Cr0 | 307,430 | 1,537,150 | 769 |
| Fe8Cr8 | 300,500 | 1,502,500 | 751 |
| Fe4Cr12 | 396,510 | 1,982,550 | 991 |
| Fe0Cr16 | 438,290 | 2,191,450 | 1,096 |

The figure below shows the K-dependency directly: v16 Eᵣ values are flat at ~40 GPa for all compositions (K-saturation artefact); v19 values track the analytical Hill-VRH curve within −12% to +7%.

![Er vs Cr% — K dependency](fem/post/fecr_Er_K_dependency.png)

### Post-processing corrections (v19)

1. **Axisymmetric wedge force scaling (×180).** CalculiX expands CAX elements into a fixed 2° sector. The reaction force in the `.dat` file is for that sector, not the full 360°. Physical force = reported force × (360/2) = ×180. Without this correction the modulus is ~180× too low. Reference: CCX 2.21 manual §6.2.30, §10.3.8, §7.118.

2. **Exclusion of zero-contact unloading points.** At K = 5×C11 (lower bound), the surface recovers faster than the indenter retracts during unloading, causing contact separation before the prescribed displacement is complete. Points where P = 0 are excluded from the Oliver-Pharr fit — they represent free-space travel, not elastic unloading mechanics. These points are marked explicitly on the P-h plots.

---

## Results

### Analytical Eᵣ — Hill VRH (Primary Result)

Hill VRH averaging of the ML-predicted elastic constants to an equivalent-isotropic indentation modulus, with diamond-indenter compliance correction. Validated against pure Fe BCC: computed 229.5 GPa vs literature ~230 GPa (~0.4% error). The Barnett-Lothe/Stroh orientation-specific path was implemented and evaluated but failed validation on Cu and is excluded.

| Composition | Cr (at%) | E_VRH (GPa) | ν_VRH | Eᵣ (GPa) |
|---|---|---|---|---|
| Fe16Cr0 | 0 | 255.0 | 0.297 | 224.9 |
| Fe8Cr8 | 50 | 252.3 | 0.300 | 223.0 |
| Fe4Cr12 | 75 | 295.7 | 0.302 | 253.5 |
| Fe0Cr16 | 100 | 315.1 | 0.224 | 257.3 |

The slight dip at Fe8Cr8 relative to Fe16Cr0 is attributed to the C44 contribution to G_VRH. The rise at high Cr is driven by increasing C11 combined with the anomalously low C12 (66 GPa) in Fe0Cr16, which reflects the AFM ground state of near-pure Cr.

### Oliver-Pharr FEM Results (v19)

| Composition | Cr (at%) | h_max (µm) | P_max (µN) | m | R² | S (µN/µm) | Eᵣ_FEM (GPa) | Eᵣ_analytical (GPa) | Deviation |
|---|---|---|---|---|---|---|---|---|---|
| Fe16Cr0 | 0 | 0.404 | 42,660 | 1.513 | 0.99996 | 274,013 | 196.9 | 224.9 | −12.4% |
| Fe8Cr8 | 50 | 0.404 | 30,784 | 2.428 | 0.99773 | 352,312 | 220.5 | 223.0 | −1.1% |
| Fe4Cr12 | 75 | — | — | — | — | — | pending | 253.5 | — |
| Fe0Cr16 | 100 | 0.404 | 51,909 | 2.149 | 0.99985 | 376,424 | 274.0 | 256.6 | +6.8% |

`Eᵣ_FEM` is the Oliver-Pharr reduced modulus after diamond-indenter compliance correction: 1/Eᵣ_sample = 1/Eᵣ − (1−νᵢ²)/Eᵢ. The ×180 wedge correction and zero-contact point exclusion are applied before fitting. Deviations of −12% to +7% relative to the analytical values are consistent with the lower-bound K, idealised conical geometry, and polycrystalline averaging in the analytical route.

---

## Known Limitations

- DFT data quality degrades with increasing Cr content. Compositions above ~56% Cr (fe07cr09 and higher) use looser convergence thresholds and/or AFM magnetic treatment, and carry larger uncertainty in the ML training set.
- Fe0Cr16 elastic constants reflect an AFM ground state. This may differ from the paramagnetic or disordered magnetic state relevant to room-temperature experimental measurement. Its Eᵣ values should be treated with caution.
- FEM results use a conical indenter geometry and a lower-bound penalty stiffness. Real nanoindentation uses a Berkovich pyramid tip and involves elastic-plastic deformation. The FEM here is purely elastic and serves as a verification of the pipeline, not a direct experimental prediction.
- The Oliver-Pharr power-law exponent m is expected to lie between 1.5 and 2.0 for standard elastic contact. The Fe16Cr0 value (m = 1.51) is at the lower end of this range; Fe8Cr8 (m = 2.43) and Fe0Cr16 (m = 2.15) are above it. These deviations are likely a consequence of the lower-bound K producing a softer effective contact response near unloading.
- Cu-Ni system is not included in the current results.

---

## Reproducing the Results

**Dependencies:** Quantum ESPRESSO 7.3.1, Python 3.12, numpy, scipy, matplotlib, scikit-learn, PyTorch, CalculiX CCX 2.21

**DFT sweep (Fe-Cr):**

```bash
# FM compositions (fe16cr00 – fe03cr13)
nohup bash scripts/run_elastic_grid.sh > elastic_grid.log 2>&1 &

# AFM compositions (fe02cr14, fe01cr15, fe00cr16)
nohup bash scripts/run_elastic_grid_afm.sh > elastic_grid_afm.log 2>&1 &
```

**ML surrogate:** Run notebooks in order:
`01_gp_surrogate.ipynb` → `03_mlp.ipynb` → `04_model_comparison.ipynb`

**FEM nanoindentation:**

```bash
# Run from the composition directory, e.g. fem/conical/fe16cr00/
ccx indentation_fe16cr00_v19

# Post-process .dat file with the Oliver-Pharr notebook
# notebooks/fecr_nanoindentation_analysis.ipynb
# Set FEM_CONICAL_DIR to point to fem/conical/
```

Key CCX parameters to preserve across any restarts:
- `*CONTROLS, PARAMETERS=CONTACT` with kscalemax=1 (prevents silent K degradation)
- `*CONTROLS` must be re-declared after every `*RESTART, READ`
- Copy `.rout` → `.rin` immediately before each restart launch

---

## Implementation Notes

- **`JOB DONE` is not a convergence gate.** QE prints it even after `convergence NOT achieved`. Always check stress tensor presence.
- **C11−C12 denominator is 3ε**, not 4ε, for the volume-conserving tetragonal strain as derived in `run_elastic_grid.sh`.
- **Wedge scaling:** CAX6 elements in CCX produce forces for a 2° sector. Multiply by 180 to recover the full 360° physical force before any modulus calculation.
- **kscalemax=1 is mandatory.** The default CCX behaviour reduces the penalty stiffness K by `kscalemax` when the face-to-face contact iteration limit is hit. With kscalemax > 1 this degradation is silent and carries into subsequent restarts undetected.
- **Penalty contact convergence does not guarantee physical correctness.** Always verify that the observed mean contact pressure is below the ceiling K × h_element. This was the failure mode of the v16 runs.
- **UNSYM is not a valid CCX 2.21 `*STEP` parameter.** It is Abaqus syntax and is silently ignored by CCX. All runs in this project used the symmetric spooles solver, which is correct for frictionless contact.
- **Bash `sed -i` edits do not affect a running script process.** Changes take effect only on the next launch.
