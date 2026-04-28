#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Elastic Constants of BCC Fe via Stress-Strain DFT
# ══════════════════════════════════════════════════════════════
#
# Computes C11, C12, C44 by applying three types of deformation
# and extracting the stress tensor from each.
#
# Deformations applied at strains: -0.02, -0.01, 0, +0.01, +0.02
# Using the conventional cubic BCC cell (2 atoms)
#
# Usage:
#   cd ~/dft_projects/fe_cr-E_UTS-dft
#   bash scripts/run_elastic_fe.sh

set -e

PW=~/q-e/bin/pw.x
ALAT_BOHR=5.355
# Convert to Angstrom: 5.355 * 0.529177 = 2.83384
ALAT_ANG=2.83384

STRAINS="-0.02 -0.01 0.00 0.01 0.02"

mkdir -p inputs outputs dft_data

echo "══════════════════════════════════════════════════════"
echo "  Elastic Constants: BCC Fe (a = ${ALAT_ANG} Å)"
echo "══════════════════════════════════════════════════════"

# ── Common settings (written to each input file) ──────────────
write_header() {
    local prefix=$1
    local a1x=$2 a1y=$3 a1z=$4
    local a2x=$5 a2y=$6 a2z=$7
    local a3x=$8 a3y=$9 a3z=${10}

    cat << EOF
&CONTROL
    calculation = 'scf',
    prefix = '${prefix}',
    pseudo_dir = './pseudo/',
    outdir = './tmp/',
    tstress = .true.
/
&SYSTEM
    ibrav = 0,
    nat = 2,
    ntyp = 1,
    ecutwfc = 80.0,
    ecutrho = 640.0,
    occupations = 'smearing',
    smearing = 'mv',
    degauss = 0.02,
    nspin = 2,
    starting_magnetization(1) = 0.5
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF
CELL_PARAMETERS angstrom
${a1x}  ${a1y}  ${a1z}
${a2x}  ${a2y}  ${a2z}
${a3x}  ${a3y}  ${a3z}
ATOMIC_POSITIONS crystal
Fe 0.0 0.0 0.0
Fe 0.5 0.5 0.5
K_POINTS automatic
12 12 12 0 0 0
EOF
}

# ══════════════════════════════════════════════════════════════
# DEFORMATION 1: Hydrostatic (isotropic volume change)
# ε_ij = ε * δ_ij → all three axes scaled by (1+ε)
# Stress response: P = -(C11 + 2*C12)/3 * 3ε = -B * 3ε
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Deformation 1: Hydrostatic ──────────────────────"

for EPS in $STRAINS; do
    A=$(echo "${ALAT_ANG} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
    PREFIX="fe_hydro_${EPS}"
    FNAME="inputs/fe_elastic_hydro_${EPS}.in"

    write_header "$PREFIX" \
        "$A" "0.000000" "0.000000" \
        "0.000000" "$A" "0.000000" \
        "0.000000" "0.000000" "$A" > "$FNAME"

    echo "  [hydro ε=${EPS}] a=${A} Å — running..."
    $PW < "$FNAME" > "outputs/fe_elastic_hydro_${EPS}.out" 2>&1

    STRESS=$(grep "P=" "outputs/fe_elastic_hydro_${EPS}.out" | tail -1 | awk -F'P=' '{print $2}' | awk '{print $1}')
    echo "  [hydro ε=${EPS}] P = ${STRESS} kbar"
    echo "${EPS} ${STRESS}" >> dft_data/fe_elastic_hydro.dat
done

# ══════════════════════════════════════════════════════════════
# DEFORMATION 2: Tetragonal (volume-conserving)
# Stretch along z by (1+ε), compress x,y by (1+ε)^(-1/2)
# Stress response: σ33 - σ11 = (C11 - C12) * (3ε/2) for small ε
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Deformation 2: Tetragonal ───────────────────────"

for EPS in $STRAINS; do
    if [ "$EPS" = "0.00" ]; then
        AXY=$(printf "%.6f" $ALAT_ANG)
        AZ=$(printf "%.6f" $ALAT_ANG)
    else
        # a_z = a*(1+eps), a_xy = a*(1+eps)^(-1/2)
        AZ=$(echo "${ALAT_ANG} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
        AXY=$(echo "${ALAT_ANG} / sqrt(1 + ${EPS})" | bc -l | xargs printf "%.6f")
    fi
    PREFIX="fe_tetra_${EPS}"
    FNAME="inputs/fe_elastic_tetra_${EPS}.in"

    write_header "$PREFIX" \
        "$AXY" "0.000000" "0.000000" \
        "0.000000" "$AXY" "0.000000" \
        "0.000000" "0.000000" "$AZ" > "$FNAME"

    echo "  [tetra ε=${EPS}] a_xy=${AXY}, a_z=${AZ} — running..."
    $PW < "$FNAME" > "outputs/fe_elastic_tetra_${EPS}.out" 2>&1

    # Extract σ11 and σ33
    S11=$(grep -A 3 "total   stress" "outputs/fe_elastic_tetra_${EPS}.out" | tail -3 | head -1 | awk '{print $4}')
    S33=$(grep -A 3 "total   stress" "outputs/fe_elastic_tetra_${EPS}.out" | tail -1 | awk '{print $6}')
    SDIFF=$(echo "${S33} - ${S11}" | bc -l | xargs printf "%.4f")
    echo "  [tetra ε=${EPS}] σ11=${S11}, σ33=${S33}, Δσ=${SDIFF} kbar"
    echo "${EPS} ${S11} ${S33} ${SDIFF}" >> dft_data/fe_elastic_tetra.dat
done

# ══════════════════════════════════════════════════════════════
# DEFORMATION 3: Monoclinic shear (for C44)
# Apply shear: a1 = a(1, ε, 0), a2 = a(ε, 1, 0), a3 = a(0, 0, 1)
# Stress response: σ12 = C44 * 2ε (engineering shear = 2ε)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Deformation 3: Shear ────────────────────────────"

for EPS in $STRAINS; do
    SHEAR=$(echo "${ALAT_ANG} * ${EPS}" | bc -l | xargs printf "%.6f")
    PREFIX="fe_shear_${EPS}"
    FNAME="inputs/fe_elastic_shear_${EPS}.in"

    write_header "$PREFIX" \
        "$ALAT_ANG" "$SHEAR" "0.000000" \
        "$SHEAR" "$ALAT_ANG" "0.000000" \
        "0.000000" "0.000000" "$ALAT_ANG" > "$FNAME"

    echo "  [shear ε=${EPS}] γ=${SHEAR} — running..."
    $PW < "$FNAME" > "outputs/fe_elastic_shear_${EPS}.out" 2>&1

    # Extract σ12 (off-diagonal stress)
    S12=$(grep -A 3 "total   stress" "outputs/fe_elastic_shear_${EPS}.out" | tail -3 | head -1 | awk '{print $5}')
    echo "  [shear ε=${EPS}] σ12=${S12} kbar"
    echo "${EPS} ${S12}" >> dft_data/fe_elastic_shear.dat
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL DONE. Data saved to dft_data/"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Files:"
echo "  dft_data/fe_elastic_hydro.dat  — ε vs P (kbar)"
echo "  dft_data/fe_elastic_tetra.dat  — ε vs σ11, σ33, Δσ (kbar)"
echo "  dft_data/fe_elastic_shear.dat  — ε vs σ12 (kbar)"
echo ""
echo "Next: run scripts/fit_elastic_constants.py to extract C11, C12, C44"
