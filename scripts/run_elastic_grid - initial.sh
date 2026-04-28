#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Elastic Constants of Fe-Cr BCC Alloys (16-atom supercell)
# Full composition grid: Fe16 → Cr16 in steps of 6.25%
# ══════════════════════════════════════════════════════════════
#
# For each composition:
#   1. vc-relax to find equilibrium lattice parameter
#   2. Hydrostatic deformation (5 strains) → B = (C11+2C12)/3
#   3. Tetragonal deformation (5 strains) → C11-C12
#   4. Shear deformation (5 strains) → C44
#
# Usage:
#   cd ~/dft_projects/fe_cr-E_UTS-dft
#   bash scripts/run_elastic_grid.sh
#
# Estimated time: 2-4 days (255 SCF + 17 vc-relax)

set -e

PW=~/q-e/bin/pw.x
STRAINS="-0.02 -0.01 0.00 0.01 0.02"

# Fe and Cr equilibrium lattice parameters (Bohr) — from our optimization
A_FE_BOHR=5.355
A_CR_BOHR=5.44   # Will be refined by vc-relax of pure Cr

mkdir -p inputs outputs dft_data

# ── 16 atomic positions in 2x2x2 BCC supercell (fractional) ──
# Positions are fixed; only the species (Fe/Cr) changes
POSITIONS=(
    "0.0000 0.0000 0.0000"   # 0
    "0.2500 0.2500 0.2500"   # 1
    "0.0000 0.0000 0.5000"   # 2
    "0.2500 0.2500 0.7500"   # 3
    "0.0000 0.5000 0.0000"   # 4
    "0.2500 0.7500 0.2500"   # 5
    "0.0000 0.5000 0.5000"   # 6
    "0.2500 0.7500 0.7500"   # 7
    "0.5000 0.0000 0.0000"   # 8
    "0.7500 0.2500 0.2500"   # 9
    "0.5000 0.0000 0.5000"   # 10
    "0.7500 0.2500 0.7500"   # 11
    "0.5000 0.5000 0.0000"   # 12
    "0.7500 0.7500 0.2500"   # 13
    "0.5000 0.5000 0.5000"   # 14
    "0.7500 0.7500 0.7500"   # 15
)

# Cr substitution order (maximizes Cr-Cr distance)
CR_ORDER=(0 14 6 8 3 13 4 10 7 9 2 12 5 11 1 15)

# ── Function: generate atomic positions block ─────────────────
generate_atoms() {
    local n_cr=$1
    local nat_fe=$((16 - n_cr))
    local nat_cr=$n_cr

    # Build array of which sites are Cr
    declare -A is_cr
    for ((i=0; i<n_cr; i++)); do
        is_cr[${CR_ORDER[$i]}]=1
    done

    echo "ATOMIC_POSITIONS crystal"
    for ((i=0; i<16; i++)); do
        if [[ ${is_cr[$i]+_} ]]; then
            echo "Cr  ${POSITIONS[$i]}"
        else
            echo "Fe  ${POSITIONS[$i]}"
        fi
    done
}

# ── Function: write QE input file ─────────────────────────────
write_input() {
    local prefix=$1
    local a1x=$2 a1y=$3 a1z=$4
    local a2x=$5 a2y=$6 a2z=$7
    local a3x=$8 a3y=$9 a3z=${10}
    local n_cr=${11}
    local calc=${12:-scf}
    local nat_fe=$((16 - n_cr))
    local nat_cr=$n_cr
    local ntyp=2
    if [ "$n_cr" -eq 0 ]; then ntyp=1; fi
    if [ "$n_cr" -eq 16 ]; then ntyp=1; fi

    # Determine starting magnetization
    # Fe: ferromagnetic (0.5), Cr: antiferromagnetic but we start ferro (0.3)
    local mag_block=""
    if [ "$ntyp" -eq 2 ]; then
        mag_block="    starting_magnetization(1) = 0.5,
    starting_magnetization(2) = 0.3,"
    elif [ "$n_cr" -eq 0 ]; then
        mag_block="    starting_magnetization(1) = 0.5,"
    else
        mag_block="    starting_magnetization(1) = 0.3,"
    fi

    # Atomic species block
    local species_block=""
    if [ "$n_cr" -eq 0 ]; then
        species_block="ATOMIC_SPECIES
Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"
    elif [ "$n_cr" -eq 16 ]; then
        species_block="ATOMIC_SPECIES
Cr 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"
    else
        species_block="ATOMIC_SPECIES
Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF
Cr 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"
    fi

    # Calc-specific blocks
    local control_extra=""
    local ions_block=""
    local cell_block=""
    if [ "$calc" = "vc-relax" ]; then
        control_extra="    forc_conv_thr = 1.0d-5,"
        ions_block="&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs',
    press = 0.0,
    press_conv_thr = 0.5
/"
    fi

    cat << EOF
&CONTROL
    calculation = '${calc}',
    prefix = '${prefix}',
    pseudo_dir = './pseudo/',
    outdir = './tmp/',
    tstress = .true.,
${control_extra}
/
&SYSTEM
    ibrav = 0,
    nat = 16,
    ntyp = ${ntyp},
    ecutwfc = 60.0,
    ecutrho = 480.0,
    occupations = 'smearing',
    smearing = 'mv',
    degauss = 0.02,
    nspin = 2,
${mag_block}
/
&ELECTRONS
    conv_thr = 1.0d-8,
    mixing_beta = 0.3
/
${ions_block}
${species_block}
CELL_PARAMETERS angstrom
${a1x}  ${a1y}  ${a1z}
${a2x}  ${a2y}  ${a2z}
${a3x}  ${a3y}  ${a3z}
$(generate_atoms $n_cr)
K_POINTS automatic
6 6 6 0 0 0
EOF
}

# ── Initialize summary file ──────────────────────────────────
echo "# n_Cr  x_Cr  a_eq(Ang)  E_total(Ry)  mag(muB)  P_hydro_slope  tetra_slope  shear_slope" > dft_data/elastic_grid_summary.dat

# ══════════════════════════════════════════════════════════════
# MAIN LOOP: over compositions
# ══════════════════════════════════════════════════════════════
for N_CR in $(seq 0 16); do

    X_CR=$(echo "scale=4; $N_CR / 16" | bc)
    N_FE=$((16 - N_CR))
    TAG=$(printf "fe%02dcr%02d" $N_FE $N_CR)

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  COMPOSITION: Fe${N_FE}Cr${N_CR} (x_Cr = ${X_CR})"
    echo "══════════════════════════════════════════════════════"

    # ── Step 1: vc-relax to find equilibrium lattice parameter ──
    # Vegard's law initial guess
    A_GUESS=$(echo "${A_FE_BOHR} * (1 - ${X_CR}) + ${A_CR_BOHR} * ${X_CR}" | bc -l)
    A_ANG=$(echo "${A_GUESS} * 0.529177" | bc -l | xargs printf "%.6f")
    SUPER_A=$(echo "${A_ANG} * 2" | bc -l | xargs printf "%.6f")

    echo "  [${TAG}] Initial guess: a = ${A_ANG} Å, supercell = ${SUPER_A} Å"

    FNAME="inputs/${TAG}_vcrelax.in"
    write_input "${TAG}_vcrelax" \
        "$SUPER_A" "0.000000" "0.000000" \
        "0.000000" "$SUPER_A" "0.000000" \
        "0.000000" "0.000000" "$SUPER_A" \
        "$N_CR" "vc-relax" > "$FNAME"

    echo "  [${TAG}] Running vc-relax..."
    $PW < "$FNAME" > "outputs/${TAG}_vcrelax.out" 2>&1 || true

    # Extract equilibrium supercell parameter
    # Look for final CELL_PARAMETERS
    FINAL_A=$(grep -A 1 "CELL_PARAMETERS" "outputs/${TAG}_vcrelax.out" | tail -1 | awk '{print $1}')
    if [ -z "$FINAL_A" ]; then
        echo "  [${TAG}] vc-relax may not have converged, using initial guess"
        FINAL_A=$SUPER_A
    fi
    A_EQ=$(echo "${FINAL_A} / 2" | bc -l | xargs printf "%.6f")
    echo "  [${TAG}] Equilibrium: a = ${A_EQ} Å (supercell = ${FINAL_A} Å)"

    # Extract energy and magnetization
    E_TOT=$(grep "!" "outputs/${TAG}_vcrelax.out" | tail -1 | awk '{print $5}')
    MAG=$(grep "total magnetization" "outputs/${TAG}_vcrelax.out" | tail -1 | awk '{print $4}')
    echo "  [${TAG}] E = ${E_TOT} Ry, mag = ${MAG} μB"

    # Use the equilibrium supercell parameter for elastic calculations
    SA=$FINAL_A

    # ── Step 2: Hydrostatic deformation ───────────────────────
    echo "  [${TAG}] Running hydrostatic deformations..."
    > "dft_data/${TAG}_hydro.dat"  # clear file

    for EPS in $STRAINS; do
        SA_DEF=$(echo "${SA} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
        PREFIX="${TAG}_hydro_${EPS}"
        FNAME="inputs/${TAG}_hydro_${EPS}.in"

        write_input "$PREFIX" \
            "$SA_DEF" "0.000000" "0.000000" \
            "0.000000" "$SA_DEF" "0.000000" \
            "0.000000" "0.000000" "$SA_DEF" \
            "$N_CR" > "$FNAME"

        $PW < "$FNAME" > "outputs/${TAG}_hydro_${EPS}.out" 2>&1 || true

        STRESS=$(grep "P=" "outputs/${TAG}_hydro_${EPS}.out" | tail -1 | awk -F'P=' '{print $2}' | awk '{print $1}')
        echo "${EPS} ${STRESS}" >> "dft_data/${TAG}_hydro.dat"
        echo "    hydro ε=${EPS}: P=${STRESS} kbar"
    done

    # ── Step 3: Tetragonal deformation ────────────────────────
    echo "  [${TAG}] Running tetragonal deformations..."
    > "dft_data/${TAG}_tetra.dat"

    for EPS in $STRAINS; do
        if [ "$EPS" = "0.00" ]; then
            SA_XY=$(printf "%.6f" $SA)
            SA_Z=$(printf "%.6f" $SA)
        else
            SA_Z=$(echo "${SA} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
            SA_XY=$(echo "${SA} / sqrt(1 + ${EPS})" | bc -l | xargs printf "%.6f")
        fi
        PREFIX="${TAG}_tetra_${EPS}"
        FNAME="inputs/${TAG}_tetra_${EPS}.in"

        write_input "$PREFIX" \
            "$SA_XY" "0.000000" "0.000000" \
            "0.000000" "$SA_XY" "0.000000" \
            "0.000000" "0.000000" "$SA_Z" \
            "$N_CR" > "$FNAME"

        $PW < "$FNAME" > "outputs/${TAG}_tetra_${EPS}.out" 2>&1 || true

        S11=$(grep -A 3 "total   stress" "outputs/${TAG}_tetra_${EPS}.out" | tail -3 | head -1 | awk '{print $4}')
        S33=$(grep -A 3 "total   stress" "outputs/${TAG}_tetra_${EPS}.out" | tail -1 | awk '{print $6}')
        SDIFF=$(echo "${S33} - ${S11}" | bc -l | xargs printf "%.4f")
        echo "${EPS} ${S11} ${S33} ${SDIFF}" >> "dft_data/${TAG}_tetra.dat"
        echo "    tetra ε=${EPS}: σ11=${S11}, σ33=${S33}, Δσ=${SDIFF}"
    done

    # ── Step 4: Shear deformation ─────────────────────────────
    echo "  [${TAG}] Running shear deformations..."
    > "dft_data/${TAG}_shear.dat"

    for EPS in $STRAINS; do
        SHEAR=$(echo "${SA} * ${EPS}" | bc -l | xargs printf "%.6f")
        PREFIX="${TAG}_shear_${EPS}"
        FNAME="inputs/${TAG}_shear_${EPS}.in"

        write_input "$PREFIX" \
            "$SA" "$SHEAR" "0.000000" \
            "$SHEAR" "$SA" "0.000000" \
            "0.000000" "0.000000" "$SA" \
            "$N_CR" > "$FNAME"

        $PW < "$FNAME" > "outputs/${TAG}_shear_${EPS}.out" 2>&1 || true

        S12=$(grep -A 3 "total   stress" "outputs/${TAG}_shear_${EPS}.out" | tail -3 | head -1 | awk '{print $5}')
        echo "${EPS} ${S12}" >> "dft_data/${TAG}_shear.dat"
        echo "    shear ε=${EPS}: σ12=${S12}"
    done

    # ── Log summary ───────────────────────────────────────────
    echo "${N_CR} ${X_CR} ${A_EQ} ${E_TOT} ${MAG}" >> dft_data/elastic_grid_summary.dat
    echo "  [${TAG}] ✓ Complete"

done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL COMPOSITIONS DONE"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Next: python scripts/fit_elastic_grid.py"
