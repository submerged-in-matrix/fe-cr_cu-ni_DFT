#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Elastic Constants of Fe-Cr BCC Alloys (16-atom supercell)
# Full composition grid: Fe16 → Cr16 in steps of 6.25%
# ══════════════════════════════════════════════════════════════
#
# Settings:
#   - 16-atom 2×2×2 BCC supercell
#   - ecutwfc = 45 Ry, ecutrho = 360 Ry
#   - K-grid: 4×4×4
#   - MPI: 4 cores
#   - Vegard's law for lattice parameters (validated: |a_Fe - a_Cr| < 0.3%)
#   - 3 strain points per deformation (-0.01, 0, +0.01)
#   - No vc-relax (saves ~60 hours)
#
# Equilibrium lattice parameters (from optimization):
#   Fe: a = 2.8234 Å (supercell = 5.6469 Å)
#   Cr: a = 2.8300 Å (supercell = 5.6600 Å)
#
# Usage:
#   cd ~/dft_projects/fe_cr-E_UTS-dft
#   nohup bash scripts/run_elastic_grid.sh > elastic_grid.log 2>&1 &
#
# Estimated time: ~3.5 days (153 SCF calculations × ~35 min each)

PW="mpirun -np 4 --oversubscribe /home/sayeed/q-e/bin/pw.x"
STRAINS="-0.01 0.00 0.01"

# Equilibrium supercell parameters (Angstrom)
SA_FE=5.6469
SA_CR=5.6600

mkdir -p inputs outputs dft_data

# ── 16 atomic positions in 2×2×2 BCC supercell (fractional) ──
POSITIONS=(
    "0.0000 0.0000 0.0000"
    "0.2500 0.2500 0.2500"
    "0.0000 0.0000 0.5000"
    "0.2500 0.2500 0.7500"
    "0.0000 0.5000 0.0000"
    "0.2500 0.7500 0.2500"
    "0.0000 0.5000 0.5000"
    "0.2500 0.7500 0.7500"
    "0.5000 0.0000 0.0000"
    "0.7500 0.2500 0.2500"
    "0.5000 0.0000 0.5000"
    "0.7500 0.2500 0.7500"
    "0.5000 0.5000 0.0000"
    "0.7500 0.7500 0.2500"
    "0.5000 0.5000 0.5000"
    "0.7500 0.7500 0.7500"
)

# Cr substitution order (maximizes Cr-Cr distance)
CR_ORDER=(0 14 6 8 3 13 4 10 7 9 2 12 5 11 1 15)

# ── Function: generate atomic positions block ─────────────────
generate_atoms() {
    local n_cr=$1
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

    local ntyp=2
    if [ "$n_cr" -eq 0 ]; then ntyp=1; fi
    if [ "$n_cr" -eq 16 ]; then ntyp=1; fi

    local mag_block=""
    if [ "$ntyp" -eq 2 ]; then
        mag_block="    starting_magnetization(1) = 0.5,
    starting_magnetization(2) = 0.3,"
    elif [ "$n_cr" -eq 0 ]; then
        mag_block="    starting_magnetization(1) = 0.5,"
    else
        mag_block="    starting_magnetization(1) = 0.3,"
    fi

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
    nat = 16,
    ntyp = ${ntyp},
    ecutwfc = 45.0,
    ecutrho = 360.0,
    occupations = 'smearing',
    smearing = 'mv',
    degauss = 0.02,
    nspin = 2,
${mag_block}
/
&ELECTRONS
    conv_thr = 1.0d-6,
    mixing_beta = 0.3
/
${species_block}
CELL_PARAMETERS angstrom
${a1x}  ${a1y}  ${a1z}
${a2x}  ${a2y}  ${a2z}
${a3x}  ${a3y}  ${a3z}
$(generate_atoms $n_cr)
K_POINTS automatic
4 4 4 0 0 0
EOF
}

# ── Initialize data files ─────────────────────────────────────
echo "# n_Cr  x_Cr  SA_eq(Ang)  E_total(Ry)  mag(muB)" > dft_data/elastic_grid_summary.dat

TOTAL_CALCS=$((17 * 9))
DONE=0
START_TIME=$(date +%s)

echo "══════════════════════════════════════════════════════"
echo "  Fe-Cr Elastic Constants Grid"
echo "  17 compositions × 9 SCFs = ${TOTAL_CALCS} calculations"
echo "  Started: $(date)"
echo "══════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════
for N_CR in $(seq 0 16); do

    X_CR=$(echo "scale=4; $N_CR / 16" | bc -l)
    N_FE=$((16 - N_CR))
    TAG=$(printf "fe%02dcr%02d" $N_FE $N_CR)

    # Vegard's law: SA(x) = SA_Fe*(1-x) + SA_Cr*x
    SA=$(echo "${SA_FE} * (1 - ${X_CR}) + ${SA_CR} * ${X_CR}" | bc -l | xargs printf "%.6f")

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  COMPOSITION: Fe${N_FE}Cr${N_CR} (x_Cr = ${X_CR}, SA = ${SA} Å)"
    echo "  Progress: $((N_CR * 9))/${TOTAL_CALCS} calculations done"
    echo "══════════════════════════════════════════════════════"

    # ── Hydrostatic deformation ───────────────────────────────
    echo "  [${TAG}] Hydrostatic..."
    > "dft_data/${TAG}_hydro.dat"

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
        DONE=$((DONE + 1))

        STRESS=$(grep "P=" "outputs/${TAG}_hydro_${EPS}.out" | tail -1 | awk -F'P=' '{print $2}' | awk '{print $1}')
        ENERGY=$(grep "!" "outputs/${TAG}_hydro_${EPS}.out" | tail -1 | awk '{print $5}')
        MAG=$(grep "total magnetization" "outputs/${TAG}_hydro_${EPS}.out" | tail -1 | awk '{print $4}')
        echo "${EPS} ${STRESS}" >> "dft_data/${TAG}_hydro.dat"

        NOW=$(date +%s)
        ELAPSED=$(( (NOW - START_TIME) / 60 ))
        RATE=$(echo "scale=1; ${ELAPSED} / ${DONE}" | bc -l)
        REMAINING=$(echo "scale=0; ${RATE} * (${TOTAL_CALCS} - ${DONE})" | bc -l)
        echo "    ε=${EPS}: P=${STRESS} kbar [${DONE}/${TOTAL_CALCS}, ~${REMAINING} min remaining]"
    done

    # ── Tetragonal deformation ────────────────────────────────
    echo "  [${TAG}] Tetragonal..."
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
        DONE=$((DONE + 1))

        S11=$(grep -A 3 "total   stress" "outputs/${TAG}_tetra_${EPS}.out" | tail -3 | head -1 | awk '{print $4}')
        S33=$(grep -A 3 "total   stress" "outputs/${TAG}_tetra_${EPS}.out" | tail -1 | awk '{print $6}')
        SDIFF=$(echo "${S33} - ${S11}" | bc -l | xargs printf "%.4f")
        echo "${EPS} ${S11} ${S33} ${SDIFF}" >> "dft_data/${TAG}_tetra.dat"

        NOW=$(date +%s)
        ELAPSED=$(( (NOW - START_TIME) / 60 ))
        RATE=$(echo "scale=1; ${ELAPSED} / ${DONE}" | bc -l)
        REMAINING=$(echo "scale=0; ${RATE} * (${TOTAL_CALCS} - ${DONE})" | bc -l)
        echo "    ε=${EPS}: Δσ=${SDIFF} kbar [${DONE}/${TOTAL_CALCS}, ~${REMAINING} min remaining]"
    done

    # ── Shear deformation ─────────────────────────────────────
    echo "  [${TAG}] Shear..."
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
        DONE=$((DONE + 1))

        S12=$(grep -A 3 "total   stress" "outputs/${TAG}_shear_${EPS}.out" | tail -3 | head -1 | awk '{print $5}')
        echo "${EPS} ${S12}" >> "dft_data/${TAG}_shear.dat"

        NOW=$(date +%s)
        ELAPSED=$(( (NOW - START_TIME) / 60 ))
        RATE=$(echo "scale=1; ${ELAPSED} / ${DONE}" | bc -l)
        REMAINING=$(echo "scale=0; ${RATE} * (${TOTAL_CALCS} - ${DONE})" | bc -l)
        echo "    ε=${EPS}: σ12=${S12} kbar [${DONE}/${TOTAL_CALCS}, ~${REMAINING} min remaining]"
    done

    # ── Log summary for this composition ──────────────────────
    # Use the unstrained (ε=0) hydrostatic run for energy and magnetization
    E_TOT=$(grep "!" "outputs/${TAG}_hydro_0.00.out" | tail -1 | awk '{print $5}')
    MAG=$(grep "total magnetization" "outputs/${TAG}_hydro_0.00.out" | tail -1 | awk '{print $4}')
    echo "${N_CR} ${X_CR} ${SA} ${E_TOT} ${MAG}" >> dft_data/elastic_grid_summary.dat
    echo "  [${TAG}] ✓ Complete (E=${E_TOT} Ry, μ=${MAG} μB)"

done

END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL 17 COMPOSITIONS DONE"
echo "  Total time: ${TOTAL_MIN} minutes"
echo "  Finished: $(date)"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Data files in dft_data/:"
ls dft_data/*_hydro.dat dft_data/*_tetra.dat dft_data/*_shear.dat 2>/dev/null | wc -l
echo " .dat files created"
echo ""
echo "Next: python scripts/fit_elastic_grid.py"