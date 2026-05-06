#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Elastic Constants of BCC Fe-Cr & FCC Cu-Ni Alloys
# ══════════════════════════════════════════════════════════════
#
# Fe-Cr: BCC, 16-atom supercell, 17 compositions, PAW PPs
# Cu-Ni: FCC, 32-atom supercell, 17 compositions, ultrasoft PPs
#
# Total: 34 compositions × 10 QE runs = 340 calculations
#
# Usage:
#   nohup bash scripts/run_elastic_grid.sh > elastic_grid.log 2>&1 &

# ── Auto-detect QE ────────────────────────────────────────────
PW_BIN=""
for loc in "$HOME/q-e/bin/pw.x" "/usr/bin/pw.x" "/usr/local/bin/pw.x" \
           "$HOME/qe/bin/pw.x" "$HOME/espresso/bin/pw.x"; do
    if [ -f "$loc" ]; then PW_BIN="$loc"; break; fi
done
if [ -z "$PW_BIN" ]; then echo "ERROR: pw.x not found."; exit 1; fi

# Auto-detect execution mode — use whatever is available
if $PW_BIN --version 2>&1 | grep -qi "gpu"; then
    PW="$PW_BIN"
    RUN_MODE="GPU"
elif command -v mpirun &> /dev/null; then
    NCORES=$(nproc)
    PW="mpirun -np $NCORES $PW_BIN"
    RUN_MODE="MPI ($NCORES cores)"
else
    PW="$PW_BIN"
    RUN_MODE="single core"
fi

STRAINS="-0.01 0.00 0.01"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p inputs outputs dft_data tmp

START_TIME=$(date +%s)
TOTAL_CALCS=340
DONE=0

echo "══════════════════════════════════════════════════════"
echo "  Elastic Constants: Fe-Cr (BCC) + Cu-Ni (FCC)"
echo "  Mode: $RUN_MODE"
echo "  QE: $PW_BIN"
echo "  Total: $TOTAL_CALCS calculations"
echo "  Started: $(date)"
echo "══════════════════════════════════════════════════════"

# ── Supercell positions ───────────────────────────────────────

BCC_POS=(
    "0.0000 0.0000 0.0000" "0.2500 0.2500 0.2500"
    "0.0000 0.0000 0.5000" "0.2500 0.2500 0.7500"
    "0.0000 0.5000 0.0000" "0.2500 0.7500 0.2500"
    "0.0000 0.5000 0.5000" "0.2500 0.7500 0.7500"
    "0.5000 0.0000 0.0000" "0.7500 0.2500 0.2500"
    "0.5000 0.0000 0.5000" "0.7500 0.2500 0.7500"
    "0.5000 0.5000 0.0000" "0.7500 0.7500 0.2500"
    "0.5000 0.5000 0.5000" "0.7500 0.7500 0.7500"
)
SUB_BCC=(0 14 6 8 3 13 4 10 7 9 2 12 5 11 1 15)

FCC_POS=(
    "0.0000 0.0000 0.0000" "0.2500 0.2500 0.0000"
    "0.2500 0.0000 0.2500" "0.0000 0.2500 0.2500"
    "0.0000 0.0000 0.5000" "0.2500 0.2500 0.5000"
    "0.2500 0.0000 0.7500" "0.0000 0.2500 0.7500"
    "0.0000 0.5000 0.0000" "0.2500 0.7500 0.0000"
    "0.2500 0.5000 0.2500" "0.0000 0.7500 0.2500"
    "0.0000 0.5000 0.5000" "0.2500 0.7500 0.5000"
    "0.2500 0.5000 0.7500" "0.0000 0.7500 0.7500"
    "0.5000 0.0000 0.0000" "0.7500 0.2500 0.0000"
    "0.7500 0.0000 0.2500" "0.5000 0.2500 0.2500"
    "0.5000 0.0000 0.5000" "0.7500 0.2500 0.5000"
    "0.7500 0.0000 0.7500" "0.5000 0.2500 0.7500"
    "0.5000 0.5000 0.0000" "0.7500 0.7500 0.0000"
    "0.7500 0.5000 0.2500" "0.5000 0.7500 0.2500"
    "0.5000 0.5000 0.5000" "0.7500 0.7500 0.5000"
    "0.7500 0.5000 0.7500" "0.5000 0.7500 0.7500"
)
SUB_FCC=(0 29 6 11 17 14 21 25 1 2 3 4 5 7 8 9 10 12 13 15 16 18 19 20 22 23 24 26 27 28 30 31)

# ══════════════════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════════════════

gen_atoms() {
    local n_sol=$1 e_host=$2 e_sol=$3 nat=$4
    shift 4; local pos=("$@")
    if [ "$nat" -eq 16 ]; then local ord=("${SUB_BCC[@]}");
    else local ord=("${SUB_FCC[@]}"); fi
    declare -A is_sol
    for ((i=0; i<n_sol; i++)); do is_sol[${ord[$i]}]=1; done
    echo "ATOMIC_POSITIONS crystal"
    for ((i=0; i<nat; i++)); do
        if [[ ${is_sol[$i]+_} ]]; then echo "${e_sol}  ${pos[$i]}"
        else echo "${e_host}  ${pos[$i]}"; fi
    done
}

write_qe() {
    local pref=$1 calc=$2
    local a1x=$3 a1y=$4 a1z=$5 a2x=$6 a2y=$7 a2z=$8 a3x=$9 a3y=${10} a3z=${11}
    local nat=${12} nt=${13} ecut=${14} erho=${15} kg=${16}
    local sp="${17}" mg="${18}" ab="${19}"
    local ce="" ib=""
    if [ "$calc" = "vc-relax" ]; then
        ce="    forc_conv_thr = 1.0d-4,"
        ib="&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs',
    press = 0.0,
    press_conv_thr = 0.5
/"
    fi
    cat << QEOF
&CONTROL
    calculation = '${calc}',
    prefix = '${pref}',
    pseudo_dir = './pseudo/',
    outdir = './tmp/',
    tstress = .true.,
${ce}
/
&SYSTEM
    ibrav = 0,
    nat = ${nat},
    ntyp = ${nt},
    ecutwfc = ${ecut}.0,
    ecutrho = ${erho}.0,
    occupations = 'smearing',
    smearing = 'mv',
    degauss = 0.02,
${mg}
/
&ELECTRONS
    conv_thr = 1.0d-8,
    mixing_beta = 0.3
/
${ib}
${sp}
CELL_PARAMETERS angstrom
${a1x}  ${a1y}  ${a1z}
${a2x}  ${a2y}  ${a2z}
${a3x}  ${a3y}  ${a3z}
${ab}
K_POINTS automatic
${kg} ${kg} ${kg} 0 0 0
QEOF
}

progress() {
    DONE=$((DONE + 1))
    local NOW=$(date +%s)
    local EL=$(( (NOW - START_TIME) / 60 ))
    if [ "$DONE" -gt 0 ] && [ "$EL" -gt 0 ]; then
        local RATE=$(echo "scale=1; ${EL} / ${DONE}" | bc -l 2>/dev/null || echo "?")
        local REM=$(echo "scale=0; ${RATE} * (${TOTAL_CALCS} - ${DONE})" | bc -l 2>/dev/null || echo "?")
        echo "    [${DONE}/${TOTAL_CALCS}] ~${REM} min remaining"
    else
        echo "    [${DONE}/${TOTAL_CALCS}]"
    fi
}

run_elastic() {
    local TAG=$1 SA=$2 nat=$3 nt=$4 ec=$5 er=$6 kg=$7
    local sp="$8" mg="$9" ab="${10}"

    echo "  [${TAG}] Hydrostatic..."
    > "dft_data/${TAG}_hydro.dat"
    for EPS in $STRAINS; do
        SD=$(echo "${SA} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
        write_qe "${TAG}_h${EPS}" "scf" \
            "$SD" "0.000000" "0.000000" "0.000000" "$SD" "0.000000" "0.000000" "0.000000" "$SD" \
            "$nat" "$nt" "$ec" "$er" "$kg" "$sp" "$mg" "$ab" > "inputs/${TAG}_h${EPS}.in"
        $PW < "inputs/${TAG}_h${EPS}.in" > "outputs/${TAG}_h${EPS}.out" 2>&1 || true
        P=$(grep "P=" "outputs/${TAG}_h${EPS}.out" | tail -1 | awk -F'P=' '{print $2}' | awk '{print $1}')
        echo "${EPS} ${P}" >> "dft_data/${TAG}_hydro.dat"
        echo "    hydro ε=${EPS}: P=${P} kbar"; progress
    done

    echo "  [${TAG}] Tetragonal..."
    > "dft_data/${TAG}_tetra.dat"
    for EPS in $STRAINS; do
        if [ "$EPS" = "0.00" ]; then SXY=$SA; SZ=$SA
        else
            SZ=$(echo "${SA} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
            SXY=$(echo "${SA} / sqrt(1 + ${EPS})" | bc -l | xargs printf "%.6f")
        fi
        write_qe "${TAG}_t${EPS}" "scf" \
            "$SXY" "0.000000" "0.000000" "0.000000" "$SXY" "0.000000" "0.000000" "0.000000" "$SZ" \
            "$nat" "$nt" "$ec" "$er" "$kg" "$sp" "$mg" "$ab" > "inputs/${TAG}_t${EPS}.in"
        $PW < "inputs/${TAG}_t${EPS}.in" > "outputs/${TAG}_t${EPS}.out" 2>&1 || true
        S11=$(grep -A 3 "total   stress" "outputs/${TAG}_t${EPS}.out" | tail -3 | head -1 | awk '{print $4}')
        S33=$(grep -A 3 "total   stress" "outputs/${TAG}_t${EPS}.out" | tail -1 | awk '{print $6}')
        DS=$(echo "${S33} - ${S11}" | bc -l | xargs printf "%.4f")
        echo "${EPS} ${S11} ${S33} ${DS}" >> "dft_data/${TAG}_tetra.dat"
        echo "    tetra ε=${EPS}: Δσ=${DS} kbar"; progress
    done

    echo "  [${TAG}] Shear..."
    > "dft_data/${TAG}_shear.dat"
    for EPS in $STRAINS; do
        SH=$(echo "${SA} * ${EPS}" | bc -l | xargs printf "%.6f")
        write_qe "${TAG}_s${EPS}" "scf" \
            "$SA" "$SH" "0.000000" "$SH" "$SA" "0.000000" "0.000000" "0.000000" "$SA" \
            "$nat" "$nt" "$ec" "$er" "$kg" "$sp" "$mg" "$ab" > "inputs/${TAG}_s${EPS}.in"
        $PW < "inputs/${TAG}_s${EPS}.in" > "outputs/${TAG}_s${EPS}.out" 2>&1 || true
        S12=$(grep -A 3 "total   stress" "outputs/${TAG}_s${EPS}.out" | tail -3 | head -1 | awk '{print $5}')
        echo "${EPS} ${S12}" >> "dft_data/${TAG}_shear.dat"
        echo "    shear ε=${EPS}: σ12=${S12} kbar"; progress
    done
}

# ══════════════════════════════════════════════════════════════
# SYSTEM 1: Fe-Cr (BCC, 16 atoms, PAW)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  SYSTEM 1: Fe-Cr (BCC, 16 atoms) — 17 compositions ║"
echo "╚══════════════════════════════════════════════════════╝"

SA_FE=5.6469; SA_CR=5.6600
echo "# system n_sol x_sol SA_eq E_tot mag" > dft_data/fecr_summary.dat

for N_CR in $(seq 0 16); do
    X=$(echo "scale=4; ${N_CR} / 16" | bc -l)
    NF=$((16 - N_CR)); TAG=$(printf "fecr_fe%02dcr%02d" $NF $N_CR)
    SA_G=$(echo "${SA_FE} * (1.0 - ${X}) + ${SA_CR} * ${X}" | bc -l | xargs printf "%.6f")

    echo ""; echo "── Fe${NF}Cr${N_CR} (x=${X}) ──────────────────"

    NT=2; [ "$N_CR" -eq 0 ] && NT=1; [ "$N_CR" -eq 16 ] && NT=1

    if [ "$N_CR" -eq 0 ]; then
        SP="ATOMIC_SPECIES
Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"
    elif [ "$N_CR" -eq 16 ]; then
        SP="ATOMIC_SPECIES
Cr 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"
    else
        SP="ATOMIC_SPECIES
Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF
Cr 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"
    fi

    if [ "$NT" -eq 2 ]; then
        MG="    nspin = 2,
    starting_magnetization(1) = 0.5,
    starting_magnetization(2) = 0.3,"
    elif [ "$N_CR" -eq 0 ]; then
        MG="    nspin = 2,
    starting_magnetization(1) = 0.5,"
    else
        MG="    nspin = 2,
    starting_magnetization(1) = 0.3,"
    fi

    AB=$(gen_atoms $N_CR "Fe" "Cr" 16 "${BCC_POS[@]}")

    echo "  [${TAG}] vc-relax..."
    write_qe "${TAG}_vcr" "vc-relax" \
        "$SA_G" "0.000000" "0.000000" "0.000000" "$SA_G" "0.000000" "0.000000" "0.000000" "$SA_G" \
        16 "$NT" 60 480 6 "$SP" "$MG" "$AB" > "inputs/${TAG}_vcr.in"
    $PW < "inputs/${TAG}_vcr.in" > "outputs/${TAG}_vcr.out" 2>&1 || true
    progress

    SA_EQ=$(grep -A 1 "CELL_PARAMETERS" "outputs/${TAG}_vcr.out" | tail -1 | awk '{print $1}')
    [ -z "$SA_EQ" ] || [ "$SA_EQ" = "CELL_PARAMETERS" ] && SA_EQ=$SA_G
    ET=$(grep "!" "outputs/${TAG}_vcr.out" | tail -1 | awk '{print $5}')
    MV=$(grep "total magnetization" "outputs/${TAG}_vcr.out" | tail -1 | awk '{print $4}')
    echo "  [${TAG}] SA=${SA_EQ} Å, E=${ET} Ry, μ=${MV}"

    run_elastic "$TAG" "$SA_EQ" 16 "$NT" 60 480 6 "$SP" "$MG" "$AB"
    echo "FeCr ${N_CR} ${X} ${SA_EQ} ${ET} ${MV}" >> dft_data/fecr_summary.dat
    echo "  [${TAG}] ✓ Complete"
done

# ══════════════════════════════════════════════════════════════
# SYSTEM 2: Cu-Ni (FCC, 32 atoms, ultrasoft)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  SYSTEM 2: Cu-Ni (FCC, 32 atoms) — 17 compositions ║"
echo "╚══════════════════════════════════════════════════════╝"

SA_CU=7.2300; SA_NI=7.0480
echo "# system n_sol x_sol SA_eq E_tot" > dft_data/cuni_summary.dat

for N_NI in $(seq 0 2 32); do
    X=$(echo "scale=4; ${N_NI} / 32" | bc -l)
    NC=$((32 - N_NI)); TAG=$(printf "cuni_cu%02dni%02d" $NC $N_NI)
    SA_G=$(echo "${SA_CU} * (1.0 - ${X}) + ${SA_NI} * ${X}" | bc -l | xargs printf "%.6f")

    echo ""; echo "── Cu${NC}Ni${N_NI} (x=${X}) ──────────────────"

    NT=2; [ "$N_NI" -eq 0 ] && NT=1; [ "$N_NI" -eq 32 ] && NT=1

    if [ "$N_NI" -eq 0 ]; then
        SP="ATOMIC_SPECIES
Cu 63.546 Cu.pbe-dn-rrkjus_psl.1.0.0.UPF"
    elif [ "$N_NI" -eq 32 ]; then
        SP="ATOMIC_SPECIES
Ni 58.693 Ni.pbe-nd-rrkjus.UPF"
    else
        SP="ATOMIC_SPECIES
Cu 63.546 Cu.pbe-dn-rrkjus_psl.1.0.0.UPF
Ni 58.693 Ni.pbe-nd-rrkjus.UPF"
    fi

    MG=""  # Cu-Ni: non-magnetic

    AB=$(gen_atoms $N_NI "Cu" "Ni" 32 "${FCC_POS[@]}")

    echo "  [${TAG}] vc-relax..."
    write_qe "${TAG}_vcr" "vc-relax" \
        "$SA_G" "0.000000" "0.000000" "0.000000" "$SA_G" "0.000000" "0.000000" "0.000000" "$SA_G" \
        32 "$NT" 60 480 4 "$SP" "$MG" "$AB" > "inputs/${TAG}_vcr.in"
    $PW < "inputs/${TAG}_vcr.in" > "outputs/${TAG}_vcr.out" 2>&1 || true
    progress

    SA_EQ=$(grep -A 1 "CELL_PARAMETERS" "outputs/${TAG}_vcr.out" | tail -1 | awk '{print $1}')
    [ -z "$SA_EQ" ] || [ "$SA_EQ" = "CELL_PARAMETERS" ] && SA_EQ=$SA_G
    ET=$(grep "!" "outputs/${TAG}_vcr.out" | tail -1 | awk '{print $5}')
    echo "  [${TAG}] SA=${SA_EQ} Å, E=${ET} Ry"

    run_elastic "$TAG" "$SA_EQ" 32 "$NT" 60 480 4 "$SP" "$MG" "$AB"
    echo "CuNi ${N_NI} ${X} ${SA_EQ} ${ET}" >> dft_data/cuni_summary.dat
    echo "  [${TAG}] ✓ Complete"
done

# ══════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL DONE — ${TOTAL_MIN} min ($(echo "scale=1; ${TOTAL_MIN}/60" | bc -l) hrs)"
echo "  Finished: $(date)"
echo "══════════════════════════════════════════════════════"
echo "  Fe-Cr: $(ls dft_data/fecr_*_hydro.dat 2>/dev/null | wc -l) compositions"
echo "  Cu-Ni: $(ls dft_data/cuni_*_hydro.dat 2>/dev/null | wc -l) compositions"
echo "  Next: python scripts/fit_elastic_grid.py"
