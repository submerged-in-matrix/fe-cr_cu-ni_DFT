#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Elastic Constants of BCC Fe-Cr & FCC Cu-Ni Alloys
# Checkpointable: completed composition tags are skipped on re-run.
# ══════════════════════════════════════════════════════════════
#
# Usage:
#   cd /path/to/project_root  (where pseudo/ and scripts/ live)
#   nohup bash scripts/run_elastic_grid.sh > elastic_grid.log 2>&1 &
#
# Resume after interruption: just re-run the same command.
# Already-completed tags are detected via checkpoint files in
# checkpoints/<TAG>.done and skipped entirely.
#
# To force re-run a specific tag, delete its checkpoint:
#   rm checkpoints/fecr_fe14cr02.done

# ── Find pw.x ─────────────────────────────────────────────────
PW=""
for loc in "/opt/q-e/bin/pw.x" "$HOME/q-e/bin/pw.x" "/usr/bin/pw.x" "/usr/local/bin/pw.x"; do
    if [ -f "$loc" ]; then PW="$loc"; break; fi
done
[ -z "$PW" ] && PW=$(which pw.x 2>/dev/null)
[ -z "$PW" ] && { echo "ERROR: pw.x not found"; exit 1; }

# ── Find project root ─────────────────────────────────────────
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

if [ ! -d "$PROJ_DIR/pseudo" ]; then
    echo "ERROR: pseudo/ not found in $PROJ_DIR"
    exit 1
fi

# ── Detect available cores (used by mpirun) ───────────────────
NCORES=$(nproc 2>/dev/null || echo 1)

STRAINS="-0.01 0.00 0.01"
mkdir -p inputs outputs dft_data tmp checkpoints

START_TIME=$(date +%s)
TOTAL_CALCS=340
DONE=0

echo "══════════════════════════════════════════════════════"
echo "  Elastic Constants: Fe-Cr (BCC) + Cu-Ni (FCC)"
echo "  QE: $PW"
echo "  Project: $PROJ_DIR"
echo "  Cores: $NCORES"
echo "  Total: $TOTAL_CALCS calculations"
echo "  Started: $(date)"
echo "══════════════════════════════════════════════════════"

# ── Checkpoint helpers ────────────────────────────────────────
# Mark a composition tag as fully complete.
mark_done() { touch "checkpoints/${1}.done"; }

# Return 0 (true) if tag is already complete.
is_done() { [ -f "checkpoints/${1}.done" ]; }

# ── Positions ─────────────────────────────────────────────────
BCC_X=(0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500)
BCC_Y=(0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500)
BCC_Z=(0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500)
SUB_BCC=(0 14 6 8 3 13 4 10 7 9 2 12 5 11 1 15)

FCC_X=(0.0000 0.2500 0.2500 0.0000 0.0000 0.2500 0.2500 0.0000 0.0000 0.2500 0.2500 0.0000 0.0000 0.2500 0.2500 0.0000 0.5000 0.7500 0.7500 0.5000 0.5000 0.7500 0.7500 0.5000 0.5000 0.7500 0.7500 0.5000 0.5000 0.7500 0.7500 0.5000)
FCC_Y=(0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500)
FCC_Z=(0.0000 0.0000 0.2500 0.2500 0.5000 0.5000 0.7500 0.7500 0.0000 0.0000 0.2500 0.2500 0.5000 0.5000 0.7500 0.7500 0.0000 0.0000 0.2500 0.2500 0.5000 0.5000 0.7500 0.7500 0.0000 0.0000 0.2500 0.2500 0.5000 0.5000 0.7500 0.7500)
SUB_FCC=(0 29 6 11 17 14 21 25 1 2 3 4 5 7 8 9 10 12 13 15 16 18 19 20 22 23 24 26 27 28 30 31)

# ── Write atoms to file ───────────────────────────────────────
append_atoms_bcc() {
    local f=$1 ns=$2 eh=$3 es=$4
    declare -A is_sol
    for ((i=0; i<ns; i++)); do is_sol[${SUB_BCC[$i]}]=1; done
    echo "ATOMIC_POSITIONS crystal" >> "$f"
    for ((i=0; i<16; i++)); do
        local e=$eh; [[ ${is_sol[$i]+_} ]] && e=$es
        echo "${e}  ${BCC_X[$i]} ${BCC_Y[$i]} ${BCC_Z[$i]}" >> "$f"
    done
}

append_atoms_fcc() {
    local f=$1 ns=$2 eh=$3 es=$4
    declare -A is_sol
    for ((i=0; i<ns; i++)); do is_sol[${SUB_FCC[$i]}]=1; done
    echo "ATOMIC_POSITIONS crystal" >> "$f"
    for ((i=0; i<32; i++)); do
        local e=$eh; [[ ${is_sol[$i]+_} ]] && e=$es
        echo "${e}  ${FCC_X[$i]} ${FCC_Y[$i]} ${FCC_Z[$i]}" >> "$f"
    done
}

# ── Write Fe-Cr input ─────────────────────────────────────────
make_fecr() {
    local f=$1 pref=$2 calc=$3
    local c1x=$4 c1y=$5 c1z=$6
    local c2x=$7 c2y=$8 c2z=$9
    shift 9
    local c3x=$1 c3y=$2 c3z=$3 ncr=$4

    local nt=2
    [ "$ncr" -eq 0 ] && nt=1
    [ "$ncr" -eq 16 ] && nt=1

    cat > "$f" << EOF
&CONTROL
    calculation = '${calc}',
    prefix = '${pref}',
    pseudo_dir = './pseudo/',
    outdir = './tmp/',
    tstress = .true.,
EOF
    [ "$calc" = "vc-relax" ] && echo "    forc_conv_thr = 1.0d-4," >> "$f"
    echo "/" >> "$f"
    cat >> "$f" << EOF
&SYSTEM
    ibrav = 0,
    nat = 16,
    ntyp = ${nt},
    ecutwfc = 60.0,
    ecutrho = 480.0,
    occupations = 'smearing',
    smearing = 'mv',
    degauss = 0.02,
    nspin = 2,
EOF
    if [ "$nt" -eq 2 ]; then
        echo "    starting_magnetization(1) = 0.5," >> "$f"
        echo "    starting_magnetization(2) = 0.3," >> "$f"
    elif [ "$ncr" -eq 0 ]; then
        echo "    starting_magnetization(1) = 0.5," >> "$f"
    else
        echo "    starting_magnetization(1) = 0.3," >> "$f"
    fi
    echo "/" >> "$f"
    cat >> "$f" << EOF
&ELECTRONS
    conv_thr = 1.0d-8,
    mixing_beta = 0.3
/
EOF
    if [ "$calc" = "vc-relax" ]; then
        cat >> "$f" << EOF
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs',
    press = 0.0,
    press_conv_thr = 0.5
/
EOF
    fi
    if [ "$ncr" -eq 0 ]; then
        echo "ATOMIC_SPECIES" >> "$f"
        echo "Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
    elif [ "$ncr" -eq 16 ]; then
        echo "ATOMIC_SPECIES" >> "$f"
        echo "Cr 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
    else
        echo "ATOMIC_SPECIES" >> "$f"
        echo "Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
        echo "Cr 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
    fi
    cat >> "$f" << EOF
CELL_PARAMETERS angstrom
${c1x}  ${c1y}  ${c1z}
${c2x}  ${c2y}  ${c2z}
${c3x}  ${c3y}  ${c3z}
EOF
    append_atoms_bcc "$f" "$ncr" "Fe" "Cr"
    echo "K_POINTS automatic" >> "$f"
    echo "6 6 6 0 0 0" >> "$f"
}

# ── Write Cu-Ni input ─────────────────────────────────────────
make_cuni() {
    local f=$1 pref=$2 calc=$3
    local c1x=$4 c1y=$5 c1z=$6
    local c2x=$7 c2y=$8 c2z=$9
    shift 9
    local c3x=$1 c3y=$2 c3z=$3 nni=$4

    local nt=2
    [ "$nni" -eq 0 ] && nt=1
    [ "$nni" -eq 32 ] && nt=1

    cat > "$f" << EOF
&CONTROL
    calculation = '${calc}',
    prefix = '${pref}',
    pseudo_dir = './pseudo/',
    outdir = './tmp/',
    tstress = .true.,
EOF
    [ "$calc" = "vc-relax" ] && echo "    forc_conv_thr = 1.0d-4," >> "$f"
    echo "/" >> "$f"
    cat >> "$f" << EOF
&SYSTEM
    ibrav = 0,
    nat = 32,
    ntyp = ${nt},
    ecutwfc = 60.0,
    ecutrho = 480.0,
    occupations = 'smearing',
    smearing = 'mv',
    degauss = 0.02,
/
&ELECTRONS
    conv_thr = 1.0d-8,
    mixing_beta = 0.3
/
EOF
    if [ "$calc" = "vc-relax" ]; then
        cat >> "$f" << EOF
&IONS
    ion_dynamics = 'bfgs'
/
&CELL
    cell_dynamics = 'bfgs',
    press = 0.0,
    press_conv_thr = 0.5
/
EOF
    fi
    if [ "$nni" -eq 0 ]; then
        echo "ATOMIC_SPECIES" >> "$f"
        echo "Cu 63.546 Cu.pbe-dn-rrkjus_psl.1.0.0.UPF" >> "$f"
    elif [ "$nni" -eq 32 ]; then
        echo "ATOMIC_SPECIES" >> "$f"
        echo "Ni 58.693 Ni.pbe-nd-rrkjus.UPF" >> "$f"
    else
        echo "ATOMIC_SPECIES" >> "$f"
        echo "Cu 63.546 Cu.pbe-dn-rrkjus_psl.1.0.0.UPF" >> "$f"
        echo "Ni 58.693 Ni.pbe-nd-rrkjus.UPF" >> "$f"
    fi
    cat >> "$f" << EOF
CELL_PARAMETERS angstrom
${c1x}  ${c1y}  ${c1z}
${c2x}  ${c2y}  ${c2z}
${c3x}  ${c3y}  ${c3z}
EOF
    append_atoms_fcc "$f" "$nni" "Cu" "Ni"
    echo "K_POINTS automatic" >> "$f"
    echo "4 4 4 0 0 0" >> "$f"
}

# ── Helpers ───────────────────────────────────────────────────
run_pw() {
    # GPU QE: 1 MPI rank per GPU — multiple ranks split work and compete for
    # the same device, which is slower on a single-GPU node.
    # OMP_NUM_THREADS hands the CPU cores to QE's threaded CPU-side routines.
    # -npool 1: k-mesh is small (6x6x6 / 4x4x4), k-point parallelism adds overhead.
    export OMP_NUM_THREADS="$NCORES"
    mpirun -np 1 "$PW" -npool 1 < "$1" > "$2" 2>&1
    if ! grep -q "JOB DONE\|convergence has been achieved\|bfgs converged" "$2" 2>/dev/null; then
        echo "    WARNING: pw.x may have failed for $1"
        return 1
    fi
    return 0
}

progress() {
    DONE=$((DONE + 1))
    local NOW=$(date +%s)
    local EL=$(( (NOW - START_TIME) / 60 ))
    if [ "$DONE" -gt 1 ] && [ "$EL" -gt 0 ]; then
        local REM=$(( EL * (TOTAL_CALCS - DONE) / DONE ))
        echo "    [${DONE}/${TOTAL_CALCS}] ~${REM} min remaining"
    else
        echo "    [${DONE}/${TOTAL_CALCS}]"
    fi
}

get_P()   { grep "P=" "$1" 2>/dev/null | tail -1 | awk -F'P=' '{print $2}' | awk '{print $1}'; }
get_s11() { grep -A 3 "total   stress" "$1" 2>/dev/null | tail -3 | head -1 | awk '{print $4}'; }
get_s33() { grep -A 3 "total   stress" "$1" 2>/dev/null | tail -1 | awk '{print $6}'; }
get_s12() { grep -A 3 "total   stress" "$1" 2>/dev/null | tail -3 | head -1 | awk '{print $5}'; }

safe_diff() {
    [ -n "$1" ] && [ -n "$2" ] && echo "$1 - $2" | bc -l 2>/dev/null | xargs printf "%.4f" 2>/dev/null || echo "0.0000"
}

# ══════════════════════════════════════════════════════════════
# SYSTEM 1: Fe-Cr (BCC, 16 atoms)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  SYSTEM 1: Fe-Cr (BCC, 16 atoms) — 17 compositions ║"
echo "╚══════════════════════════════════════════════════════╝"

SA_FE=5.6469; SA_CR=5.6600
# initialising summary file only if it does not exist (preserves prior runs)
[ -f "dft_data/fecr_summary.dat" ] || echo "# system n_sol x_sol SA_eq E_tot mag" > dft_data/fecr_summary.dat

for N_CR in $(seq 0 16); do
    X=$(echo "scale=4; ${N_CR} / 16" | bc -l)
    NF=$((16 - N_CR))
    TAG=$(printf "fecr_fe%02dcr%02d" $NF $N_CR)
    SA=$(echo "${SA_FE} * (1.0 - ${X}) + ${SA_CR} * ${X}" | bc -l | xargs printf "%.6f")

    # ── Checkpoint check ──────────────────────────────────────
    if is_done "$TAG"; then
        echo "── ${TAG}: already complete, skipping ──"
        # still advancing the progress counter for accurate ETA
        DONE=$((DONE + 10))
        continue
    fi

    echo ""; echo "── Fe${NF}Cr${N_CR} (x=${X}) ──────────────────"

    # ── vc-relax ──────────────────────────────────────────────
    echo "  [${TAG}] vc-relax..."
    make_fecr "inputs/${TAG}_vcr.in" "${TAG}_vcr" "vc-relax" \
        "$SA" "0.000000" "0.000000" \
        "0.000000" "$SA" "0.000000" \
        "0.000000" "0.000000" "$SA" "$N_CR"
    run_pw "inputs/${TAG}_vcr.in" "outputs/${TAG}_vcr.out"; progress

    SA_EQ=$(grep -A 1 "CELL_PARAMETERS" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $1}')
    [ -z "$SA_EQ" ] || [ "$SA_EQ" = "CELL_PARAMETERS" ] && SA_EQ=$SA
    ET=$(grep "!" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $5}')
    MV=$(grep "total magnetization" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $4}')
    echo "  [${TAG}] SA=${SA_EQ}, E=${ET}, μ=${MV}"

    # ── Hydrostatic ───────────────────────────────────────────
    echo "  [${TAG}] Hydrostatic..."
    > "dft_data/${TAG}_hydro.dat"
    for EPS in $STRAINS; do
        SD=$(echo "${SA_EQ} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
        make_fecr "inputs/${TAG}_h${EPS}.in" "${TAG}_h${EPS}" "scf" \
            "$SD" "0.000000" "0.000000" \
            "0.000000" "$SD" "0.000000" \
            "0.000000" "0.000000" "$SD" "$N_CR"
        run_pw "inputs/${TAG}_h${EPS}.in" "outputs/${TAG}_h${EPS}.out"
        P=$(get_P "outputs/${TAG}_h${EPS}.out")
        echo "${EPS} ${P}" >> "dft_data/${TAG}_hydro.dat"
        echo "    hydro ε=${EPS}: P=${P} kbar"; progress
    done

    # ── Tetragonal ────────────────────────────────────────────
    echo "  [${TAG}] Tetragonal..."
    > "dft_data/${TAG}_tetra.dat"
    for EPS in $STRAINS; do
        if [ "$EPS" = "0.00" ]; then SXY=$SA_EQ; SZ=$SA_EQ
        else
            SZ=$(echo "${SA_EQ} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
            SXY=$(echo "${SA_EQ} / sqrt(1 + ${EPS})" | bc -l | xargs printf "%.6f")
        fi
        make_fecr "inputs/${TAG}_t${EPS}.in" "${TAG}_t${EPS}" "scf" \
            "$SXY" "0.000000" "0.000000" \
            "0.000000" "$SXY" "0.000000" \
            "0.000000" "0.000000" "$SZ" "$N_CR"
        run_pw "inputs/${TAG}_t${EPS}.in" "outputs/${TAG}_t${EPS}.out"
        S11=$(get_s11 "outputs/${TAG}_t${EPS}.out")
        S33=$(get_s33 "outputs/${TAG}_t${EPS}.out")
        DS=$(safe_diff "$S33" "$S11")
        echo "${EPS} ${S11} ${S33} ${DS}" >> "dft_data/${TAG}_tetra.dat"
        echo "    tetra ε=${EPS}: Δσ=${DS} kbar"; progress
    done

    # ── Shear ─────────────────────────────────────────────────
    echo "  [${TAG}] Shear..."
    > "dft_data/${TAG}_shear.dat"
    for EPS in $STRAINS; do
        SH=$(echo "${SA_EQ} * ${EPS}" | bc -l | xargs printf "%.6f")
        make_fecr "inputs/${TAG}_s${EPS}.in" "${TAG}_s${EPS}" "scf" \
            "$SA_EQ" "$SH" "0.000000" \
            "$SH" "$SA_EQ" "0.000000" \
            "0.000000" "0.000000" "$SA_EQ" "$N_CR"
        run_pw "inputs/${TAG}_s${EPS}.in" "outputs/${TAG}_s${EPS}.out"
        S12=$(get_s12 "outputs/${TAG}_s${EPS}.out")
        echo "${EPS} ${S12}" >> "dft_data/${TAG}_shear.dat"
        echo "    shear ε=${EPS}: σ12=${S12} kbar"; progress
    done

    echo "FeCr ${N_CR} ${X} ${SA_EQ} ${ET} ${MV}" >> dft_data/fecr_summary.dat
    echo "  [${TAG}] ✓ Complete"

    # ── Write checkpoint — only after all 10 jobs for this tag succeeded ──
    mark_done "$TAG"
done

# ══════════════════════════════════════════════════════════════
# SYSTEM 2: Cu-Ni (FCC, 32 atoms)
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  SYSTEM 2: Cu-Ni (FCC, 32 atoms) — 17 compositions ║"
echo "╚══════════════════════════════════════════════════════╝"

SA_CU=7.2300; SA_NI=7.0480
[ -f "dft_data/cuni_summary.dat" ] || echo "# system n_sol x_sol SA_eq E_tot" > dft_data/cuni_summary.dat

for N_NI in $(seq 0 2 32); do
    X=$(echo "scale=4; ${N_NI} / 32" | bc -l)
    NC=$((32 - N_NI))
    TAG=$(printf "cuni_cu%02dni%02d" $NC $N_NI)
    SA=$(echo "${SA_CU} * (1.0 - ${X}) + ${SA_NI} * ${X}" | bc -l | xargs printf "%.6f")

    # ── Checkpoint check ──────────────────────────────────────
    if is_done "$TAG"; then
        echo "── ${TAG}: already complete, skipping ──"
        DONE=$((DONE + 10))
        continue
    fi

    echo ""; echo "── Cu${NC}Ni${N_NI} (x=${X}) ──────────────────"

    # ── vc-relax ──────────────────────────────────────────────
    echo "  [${TAG}] vc-relax..."
    make_cuni "inputs/${TAG}_vcr.in" "${TAG}_vcr" "vc-relax" \
        "$SA" "0.000000" "0.000000" \
        "0.000000" "$SA" "0.000000" \
        "0.000000" "0.000000" "$SA" "$N_NI"
    run_pw "inputs/${TAG}_vcr.in" "outputs/${TAG}_vcr.out"; progress

    SA_EQ=$(grep -A 1 "CELL_PARAMETERS" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $1}')
    [ -z "$SA_EQ" ] || [ "$SA_EQ" = "CELL_PARAMETERS" ] && SA_EQ=$SA
    ET=$(grep "!" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $5}')
    echo "  [${TAG}] SA=${SA_EQ}, E=${ET}"

    # ── Hydrostatic ───────────────────────────────────────────
    echo "  [${TAG}] Hydrostatic..."
    > "dft_data/${TAG}_hydro.dat"
    for EPS in $STRAINS; do
        SD=$(echo "${SA_EQ} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
        make_cuni "inputs/${TAG}_h${EPS}.in" "${TAG}_h${EPS}" "scf" \
            "$SD" "0.000000" "0.000000" \
            "0.000000" "$SD" "0.000000" \
            "0.000000" "0.000000" "$SD" "$N_NI"
        run_pw "inputs/${TAG}_h${EPS}.in" "outputs/${TAG}_h${EPS}.out"
        P=$(get_P "outputs/${TAG}_h${EPS}.out")
        echo "${EPS} ${P}" >> "dft_data/${TAG}_hydro.dat"
        echo "    hydro ε=${EPS}: P=${P} kbar"; progress
    done

    # ── Tetragonal ────────────────────────────────────────────
    echo "  [${TAG}] Tetragonal..."
    > "dft_data/${TAG}_tetra.dat"
    for EPS in $STRAINS; do
        if [ "$EPS" = "0.00" ]; then SXY=$SA_EQ; SZ=$SA_EQ
        else
            SZ=$(echo "${SA_EQ} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
            SXY=$(echo "${SA_EQ} / sqrt(1 + ${EPS})" | bc -l | xargs printf "%.6f")
        fi
        make_cuni "inputs/${TAG}_t${EPS}.in" "${TAG}_t${EPS}" "scf" \
            "$SXY" "0.000000" "0.000000" \
            "0.000000" "$SXY" "0.000000" \
            "0.000000" "0.000000" "$SZ" "$N_NI"
        run_pw "inputs/${TAG}_t${EPS}.in" "outputs/${TAG}_t${EPS}.out"
        S11=$(get_s11 "outputs/${TAG}_t${EPS}.out")
        S33=$(get_s33 "outputs/${TAG}_t${EPS}.out")
        DS=$(safe_diff "$S33" "$S11")
        echo "${EPS} ${S11} ${S33} ${DS}" >> "dft_data/${TAG}_tetra.dat"
        echo "    tetra ε=${EPS}: Δσ=${DS} kbar"; progress
    done

    # ── Shear ─────────────────────────────────────────────────
    echo "  [${TAG}] Shear..."
    > "dft_data/${TAG}_shear.dat"
    for EPS in $STRAINS; do
        SH=$(echo "${SA_EQ} * ${EPS}" | bc -l | xargs printf "%.6f")
        make_cuni "inputs/${TAG}_s${EPS}.in" "${TAG}_s${EPS}" "scf" \
            "$SA_EQ" "$SH" "0.000000" \
            "$SH" "$SA_EQ" "0.000000" \
            "0.000000" "0.000000" "$SA_EQ" "$N_NI"
        run_pw "inputs/${TAG}_s${EPS}.in" "outputs/${TAG}_s${EPS}.out"
        S12=$(get_s12 "outputs/${TAG}_s${EPS}.out")
        echo "${EPS} ${S12}" >> "dft_data/${TAG}_shear.dat"
        echo "    shear ε=${EPS}: σ12=${S12} kbar"; progress
    done

    echo "CuNi ${N_NI} ${X} ${SA_EQ} ${ET}" >> dft_data/cuni_summary.dat
    echo "  [${TAG}] ✓ Complete"

    # ── Write checkpoint — only after all 10 jobs for this tag succeeded ──
    mark_done "$TAG"
done

# ══════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL DONE — ${TOTAL_MIN} min"
echo "  Finished: $(date)"
echo "══════════════════════════════════════════════════════"
echo "  Fe-Cr: $(ls dft_data/fecr_*_hydro.dat 2>/dev/null | wc -l) compositions"
echo "  Cu-Ni: $(ls dft_data/cuni_*_hydro.dat 2>/dev/null | wc -l) compositions"
echo "  Checkpoints: $(ls checkpoints/*.done 2>/dev/null | wc -l)/34 tags complete"
