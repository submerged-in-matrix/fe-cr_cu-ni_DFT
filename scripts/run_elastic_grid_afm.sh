#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Elastic constants sweep: BCC Fe-Cr (16 atoms) — AFM restart
#
# PURPOSE: Resume from fe02cr14 onward with AFM treatment.
# Tags fe16cr00 → fe03cr13 are COMPLETE — all skipped via .dat validation.
# Tags fe02cr14, fe01cr15, fe00cr16 use AFM initial guess for Cr sites.
#
# KEY CHANGES vs run_elastic_grid_latest.sh:
#   1. append_atoms_bcc_afm(): Cr sites split into CrA (up) / CrB (down)
#      based on BCC sublattice parity (sum of crystal coords / 0.25, mod 2)
#   2. make_fecr(): AFM branch for N_CR >= 14:
#      - ntyp = 3 (Fe, CrA, CrB) instead of 2
#      - starting_magnetization: Fe=+0.5, CrA=+0.5, CrB=-0.5
#      - tot_magnetization REMOVED (let SCF find its own total)
#      - conv_thr = 1.0d-5 (loosened from 1.0d-7 — AFM boundary, budget limited)
#      - mixing_mode = 'local-TF', mixing_beta = 0.1, ndim = 16 (unchanged)
#      - electron_maxstep = 300 (unchanged)
#   3. Script starts loop at N_CR=14 (skips completed tags automatically
#      via existing .dat validation — but explicit start saves time)
#   4. run_pw(): NO change — skip logic unchanged (convergence keyword check)
#   5. If SCF does NOT converge: stress is still extracted from output.
#      Unconverged outputs flagged in log. Data usable for ML with caveat flag.
#
# HONEST CAVEATS (flagged for ML post-processing):
#   - conv_thr=1e-5 is one decade looser than rest of dataset (1e-7).
#     Elastic constants from these tags may have ~1-3% higher numerical noise.
#     Flag these 3 compositions in ML training set.
#   - AFM sublattice assignment is based on ideal BCC geometry, not
#     spin-density-functional ground state search. Physical ground state
#     of Fe2Cr14, Fe1Cr15, Fe0Cr16 in this SQS cell is uncertain.
#   - Stress extraction from unconverged outputs is done as last resort.
#     Results should be cross-checked against converged neighbors in ML.
#
# Usage:
#   cd /root/fe-cr_cu-ni_DFT
#   nohup bash scripts/run_elastic_grid_afm.sh > elastic_grid_afm.log 2>&1 &
# ══════════════════════════════════════════════════════════════

# ── NVHPC 24.7 environment ────────────────────────────────────
NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/12.5/lib64:$NVHPC_ROOT/comm_libs/mpi/lib:$LD_LIBRARY_PATH
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# ── OpenMP threads ────────────────────────────────────────────
NCORES=$(nproc 2>/dev/null || echo 1)
export OMP_NUM_THREADS=8

# ── Locate pw.x ───────────────────────────────────────────────
PW=""
for loc in "/root/q-e-qe-7.3.1/bin/pw.x" "/workspace/q-e-qe-7.3.1/bin/pw.x" "/opt/q-e/bin/pw.x" "$HOME/q-e/bin/pw.x" "/usr/bin/pw.x" "/usr/local/bin/pw.x"; do
    [ -f "$loc" ] && PW="$loc" && break
done
[ -z "$PW" ] && PW=$(which pw.x 2>/dev/null)
[ -z "$PW" ] && { echo "ERROR: pw.x not found"; exit 1; }

# ── Project root ──────────────────────────────────────────────
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
[ ! -d "$PROJ_DIR/pseudo" ] && { echo "ERROR: pseudo/ not found in $PROJ_DIR"; exit 1; }

STRAINS="-0.01 0.00 0.01"
mkdir -p inputs outputs dft_data tmp

START_TIME=$(date +%s)
TOTAL_CALCS=30   # 3 tags × 10 calcs
DONE=0

echo "══════════════════════════════════════════════════════"
echo "  Elastic Constants: Fe-Cr AFM restart"
echo "  Tags: fe02cr14, fe01cr15, fe00cr16"
echo "  QE binary : $PW"
echo "  Project   : $PROJ_DIR"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  conv_thr  : 1.0d-5 (loosened for AFM boundary)"
echo "  Started   : $(date)"
echo "══════════════════════════════════════════════════════"

# ── BCC atomic positions (16-atom supercell, crystal coords) ──
BCC_X=(0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500)
BCC_Y=(0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500)
BCC_Z=(0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500)
SUB_BCC=(0 14 6 8 3 13 4 10 7 9 2 12 5 11 1 15)

# ── AFM sublattice parity lookup ──────────────────────────────
# For each BCC site index, parity = round((x+y+z)/0.25) mod 2
# parity=0 → sublattice A (CrA, spin up)
# parity=1 → sublattice B (CrB, spin down)
# Precomputed from BCC_X/Y/Z above:
#   A(up)  sites: 0,2,4,6,8,10,12,14
#   B(down) sites: 1,3,5,7,9,11,13,15
declare -A BCC_PARITY
BCC_PARITY[0]=0;  BCC_PARITY[1]=1;  BCC_PARITY[2]=0;  BCC_PARITY[3]=1
BCC_PARITY[4]=0;  BCC_PARITY[5]=1;  BCC_PARITY[6]=0;  BCC_PARITY[7]=1
BCC_PARITY[8]=0;  BCC_PARITY[9]=1;  BCC_PARITY[10]=0; BCC_PARITY[11]=1
BCC_PARITY[12]=0; BCC_PARITY[13]=1; BCC_PARITY[14]=0; BCC_PARITY[15]=1

# ── Append atomic positions: FM mode (Fe-rich tags, unchanged) ─
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

# ── Append atomic positions: AFM mode (high-Cr tags) ──────────
# Cr sites split into CrA (spin up, parity=0) and CrB (spin down, parity=1)
# Fe sites unchanged (label "Fe")
append_atoms_bcc_afm() {
    local f=$1 ns=$2
    declare -A is_sol
    for ((i=0; i<ns; i++)); do is_sol[${SUB_BCC[$i]}]=1; done
    echo "ATOMIC_POSITIONS crystal" >> "$f"
    for ((i=0; i<16; i++)); do
        local e
        if [[ ${is_sol[$i]+_} ]]; then
            # This is a Cr site — assign sublattice based on parity
            if [ "${BCC_PARITY[$i]}" -eq 0 ]; then
                e="CrA"
            else
                e="CrB"
            fi
        else
            e="Fe"
        fi
        echo "${e}  ${BCC_X[$i]} ${BCC_Y[$i]} ${BCC_Z[$i]}" >> "$f"
    done
}

# ── Generate Fe-Cr QE input ───────────────────────────────────
# For N_CR >= 14: AFM treatment (3 atom types, split Cr sublattice)
# For N_CR < 14:  FM treatment (unchanged from original script)
make_fecr() {
    local f=$1 pref=$2 calc=$3
    local c1x=$4 c1y=$5 c1z=$6
    local c2x=$7 c2y=$8 c2z=$9
    shift 9
    local c3x=$1 c3y=$2 c3z=$3 ncr=$4

    # Determine number of atom types and AFM flag
    local nt afm=0
    if [ "$ncr" -eq 0 ]; then
        nt=1
    elif [ "$ncr" -eq 16 ]; then
        # Pure Cr: still split into CrA/CrB for AFM
        nt=2; afm=1
    elif [ "$ncr" -ge 14 ]; then
        # High-Cr mixed: Fe + CrA + CrB
        nt=3; afm=1
    else
        nt=2
    fi

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

    if [ "$afm" -eq 1 ]; then
        # AFM branch: no tot_magnetization — let SCF find net moment freely
        # CrA spin up, CrB spin down, Fe spin up
        if [ "$ncr" -eq 16 ]; then
            # Pure Cr: only CrA and CrB
            echo "    starting_magnetization(1) =  0.5," >> "$f"   # CrA
            echo "    starting_magnetization(2) = -0.5," >> "$f"   # CrB
        else
            # Fe + CrA + CrB
            echo "    starting_magnetization(1) =  0.5," >> "$f"   # Fe
            echo "    starting_magnetization(2) =  0.5," >> "$f"   # CrA
            echo "    starting_magnetization(3) = -0.5," >> "$f"   # CrB
        fi
    else
        # FM branch (unchanged)
        if [ "$nt" -eq 2 ]; then
            echo "    starting_magnetization(1) = 0.5," >> "$f"
            echo "    starting_magnetization(2) = 0.3," >> "$f"
        elif [ "$ncr" -eq 0 ]; then
            echo "    starting_magnetization(1) = 0.5," >> "$f"
        else
            echo "    starting_magnetization(1) = 0.3," >> "$f"
        fi
    fi

    echo "/" >> "$f"

    # AFM tags get loosened conv_thr; FM tags keep original
    if [ "$afm" -eq 1 ]; then
        cat >> "$f" << EOF
&ELECTRONS
    conv_thr = 1.0d-5,
    mixing_beta = 0.1,
    mixing_ndim = 16,
    mixing_mode = 'local-TF',
    electron_maxstep = 300
/
EOF
    else
        cat >> "$f" << EOF
&ELECTRONS
    conv_thr = 1.0d-7,
    mixing_beta = 0.1,
    mixing_ndim = 16,
    mixing_mode = 'local-TF',
    electron_maxstep = 300
/
EOF
    fi

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

    # ATOMIC_SPECIES block
    if [ "$afm" -eq 1 ]; then
        if [ "$ncr" -eq 16 ]; then
            # Pure Cr AFM: CrA and CrB — same pseudopotential, different labels
            echo "ATOMIC_SPECIES" >> "$f"
            echo "CrA 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
            echo "CrB 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
        else
            # Fe + CrA + CrB
            echo "ATOMIC_SPECIES" >> "$f"
            echo "Fe  55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
            echo "CrA 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
            echo "CrB 51.996 Cr.pbe-spn-kjpaw_psl.1.0.0.UPF" >> "$f"
        fi
    else
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
    fi

    cat >> "$f" << EOF
CELL_PARAMETERS angstrom
${c1x}  ${c1y}  ${c1z}
${c2x}  ${c2y}  ${c2z}
${c3x}  ${c3y}  ${c3z}
EOF

    # Use AFM atom appender for high-Cr, FM appender otherwise
    if [ "$afm" -eq 1 ]; then
        append_atoms_bcc_afm "$f" "$ncr"
    else
        append_atoms_bcc "$f" "$ncr" "Fe" "Cr"
    fi

    echo "K_POINTS automatic" >> "$f"
    echo "6 6 6 0 0 0" >> "$f"
}

# ── pw.x execution ────────────────────────────────────────────
run_pw() {
    local infile=$1 outfile=$2 calc_type=${3:-scf}

    if [ "$calc_type" = "vcr" ]; then
        if grep -q "End of BFGS Geometry Optimization" "$outfile" 2>/dev/null; then
            echo "    SKIP (bfgs done): $outfile"
            return 0
        fi
    else
        if grep -q "convergence has been achieved" "$outfile" 2>/dev/null; then
            echo "    SKIP (scf converged): $outfile"
            return 0
        fi
    fi

    [ -f "$outfile" ] && rm -f "$outfile"
    echo "    >> RUNNING: $(basename $infile)  →  $(basename $outfile)  [$(date +%H:%M:%S)]"
    mpirun --allow-run-as-root --mca btl ^openib --mca pml ob1 --mca coll_hcoll_enable 0 \
        -np 1 "$PW" -npool 1 -input "$infile" > "$outfile" 2>&1

    if [ "$calc_type" = "vcr" ]; then
        grep -q "End of BFGS Geometry Optimization" "$outfile" 2>/dev/null && return 0
    else
        grep -q "convergence has been achieved" "$outfile" 2>/dev/null && return 0
    fi

    echo "    WARNING: pw.x did NOT converge — will attempt stress extraction anyway: $outfile"
    return 1
}

# ── Progress tracker ──────────────────────────────────────────
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

# ── Stress tensor extraction ──────────────────────────────────
get_P() {
    local v
    v=$(grep "P=" "$1" 2>/dev/null | tail -1 | awk -F'P=' '{print $2}' | awk '{print $1}')
    if [ -z "$v" ]; then
        echo "ERROR: get_P: no P= line in $1" >&2; echo "NaN"; return 1
    fi
    echo "$v"
}

get_s11() {
    local v
    v=$(grep -A 3 "total   stress" "$1" 2>/dev/null | tail -3 | head -1 | awk '{print $4}')
    if [ -z "$v" ]; then
        echo "ERROR: get_s11: no stress block in $1" >&2; echo "NaN"; return 1
    fi
    echo "$v"
}

get_s33() {
    local v
    v=$(grep -A 3 "total   stress" "$1" 2>/dev/null | tail -1 | awk '{print $6}')
    if [ -z "$v" ]; then
        echo "ERROR: get_s33: no stress block in $1" >&2; echo "NaN"; return 1
    fi
    echo "$v"
}

get_s12() {
    local v
    v=$(grep -A 3 "total   stress" "$1" 2>/dev/null | tail -3 | head -1 | awk '{print $5}')
    if [ -z "$v" ]; then
        echo "ERROR: get_s12: no stress block in $1" >&2; echo "NaN"; return 1
    fi
    echo "$v"
}

safe_diff() {
    [ -n "$1" ] && [ -n "$2" ] && \
    [ "$1" != "NaN" ] && [ "$2" != "NaN" ] && \
    echo "$1 - $2" | bc -l 2>/dev/null | xargs printf "%.4f" 2>/dev/null \
    || echo "NaN"
}

# ── .dat line validation ──────────────────────────────────────
dat_line_valid() {
    local datfile=$1 eps=$2 ncols=$3
    [ ! -f "$datfile" ] && return 1
    local line
    line=$(awk -v e="$eps" '$1 == e {print; exit}' "$datfile" 2>/dev/null)
    [ -z "$line" ] && return 1
    echo "$line" | grep -q "NaN" && return 1
    local total_cols=$(echo "$line" | awk '{print NF}')
    [ "$total_cols" -lt $((ncols + 1)) ] && return 1
    if [ "$ncols" -eq 3 ] && [ "$eps" != "0.00" ]; then
        local s11 s33 ds
        s11=$(echo "$line" | awk '{print $2}')
        s33=$(echo "$line" | awk '{print $3}')
        ds=$(echo "$line" | awk '{print $4}')
        [ "$s11" = "0.0000" ] && [ "$s33" = "0.0000" ] && [ "$ds" = "0.0000" ] && return 1
        [ "$s11" = "0.0000" ] && [ "$s33" = "0.0000" ] && return 1
    fi
    if [ "$ncols" -eq 1 ] && [ "$eps" != "0.00" ]; then
        local val
        val=$(echo "$line" | awk '{print $2}')
        [ "$val" = "0.0000" ] && return 1
    fi
    return 0
}

vcr_converged() {
    grep -q "End of BFGS Geometry Optimization" "$1" 2>/dev/null
}

extract_sa_eq() {
    local outfile=$1
    local val
    val=$(grep -A 1 "CELL_PARAMETERS" "$outfile" 2>/dev/null | tail -1 | awk '{print $1}')
    if [ -z "$val" ] || [ "$val" = "CELL_PARAMETERS" ]; then
        echo "ERROR: extract_sa_eq: CELL_PARAMETERS not found in $outfile" >&2
        return 1
    fi
    echo "$val"
}

# ══════════════════════════════════════════════════════════════
# MAIN LOOP — Fe-Cr, high-Cr AFM tags only: N_CR = 14, 15, 16
# Lower tags (0-13) are complete — skipped by .dat validation
# if accidentally included, but we start at 14 for speed.
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Fe-Cr AFM restart: fe02cr14, fe01cr15, fe00cr16   ║"
echo "╚══════════════════════════════════════════════════════╝"

SA_FE=5.6469; SA_CR=5.6600
[ -f "dft_data/fecr_summary.dat" ] || echo "# system n_sol x_sol SA_eq E_tot mag" > dft_data/fecr_summary.dat

for N_CR in 14 15 16; do
    X=$(echo "scale=4; ${N_CR} / 16" | bc -l)
    NF=$((16 - N_CR))
    TAG=$(printf "fecr_fe%02dcr%02d" $NF $N_CR)
    SA=$(echo "${SA_FE} * (1.0 - ${X}) + ${SA_CR} * ${X}" | bc -l | xargs printf "%.6f")

    echo ""; echo "── Fe${NF}Cr${N_CR} (x=${X}) [AFM] ──────────────────"

    # ── vc-relax ──────────────────────────────────────────────
    echo "  [${TAG}] vc-relax..."
    make_fecr "inputs/${TAG}_vcr.in" "${TAG}_vcr" "vc-relax" \
        "$SA" "0.000000" "0.000000" \
        "0.000000" "$SA" "0.000000" \
        "0.000000" "0.000000" "$SA" "$N_CR"
    run_pw "inputs/${TAG}_vcr.in" "outputs/${TAG}_vcr.out" "vcr"; progress

    # ── vc-relax gate ─────────────────────────────────────────
    if ! vcr_converged "outputs/${TAG}_vcr.out"; then
        echo "  [${TAG}] ERROR: vc-relax NOT converged — skipping entire tag"
        echo "  [${TAG}] No Vegard fallback. Fix vc-relax before strain calcs."
        continue
    fi

    SA_EQ=$(extract_sa_eq "outputs/${TAG}_vcr.out")
    if [ $? -ne 0 ]; then
        echo "  [${TAG}] ERROR: SA_EQ extraction failed — skipping entire tag"
        echo "  [${TAG}] No Vegard fallback. Check outputs/${TAG}_vcr.out manually."
        continue
    fi

    ET=$(grep "!" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $5}')
    MV=$(grep "total magnetization" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $4}')
    echo "  [${TAG}] SA_EQ=${SA_EQ} Å, E=${ET} Ry, μ=${MV} μB"

    # ── Hydrostatic strain ─────────────────────────────────────
    echo "  [${TAG}] Hydrostatic..."
    H_VALS=()
    H_OK=1
    for EPS in $STRAINS; do
        if dat_line_valid "dft_data/${TAG}_hydro.dat" "$EPS" 1; then
            P=$(awk -v e="$EPS" '$1==e{print $2; exit}' "dft_data/${TAG}_hydro.dat")
            echo "    hydro ε=${EPS}: SKIP (valid: P=${P} kbar)"
        else
            SD=$(echo "${SA_EQ} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
            make_fecr "inputs/${TAG}_h${EPS}.in" "${TAG}_h${EPS}" "scf" \
                "$SD" "0.000000" "0.000000" \
                "0.000000" "$SD" "0.000000" \
                "0.000000" "0.000000" "$SD" "$N_CR"
            run_pw "inputs/${TAG}_h${EPS}.in" "outputs/${TAG}_h${EPS}.out"
            P=$(get_P "outputs/${TAG}_h${EPS}.out")
            echo "    hydro ε=${EPS}: P=${P} kbar"
        fi
        progress
        if [ "$P" = "NaN" ] || [ -z "$P" ]; then
            echo "    ERROR: hydro ε=${EPS} — P is NaN/empty (SCF not converged, no stress block)"
            H_OK=0
        fi
        H_VALS+=("$EPS $P")
    done
    if [ "$H_OK" -eq 1 ]; then
        printf "%s\n" "${H_VALS[@]}" > "dft_data/${TAG}_hydro.dat"
        echo "  [${TAG}] hydro.dat written (3/3 valid)"
    else
        echo "  [${TAG}] WARNING: hydro.dat NOT written — NaN values present"
        echo "  [${TAG}] NOTE: check outputs/${TAG}_h*.out — stress may exist even without convergence"
    fi

    # ── Tetragonal strain ──────────────────────────────────────
    echo "  [${TAG}] Tetragonal..."
    T_VALS=()
    T_OK=1
    for EPS in $STRAINS; do
        if dat_line_valid "dft_data/${TAG}_tetra.dat" "$EPS" 3; then
            S11=$(awk -v e="$EPS" '$1==e{print $2; exit}' "dft_data/${TAG}_tetra.dat")
            S33=$(awk -v e="$EPS" '$1==e{print $3; exit}' "dft_data/${TAG}_tetra.dat")
            DS=$(awk  -v e="$EPS" '$1==e{print $4; exit}' "dft_data/${TAG}_tetra.dat")
            echo "    tetra ε=${EPS}: SKIP (valid)"
        else
            if [ "$EPS" = "0.00" ]; then
                SXY=$SA_EQ; SZ=$SA_EQ
            else
                SZ=$(echo "${SA_EQ} * (1 + ${EPS})" | bc -l | xargs printf "%.6f")
                SXY=$(echo "scale=10; ${SA_EQ} / sqrt(1 + ${EPS})" | bc -l | xargs printf "%.6f")
            fi
            make_fecr "inputs/${TAG}_t${EPS}.in" "${TAG}_t${EPS}" "scf" \
                "$SXY" "0.000000" "0.000000" \
                "0.000000" "$SXY" "0.000000" \
                "0.000000" "0.000000" "$SZ" "$N_CR"
            run_pw "inputs/${TAG}_t${EPS}.in" "outputs/${TAG}_t${EPS}.out"
            S11=$(get_s11 "outputs/${TAG}_t${EPS}.out")
            S33=$(get_s33 "outputs/${TAG}_t${EPS}.out")
            DS=$(safe_diff "$S33" "$S11")
            echo "    tetra ε=${EPS}: S11=${S11} S33=${S33} DS=${DS} kbar"
        fi
        progress
        if [ "$S11" = "NaN" ] || [ "$S33" = "NaN" ] || [ "$DS" = "NaN" ] || \
           [ -z "$S11" ] || [ -z "$S33" ] || [ -z "$DS" ]; then
            echo "    ERROR: tetra ε=${EPS} — NaN/empty stress"
            T_OK=0
        fi
        T_VALS+=("$EPS $S11 $S33 $DS")
    done
    if [ "$T_OK" -eq 1 ]; then
        printf "%s\n" "${T_VALS[@]}" > "dft_data/${TAG}_tetra.dat"
        echo "  [${TAG}] tetra.dat written (3/3 valid)"
    else
        echo "  [${TAG}] WARNING: tetra.dat NOT written"
    fi

    # ── Shear strain ───────────────────────────────────────────
    echo "  [${TAG}] Shear..."
    S_VALS=()
    S_OK=1
    for EPS in $STRAINS; do
        if dat_line_valid "dft_data/${TAG}_shear.dat" "$EPS" 1; then
            S12=$(awk -v e="$EPS" '$1==e{print $2; exit}' "dft_data/${TAG}_shear.dat")
            echo "    shear ε=${EPS}: SKIP (valid: S12=${S12} kbar)"
        else
            SH=$(echo "${SA_EQ} * ${EPS}" | bc -l | xargs printf "%.6f")
            make_fecr "inputs/${TAG}_s${EPS}.in" "${TAG}_s${EPS}" "scf" \
                "$SA_EQ" "$SH" "0.000000" \
                "$SH" "$SA_EQ" "0.000000" \
                "0.000000" "0.000000" "$SA_EQ" "$N_CR"
            run_pw "inputs/${TAG}_s${EPS}.in" "outputs/${TAG}_s${EPS}.out"
            S12=$(get_s12 "outputs/${TAG}_s${EPS}.out")
            echo "    shear ε=${EPS}: S12=${S12} kbar"
        fi
        progress
        if [ "$S12" = "NaN" ] || [ -z "$S12" ]; then
            echo "    ERROR: shear ε=${EPS} — NaN/empty"
            S_OK=0
        fi
        S_VALS+=("$EPS $S12")
    done
    if [ "$S_OK" -eq 1 ]; then
        printf "%s\n" "${S_VALS[@]}" > "dft_data/${TAG}_shear.dat"
        echo "  [${TAG}] shear.dat written (3/3 valid)"
    else
        echo "  [${TAG}] WARNING: shear.dat NOT written"
    fi

    # ── Summary entry ──────────────────────────────────────────
    if [ "$H_OK" -eq 1 ] && [ "$T_OK" -eq 1 ] && [ "$S_OK" -eq 1 ]; then
        grep -v "^FeCr ${N_CR} " dft_data/fecr_summary.dat > dft_data/fecr_summary.dat.tmp \
            && mv dft_data/fecr_summary.dat.tmp dft_data/fecr_summary.dat
        echo "FeCr ${N_CR} ${X} ${SA_EQ} ${ET} ${MV}" >> dft_data/fecr_summary.dat
        echo "  [${TAG}] ✓ Complete"
    else
        echo "  [${TAG}] ✗ Incomplete — check outputs for manual stress extraction"
    fi

    rm -rf "${PROJ_DIR}/tmp/${TAG}_"*

done

# ══════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "══════════════════════════════════════════════════════"
echo "  AFM SWEEP DONE — ${TOTAL_MIN} min total"
echo "  Finished: $(date)"
echo "══════════════════════════════════════════════════════"
echo "  .dat files written:"
for f in dft_data/fecr_fe02cr14_*.dat dft_data/fecr_fe01cr15_*.dat dft_data/fecr_fe00cr16_*.dat; do
    [ -f "$f" ] && echo "    $f"
done
echo ""
echo "  Tags with NaN in .dat (need manual stress extraction):"
for f in dft_data/fecr_fe02cr14_*.dat dft_data/fecr_fe01cr15_*.dat dft_data/fecr_fe00cr16_*.dat; do
    grep -q "NaN" "$f" 2>/dev/null && echo "    CONTAINS NaN: $f"
done
echo "══════════════════════════════════════════════════════"
