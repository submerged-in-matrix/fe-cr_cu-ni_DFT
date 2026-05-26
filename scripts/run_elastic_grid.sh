#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Elastic constants sweep: BCC Fe-Cr (16 atoms)
# 17 compositions → 170 QE calculations total (10 per tag).
# 64 calculations already converged and reusable from prior run.
#
# Cu-Ni (FCC, 32 atoms) system is commented out below.
# Deferred to a later run — not dropped permanently.
#
# Usage:
#   cd /path/to/project_root   # directory containing pseudo/ and scripts/
#   nohup bash scripts/run_elastic_grid.sh > elastic_grid.log 2>&1 &
#
# Resume / reuse logic (NO tag-level checkpoint folder):
#   - vc-relax  : skip if "End of BFGS Geometry Optimization" found in output
#   - strain SCF: skip if "convergence has been achieved" found in output
#   - .dat files: per-line validation; only rerun missing/invalid strains
#   - vc-relax gate: if vc-relax not converged → skip entire tag (no .dat touched)
#   - SA_EQ: never silently falls back to Vegard — hard abort if CELL_PARAMETERS missing
#
# Changes from previous version:
#   1. Removed tag-level checkpoint system entirely
#   2. run_pw() skip: convergence keyword only (not JOB DONE)
#   3. run_pw() fixed: mpirun -np 1 with --mca flags, OMP_NUM_THREADS=8
#   4. electron_maxstep = 300 (was default 100) — primary fix for high-Cr failures
#   5. mixing_beta = 0.1 (unchanged — root cause was maxstep, not beta)
#   6. vc-relax gate per tag — corrupted tags skipped cleanly
#   7. SA_EQ extraction: hard error if CELL_PARAMETERS absent
#   8. Stress extractors return NaN on failure — never silently zero
#   9. .dat write: validate per-line before skipping; write only when all 3 strains valid
#  10. Tetra .dat: confirmed 4-column format (EPS S11 S33 DS)
# ══════════════════════════════════════════════════════════════

# ── NVHPC 24.7 environment ────────────────────────────────────
# Configured for Vast.ai RTX 4090 (NVHPC 24.7, CUDA 12.5, Ubuntu 22.04).
# For a different instance, verify this path with: ls /opt/nvidia/hpc_sdk/Linux_x86_64/
# For a non-NVHPC environment (e.g. plain CUDA + gfortran build), comment this block out.
NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/12.5/lib64:$NVHPC_ROOT/comm_libs/mpi/lib:$LD_LIBRARY_PATH
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# ── OpenMP threads ────────────────────────────────────────────
# Capped at 8: more threads on a single-GPU node adds overhead, not speed.
# RTX 3090 run confirmed 1 MPI rank + 8 OMP threads is optimal here.
NCORES=$(nproc 2>/dev/null || echo 1)
export OMP_NUM_THREADS=8

# ── Locate pw.x ───────────────────────────────────────────────
PW=""
for loc in "/root/q-e-qe-7.3.1/bin/pw.x" "/workspace/q-e-qe-7.3.1/bin/pw.x" "/opt/q-e/bin/pw.x" "$HOME/q-e/bin/pw.x" "/usr/bin/pw.x" "/usr/local/bin/pw.x"; do
    [ -f "$loc" ] && PW="$loc" && break
done
[ -z "$PW" ] && PW=$(which pw.x 2>/dev/null)
[ -z "$PW" ] && { echo "ERROR: pw.x not found"; exit 1; }

# ── Project root (parent of scripts/) ─────────────────────────
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
[ ! -d "$PROJ_DIR/pseudo" ] && { echo "ERROR: pseudo/ not found in $PROJ_DIR"; exit 1; }

STRAINS="-0.01 0.00 0.01"
mkdir -p inputs outputs dft_data tmp

START_TIME=$(date +%s)
TOTAL_CALCS=170   # Fe-Cr only: 17 tags × 10 calcs. 64 already converged → ~106 to run.
DONE=0

echo "══════════════════════════════════════════════════════"
echo "  Elastic Constants: Fe-Cr (BCC) — 17 compositions"
echo "  Cu-Ni system deferred (commented out below)"
echo "  QE binary : $PW"
echo "  Project   : $PROJ_DIR"
echo "  CPU cores : $NCORES  (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
echo "  Total calc: $TOTAL_CALCS  (64 reusable from prior run)"
echo "  Started   : $(date)"
echo "══════════════════════════════════════════════════════"

# ── BCC atomic positions (16-atom supercell, crystal coords) ──
BCC_X=(0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500)
BCC_Y=(0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500)
BCC_Z=(0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500 0.0000 0.2500 0.5000 0.7500)
SUB_BCC=(0 14 6 8 3 13 4 10 7 9 2 12 5 11 1 15)

# ── FCC atomic positions (32-atom supercell, crystal coords) ──
FCC_X=(0.0000 0.2500 0.2500 0.0000 0.0000 0.2500 0.2500 0.0000 0.0000 0.2500 0.2500 0.0000 0.0000 0.2500 0.2500 0.0000 0.5000 0.7500 0.7500 0.5000 0.5000 0.7500 0.7500 0.5000 0.5000 0.7500 0.7500 0.5000 0.5000 0.7500 0.7500 0.5000)
FCC_Y=(0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.0000 0.2500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500 0.5000 0.7500)
FCC_Z=(0.0000 0.0000 0.2500 0.2500 0.5000 0.5000 0.7500 0.7500 0.0000 0.0000 0.2500 0.2500 0.5000 0.5000 0.7500 0.7500 0.0000 0.0000 0.2500 0.2500 0.5000 0.5000 0.7500 0.7500 0.0000 0.0000 0.2500 0.2500 0.5000 0.5000 0.7500 0.7500)
SUB_FCC=(0 29 6 11 17 14 21 25 1 2 3 4 5 7 8 9 10 12 13 15 16 18 19 20 22 23 24 26 27 28 30 31)

# ── Append atomic positions to QE input file ──────────────────
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

# ── Generate Fe-Cr QE input ───────────────────────────────────
# Magnetic: nspin=2, smearing mv/degauss=0.02, ecutwfc=60 Ry
# k-mesh: 6x6x6 for BCC 16-atom cell
# electron_maxstep=300: primary fix for high-Cr SCF failures (was default 100)
make_fecr() {
    local f=$1 pref=$2 calc=$3
    local c1x=$4 c1y=$5 c1z=$6
    local c2x=$7 c2y=$8 c2z=$9
    shift 9
    local c3x=$1 c3y=$2 c3z=$3 ncr=$4

    local nt=2
    [ "$ncr" -eq 0 ]  && nt=1
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
    tot_magnetization = 3.16,
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
    conv_thr = 1.0d-7,
    mixing_beta = 0.1,
    mixing_ndim = 16,
    mixing_mode = 'local-TF',
    electron_maxstep = 300
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

# ── Generate Cu-Ni QE input ───────────────────────────────────
# Non-magnetic: nspin omitted, ecutwfc=60 Ry
# k-mesh: 4x4x4 for FCC 32-atom cell
# electron_maxstep=300: applied uniformly for consistency
make_cuni() {
    local f=$1 pref=$2 calc=$3
    local c1x=$4 c1y=$5 c1z=$6
    local c2x=$7 c2y=$8 c2z=$9
    shift 9
    local c3x=$1 c3y=$2 c3z=$3 nni=$4

    local nt=2
    [ "$nni" -eq 0 ]  && nt=1
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
    conv_thr = 1.0d-7,
    mixing_beta = 0.1,
    mixing_ndim = 16,
    mixing_mode = 'local-TF',
    electron_maxstep = 300
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

# ── pw.x execution ────────────────────────────────────────────
# Usage: run_pw <infile> <outfile> <calc_type>
#   calc_type: "vcr" for vc-relax, "scf" for all strain calculations
#
# Skip logic strictly separated by calc_type:
#   vcr: skip ONLY if "End of BFGS Geometry Optimization" found — never on "convergence has been achieved"
#        because that keyword appears after every intermediate SCF inside the BFGS
#        loop and does NOT mean geometry optimization is complete.
#   scf: skip ONLY if "convergence has been achieved"
#
# 1 MPI rank + 8 OMP threads: optimal for single-GPU node.
# --mca flags suppress harmless OpenMPI help-file errors on this instance.
# OMP_NUM_THREADS already exported at top of script.
run_pw() {
    local infile=$1 outfile=$2 calc_type=${3:-scf}

    if [ "$calc_type" = "vcr" ]; then
        if grep -q "End of BFGS Geometry Optimization" "$outfile" 2>/dev/null; then
            if grep -q "convergence has been achieved" "$outfile" 2>/dev/null; then
                echo "    SKIP (bfgs done, sanity SCF converged): $outfile"
            else
                echo "    SKIP (bfgs done, sanity SCF NOT converged — vcr data still usable): $outfile"
            fi
            return 0
        fi
    else
        if grep -q "convergence has been achieved" "$outfile" 2>/dev/null; then
            echo "    SKIP (scf converged): $outfile"
            return 0
        fi
    fi

    # Delete stale/non-converged output before rerunning
    [ -f "$outfile" ] && rm -f "$outfile"
    echo "    >> RUNNING: $(basename $infile)  →  $(basename $outfile)  [$(date +%H:%M:%S)]"
    mpirun --allow-run-as-root --mca btl ^openib --mca pml ob1 --mca coll_hcoll_enable 0 \
        -np 1 "$PW" -npool 1 -input "$infile" > "$outfile" 2>&1

    # Check result — separate criteria per calc type
    if [ "$calc_type" = "vcr" ]; then
        if grep -q "End of BFGS Geometry Optimization" "$outfile" 2>/dev/null; then
            if ! grep -q "convergence has been achieved" "$outfile" 2>/dev/null; then
                echo "    WARNING: bfgs done, sanity SCF NOT converged — vcr data still usable: $outfile"
            fi
            return 0
        fi
    else
        if grep -q "convergence has been achieved" "$outfile" 2>/dev/null; then
            return 0
        fi
    fi

    echo "    WARNING: pw.x did not converge — check $outfile"
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
# All extractors return NaN on failure — never silently return empty or zero.
# Downstream .dat validation checks for NaN and refuses to write corrupt lines.
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
        echo "ERROR: get_s11: no stress block in $1 (SCF not converged?)" >&2; echo "NaN"; return 1
    fi
    echo "$v"
}

get_s33() {
    local v
    v=$(grep -A 3 "total   stress" "$1" 2>/dev/null | tail -1 | awk '{print $6}')
    if [ -z "$v" ]; then
        echo "ERROR: get_s33: no stress block in $1 (SCF not converged?)" >&2; echo "NaN"; return 1
    fi
    echo "$v"
}

get_s12() {
    local v
    v=$(grep -A 3 "total   stress" "$1" 2>/dev/null | tail -3 | head -1 | awk '{print $5}')
    if [ -z "$v" ]; then
        echo "ERROR: get_s12: no stress block in $1 (SCF not converged?)" >&2; echo "NaN"; return 1
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
# Check whether a given EPS value already has a valid (non-NaN, non-zero) entry
# in a .dat file. Returns 0 (valid/skip) or 1 (missing/invalid/rerun needed).
#
# Usage: dat_line_valid <datfile> <eps> <value_col_count>
#   value_col_count: number of data columns after EPS (hydro=1, shear=1, tetra=3)
#
# Validation rules:
#   - Line for EPS must exist
#   - No column must be "NaN"
#   - Not ALL value columns can be exactly 0.0000 (catches silent-zero corruption)
#   - Exception: EPS=0.00 is allowed to have all-zero values only for hydro/shear
#     (zero strain on equilibrium cell can give near-zero stress legitimately)
#     For tetra at EPS=0.00, DS=0.0000 is expected and valid — checked separately.
dat_line_valid() {
    local datfile=$1 eps=$2 ncols=$3
    [ ! -f "$datfile" ] && return 1
    # Find the line matching this eps value
    local line
    line=$(awk -v e="$eps" '$1 == e {print; exit}' "$datfile" 2>/dev/null)
    [ -z "$line" ] && return 1
    # Check for NaN
    echo "$line" | grep -q "NaN" && return 1
    # Check column count: must have EPS + ncols columns
    local total_cols=$(echo "$line" | awk '{print NF}')
    [ "$total_cols" -lt $((ncols + 1)) ] && return 1
    # For tetra (ncols=3): DS is col 4 (S33-S11). At EPS=0.00 DS=0 is expected.
    # Only flag zero if ALL three value columns are zero at non-zero EPS.
    if [ "$ncols" -eq 3 ] && [ "$eps" != "0.00" ]; then
        local s11 s33 ds
        s11=$(echo "$line" | awk '{print $2}')
        s33=$(echo "$line" | awk '{print $3}')
        ds=$(echo "$line" | awk '{print $4}')
        [ "$s11" = "0.0000" ] && [ "$s33" = "0.0000" ] && [ "$ds" = "0.0000" ] && return 1
        [ "$s11" = "0.0000" ] && [ "$s33" = "0.0000" ] && return 1
    fi
    # For hydro/shear (ncols=1): zero at EPS=0.00 is allowed; zero at non-zero EPS is suspect
    if [ "$ncols" -eq 1 ] && [ "$eps" != "0.00" ]; then
        local val
        val=$(echo "$line" | awk '{print $2}')
        [ "$val" = "0.0000" ] && return 1
    fi
    return 0
}

# ── vc-relax convergence gate ─────────────────────────────────
# Returns 0 if vc-relax is converged (safe to proceed with strains).
# Returns 1 if not converged (entire tag must be skipped).
vcr_converged() {
    grep -q "End of BFGS Geometry Optimization" "$1" 2>/dev/null
}

# ── SA_EQ extraction with hard error ─────────────────────────
# Extracts the relaxed lattice parameter from vc-relax output.
# Returns non-empty string on success; prints ERROR and returns 1 on failure.
# NEVER silently falls back to Vegard — caller must handle failure.
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
# SYSTEM 1: Fe-Cr — BCC, 16 atoms, 17 compositions (N_CR = 0..16)
# Per tag: 1 vc-relax + 3 hydrostatic + 3 tetragonal + 3 shear = 10 calcs
# ══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  SYSTEM 1: Fe-Cr (BCC, 16 atoms) — 17 compositions ║"
echo "╚══════════════════════════════════════════════════════╝"

SA_FE=5.6469; SA_CR=5.6600
[ -f "dft_data/fecr_summary.dat" ] || echo "# system n_sol x_sol SA_eq E_tot mag" > dft_data/fecr_summary.dat

for N_CR in $(seq 0 16); do
    X=$(echo "scale=4; ${N_CR} / 16" | bc -l)
    NF=$((16 - N_CR))
    TAG=$(printf "fecr_fe%02dcr%02d" $NF $N_CR)
    SA=$(echo "${SA_FE} * (1.0 - ${X}) + ${SA_CR} * ${X}" | bc -l | xargs printf "%.6f")

    echo ""; echo "── Fe${NF}Cr${N_CR} (x=${X}) ──────────────────"

    # ── vc-relax ──────────────────────────────────────────────
    echo "  [${TAG}] vc-relax..."
    make_fecr "inputs/${TAG}_vcr.in" "${TAG}_vcr" "vc-relax" \
        "$SA" "0.000000" "0.000000" \
        "0.000000" "$SA" "0.000000" \
        "0.000000" "0.000000" "$SA" "$N_CR"
    run_pw "inputs/${TAG}_vcr.in" "outputs/${TAG}_vcr.out" "vcr"; progress

    # ── vc-relax gate ─────────────────────────────────────────
    if ! vcr_converged "outputs/${TAG}_vcr.out"; then
        echo "  [${TAG}] WARNING: vc-relax NOT converged — skipping all strain calcs for this tag"
        echo "  [${TAG}] Check outputs/${TAG}_vcr.out for details"
        continue
    fi

    # ── SA_EQ extraction (hard error — no silent Vegard fallback) ──
    SA_EQ=$(extract_sa_eq "outputs/${TAG}_vcr.out")
    if [ $? -ne 0 ]; then
        echo "  [${TAG}] ERROR: cannot extract SA_EQ — skipping tag"
        continue
    fi
    ET=$(grep "!" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $5}')
    MV=$(grep "total magnetization" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $4}')
    echo "  [${TAG}] SA_EQ=${SA_EQ} Å, E=${ET} Ry, μ=${MV} μB"

    # ── Hydrostatic strain → bulk modulus ─────────────────────
    echo "  [${TAG}] Hydrostatic..."
    H_VALS=()   # will hold "EPS P" pairs for all 3 strains
    H_OK=1      # set to 0 if any strain fails
    for EPS in $STRAINS; do
        # Check if this line is already valid in existing .dat
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
            echo "    ERROR: hydro ε=${EPS} failed — P is NaN/empty"
            H_OK=0
        fi
        H_VALS+=("$EPS $P")
    done
    if [ "$H_OK" -eq 1 ]; then
        printf "%s\n" "${H_VALS[@]}" > "dft_data/${TAG}_hydro.dat"
        echo "  [${TAG}] hydro.dat written (3/3 valid)"
    else
        echo "  [${TAG}] WARNING: hydro.dat NOT written — one or more strains failed"
    fi

    # ── Tetragonal strain → C11-C12 combination ───────────────
    # Cell: a=b=SXY, c=SZ. Volume-conserving: SXY = SA_EQ/sqrt(1+EPS), SZ = SA_EQ*(1+EPS)
    # Extracts S11 (xx), S33 (zz), DS = S33-S11
    # 4-column .dat format: EPS S11 S33 DS
    echo "  [${TAG}] Tetragonal..."
    T_VALS=()
    T_OK=1
    for EPS in $STRAINS; do
        if dat_line_valid "dft_data/${TAG}_tetra.dat" "$EPS" 3; then
            S11=$(awk -v e="$EPS" '$1==e{print $2; exit}' "dft_data/${TAG}_tetra.dat")
            S33=$(awk -v e="$EPS" '$1==e{print $3; exit}' "dft_data/${TAG}_tetra.dat")
            DS=$(awk  -v e="$EPS" '$1==e{print $4; exit}' "dft_data/${TAG}_tetra.dat")
            echo "    tetra ε=${EPS}: SKIP (valid: S11=${S11} S33=${S33} DS=${DS} kbar)"
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
            echo "    ERROR: tetra ε=${EPS} failed — NaN/empty stress"
            T_OK=0
        fi
        T_VALS+=("$EPS $S11 $S33 $DS")
    done
    if [ "$T_OK" -eq 1 ]; then
        printf "%s\n" "${T_VALS[@]}" > "dft_data/${TAG}_tetra.dat"
        echo "  [${TAG}] tetra.dat written (3/3 valid)"
    else
        echo "  [${TAG}] WARNING: tetra.dat NOT written — one or more strains failed"
    fi

    # ── Shear strain → C44 ────────────────────────────────────
    # Off-diagonal shear: a12=a21=SH=SA_EQ*EPS, diagonal = SA_EQ
    # Extracts S12. 2-column .dat format: EPS S12
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
            echo "    ERROR: shear ε=${EPS} failed — NaN/empty stress"
            S_OK=0
        fi
        S_VALS+=("$EPS $S12")
    done
    if [ "$S_OK" -eq 1 ]; then
        printf "%s\n" "${S_VALS[@]}" > "dft_data/${TAG}_shear.dat"
        echo "  [${TAG}] shear.dat written (3/3 valid)"
    else
        echo "  [${TAG}] WARNING: shear.dat NOT written — one or more strains failed"
    fi

    # ── Summary entry (only if all three .dat files were written) ─
    if [ "$H_OK" -eq 1 ] && [ "$T_OK" -eq 1 ] && [ "$S_OK" -eq 1 ]; then
        # Remove any stale summary line for this tag before appending
        grep -v "^FeCr ${N_CR} " dft_data/fecr_summary.dat > dft_data/fecr_summary.dat.tmp \
            && mv dft_data/fecr_summary.dat.tmp dft_data/fecr_summary.dat
        echo "FeCr ${N_CR} ${X} ${SA_EQ} ${ET} ${MV}" >> dft_data/fecr_summary.dat
        echo "  [${TAG}] ✓ Complete"
    else
        echo "  [${TAG}] ✗ Incomplete — rerun when failures are fixed"
    fi

    rm -rf "${PROJ_DIR}/tmp/${TAG}_"*

done

# ══════════════════════════════════════════════════════════════
# SYSTEM 2: Cu-Ni — commented out for the current Vast.ai run.
# Fe-Cr sweep must complete first within the available budget (~$7.95).
# Re-enable by removing the comment markers below when ready to run Cu-Ni.
# All logic is preserved unchanged — no edits needed before re-enabling.
# ══════════════════════════════════════════════════════════════

: <<'CUNI_DISABLED'

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

    echo ""; echo "── Cu${NC}Ni${N_NI} (x=${X}) ──────────────────"

    # ── vc-relax ──────────────────────────────────────────────
    echo "  [${TAG}] vc-relax..."
    make_cuni "inputs/${TAG}_vcr.in" "${TAG}_vcr" "vc-relax" \
        "$SA" "0.000000" "0.000000" \
        "0.000000" "$SA" "0.000000" \
        "0.000000" "0.000000" "$SA" "$N_NI"
    run_pw "inputs/${TAG}_vcr.in" "outputs/${TAG}_vcr.out" "vcr"; progress

    # ── vc-relax gate ─────────────────────────────────────────
    if ! vcr_converged "outputs/${TAG}_vcr.out"; then
        echo "  [${TAG}] WARNING: vc-relax NOT converged — skipping all strain calcs for this tag"
        echo "  [${TAG}] Check outputs/${TAG}_vcr.out for details"
        continue
    fi

    # ── SA_EQ extraction (hard error) ─────────────────────────
    SA_EQ=$(extract_sa_eq "outputs/${TAG}_vcr.out")
    if [ $? -ne 0 ]; then
        echo "  [${TAG}] ERROR: cannot extract SA_EQ — skipping tag"
        continue
    fi
    ET=$(grep "!" "outputs/${TAG}_vcr.out" 2>/dev/null | tail -1 | awk '{print $5}')
    echo "  [${TAG}] SA_EQ=${SA_EQ} Å, E=${ET} Ry"

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
            make_cuni "inputs/${TAG}_h${EPS}.in" "${TAG}_h${EPS}" "scf" \
                "$SD" "0.000000" "0.000000" \
                "0.000000" "$SD" "0.000000" \
                "0.000000" "0.000000" "$SD" "$N_NI"
            run_pw "inputs/${TAG}_h${EPS}.in" "outputs/${TAG}_h${EPS}.out"
            P=$(get_P "outputs/${TAG}_h${EPS}.out")
            echo "    hydro ε=${EPS}: P=${P} kbar"
        fi
        progress
        if [ "$P" = "NaN" ] || [ -z "$P" ]; then
            echo "    ERROR: hydro ε=${EPS} failed"
            H_OK=0
        fi
        H_VALS+=("$EPS $P")
    done
    if [ "$H_OK" -eq 1 ]; then
        printf "%s\n" "${H_VALS[@]}" > "dft_data/${TAG}_hydro.dat"
        echo "  [${TAG}] hydro.dat written (3/3 valid)"
    else
        echo "  [${TAG}] WARNING: hydro.dat NOT written"
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
            make_cuni "inputs/${TAG}_t${EPS}.in" "${TAG}_t${EPS}" "scf" \
                "$SXY" "0.000000" "0.000000" \
                "0.000000" "$SXY" "0.000000" \
                "0.000000" "0.000000" "$SZ" "$N_NI"
            run_pw "inputs/${TAG}_t${EPS}.in" "outputs/${TAG}_t${EPS}.out"
            S11=$(get_s11 "outputs/${TAG}_t${EPS}.out")
            S33=$(get_s33 "outputs/${TAG}_t${EPS}.out")
            DS=$(safe_diff "$S33" "$S11")
            echo "    tetra ε=${EPS}: S11=${S11} S33=${S33} DS=${DS} kbar"
        fi
        progress
        if [ "$S11" = "NaN" ] || [ "$S33" = "NaN" ] || [ "$DS" = "NaN" ] || \
           [ -z "$S11" ] || [ -z "$S33" ] || [ -z "$DS" ]; then
            echo "    ERROR: tetra ε=${EPS} failed"
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
            make_cuni "inputs/${TAG}_s${EPS}.in" "${TAG}_s${EPS}" "scf" \
                "$SA_EQ" "$SH" "0.000000" \
                "$SH" "$SA_EQ" "0.000000" \
                "0.000000" "0.000000" "$SA_EQ" "$N_NI"
            run_pw "inputs/${TAG}_s${EPS}.in" "outputs/${TAG}_s${EPS}.out"
            S12=$(get_s12 "outputs/${TAG}_s${EPS}.out")
            echo "    shear ε=${EPS}: S12=${S12} kbar"
        fi
        progress
        if [ "$S12" = "NaN" ] || [ -z "$S12" ]; then
            echo "    ERROR: shear ε=${EPS} failed"
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
        grep -v "^CuNi ${N_NI} " dft_data/cuni_summary.dat > dft_data/cuni_summary.dat.tmp \
            && mv dft_data/cuni_summary.dat.tmp dft_data/cuni_summary.dat
        echo "CuNi ${N_NI} ${X} ${SA_EQ} ${ET}" >> dft_data/cuni_summary.dat
        echo "  [${TAG}] ✓ Complete"
    else
        echo "  [${TAG}] ✗ Incomplete — rerun when failures are fixed"
    fi

    rm -rf "${PROJ_DIR}/tmp/${TAG}_"*

done

CUNI_DISABLED

# ══════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL DONE — ${TOTAL_MIN} min total"
echo "  Finished: $(date)"
echo "══════════════════════════════════════════════════════"
echo "  Fe-Cr: $(ls dft_data/fecr_*_hydro.dat 2>/dev/null | wc -l) hydro.dat files"
echo ""
echo "  Tags with any failed .dat:"
for f in dft_data/fecr_*_tetra.dat dft_data/fecr_*_hydro.dat dft_data/fecr_*_shear.dat; do
    grep -q "NaN" "$f" 2>/dev/null && echo "    CONTAINS NaN: $f"
done
echo "══════════════════════════════════════════════════════"