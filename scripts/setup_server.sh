#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Server Setup — NVIDIA GPU (CUDA 12)
# Installs QE with GPU support, Python, pseudopotentials
# ══════════════════════════════════════════════════════════════
# Usage: bash setup_server.sh

set -e

echo "══════════════════════════════════════════════════════"
echo "  Server Setup: QE with GPU (CUDA 12)"
echo "  $(date)"
echo "══════════════════════════════════════════════════════"

# ── 1. System check ───────────────────────────────────────────
echo ""; echo "── System Info ──────────────────────────────────────"
echo "  OS:     $(uname -s -r)"
echo "  Cores:  $(nproc)"
echo "  Memory: $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo 'unknown')"

if ! command -v nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not found. NVIDIA driver required."; exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "  GPU:    ${GPU_NAME}"
echo "  Driver: ${DRIVER_VER}"
echo "  CUDA:   ${CUDA_VER}"

# Detect compute capability
CUDA_CC=""
case "$GPU_NAME" in
    *V100*) CUDA_CC=70;; *A100*|*A30*) CUDA_CC=80;;
    *H100*|*H200*) CUDA_CC=90;; *RTX*3090*|*RTX*3080*|*A6000*) CUDA_CC=86;;
    *RTX*4090*|*RTX*4080*) CUDA_CC=89;; *RTX*2080*|*RTX*2070*|*T4*) CUDA_CC=75;;
    *P100*) CUDA_CC=60;;
    *) read -p "  Enter compute capability for '${GPU_NAME}' (e.g., 80): " CUDA_CC;;
esac
echo "  Compute: sm_${CUDA_CC}"

# ── 2. NVIDIA HPC SDK ────────────────────────────────────────
echo ""; echo "── NVIDIA HPC SDK ────────────────────────────────────"
NVHPC_READY=false

if command -v nvfortran &> /dev/null; then
    echo "  ✓ nvfortran found"
    NVHPC_READY=true
else
    NVHPC_PATH=""
    for loc in /opt/nvidia/hpc_sdk /usr/local/nvidia/hpc_sdk "$HOME/nvidia/hpc_sdk"; do
        [ -d "$loc" ] && NVHPC_PATH="$loc" && break
    done

    if [ -n "$NVHPC_PATH" ]; then
        NVHPC_VER=$(ls -d ${NVHPC_PATH}/Linux_x86_64/*/compilers/bin 2>/dev/null | sort -V | tail -1 | sed 's|.*/Linux_x86_64/\([^/]*\)/.*|\1|')
        if [ -n "$NVHPC_VER" ]; then
            export PATH=${NVHPC_PATH}/Linux_x86_64/${NVHPC_VER}/compilers/bin:$PATH
            export LD_LIBRARY_PATH=${NVHPC_PATH}/Linux_x86_64/${NVHPC_VER}/compilers/lib:${LD_LIBRARY_PATH:-}
            export LD_LIBRARY_PATH=${NVHPC_PATH}/Linux_x86_64/${NVHPC_VER}/math_libs/lib64:$LD_LIBRARY_PATH
            echo "  ✓ Found HPC SDK ${NVHPC_VER}"
            NVHPC_READY=true
        fi
    fi

    if ! $NVHPC_READY; then
        echo "  Installing NVIDIA HPC SDK (~3 GB, 10-15 min)..."
        if command -v apt-get &> /dev/null; then
            curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
            echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
            sudo apt-get update -qq && sudo apt-get install -y nvhpc-24-7
        elif command -v yum &> /dev/null; then
            sudo yum install -y https://developer.download.nvidia.com/hpc-sdk/nvhpc-2024-24.7-1.x86_64.rpm \
                https://developer.download.nvidia.com/hpc-sdk/nvhpc-24.7-1.x86_64.rpm
        else
            echo "  ERROR: Auto-install not supported. Download from: https://developer.nvidia.com/hpc-sdk-downloads"; exit 1
        fi
        NVHPC_PATH="/opt/nvidia/hpc_sdk"; NVHPC_VER="24.7"
        export PATH=${NVHPC_PATH}/Linux_x86_64/${NVHPC_VER}/compilers/bin:$PATH
        export LD_LIBRARY_PATH=${NVHPC_PATH}/Linux_x86_64/${NVHPC_VER}/compilers/lib:${LD_LIBRARY_PATH:-}
        export LD_LIBRARY_PATH=${NVHPC_PATH}/Linux_x86_64/${NVHPC_VER}/math_libs/lib64:$LD_LIBRARY_PATH
        command -v nvfortran &> /dev/null && echo "  ✓ HPC SDK installed" || { echo "  ERROR: Installation failed"; exit 1; }
    fi
fi

# Find CUDA path
CUDA_PATH=""
for loc in ${NVHPC_PATH:-/opt/nvidia/hpc_sdk}/Linux_x86_64/*/cuda /usr/local/cuda /opt/cuda; do
    if [ -d "$loc" ]; then
        CUDA_PATH=$(ls -d ${loc}/12* 2>/dev/null | sort -V | tail -1)
        [ -z "$CUDA_PATH" ] && CUDA_PATH="$loc"
        break
    fi
done
[ -z "$CUDA_PATH" ] && CUDA_PATH="/usr/local/cuda"
echo "  CUDA path: ${CUDA_PATH}"

# ── 3. Quantum ESPRESSO ──────────────────────────────────────
echo ""; echo "── Quantum ESPRESSO ──────────────────────────────────"
QE_DIR="$HOME/q-e"; PW_BIN=""

if [ -f "$QE_DIR/bin/pw.x" ]; then
    PW_BIN="$QE_DIR/bin/pw.x"
    if $PW_BIN --version 2>&1 | grep -qi "gpu\|cuda"; then
        echo "  ✓ GPU-enabled QE found: $PW_BIN"
    else
        echo "  CPU-only QE found. Rebuilding with GPU..."
        PW_BIN=""
    fi
fi

if [ -z "$PW_BIN" ]; then
    cd $HOME
    QE_VERSION="7.3"
    [ ! -f "q-e-qe-${QE_VERSION}.tar.gz" ] && {
        echo "  Downloading QE v${QE_VERSION}..."
        wget -q "https://github.com/QEF/q-e/archive/refs/tags/qe-${QE_VERSION}.tar.gz" -O "q-e-qe-${QE_VERSION}.tar.gz"
    }
    [ -d "q-e" ] && mv q-e q-e-backup-$(date +%Y%m%d)
    tar xzf "q-e-qe-${QE_VERSION}.tar.gz" && mv "q-e-qe-${QE_VERSION}" q-e
    cd q-e
    echo "  Configuring with CUDA (cc=${CUDA_CC})..."
    ./configure CC=nvc CXX=nvc++ FC=nvfortran \
        --with-cuda=${CUDA_PATH} --with-cuda-runtime=12.0 --with-cuda-cc=${CUDA_CC} \
        --enable-openmp --with-scalapack=no 2>&1 | tail -5
    echo "  Compiling pw.x with $(nproc) cores (15-30 min)..."
    make -j$(nproc) pw 2>&1 | tail -3
    [ -f "bin/pw.x" ] && { PW_BIN="$HOME/q-e/bin/pw.x"; echo "  ✓ QE compiled: $PW_BIN"; } || { echo "  ERROR: Compilation failed"; exit 1; }
    cd $HOME
fi

# ── 4. Python ─────────────────────────────────────────────────
echo ""; echo "── Python ────────────────────────────────────────────"
if command -v python3 &> /dev/null; then
    echo "  ✓ $(python3 --version)"
    python3 -c "import numpy" 2>/dev/null || pip3 install numpy --quiet 2>/dev/null
    python3 -c "import matplotlib" 2>/dev/null || pip3 install matplotlib --quiet 2>/dev/null
else
    echo "  Installing Python3..."
    command -v apt-get &> /dev/null && sudo apt-get install -y -qq python3 python3-pip python3-numpy python3-matplotlib
    command -v yum &> /dev/null && sudo yum install -y python3 python3-pip
fi

# ── 5. Pseudopotentials ───────────────────────────────────────
echo ""; echo "── Pseudopotentials ──────────────────────────────────"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PSEUDO_DIR="${SCRIPT_DIR}/../pseudo"
[ ! -d "$PSEUDO_DIR" ] && PSEUDO_DIR="./pseudo" && mkdir -p "$PSEUDO_DIR"

declare -A PP_FILES=(
    ["Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"]="https://pseudopotentials.quantum-espresso.org/upf_files/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"
    ["Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"]="https://pseudopotentials.quantum-espresso.org/upf_files/Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"
    ["Cu.pbe-dn-rrkjus_psl.1.0.0.UPF"]="https://pseudopotentials.quantum-espresso.org/upf_files/Cu.pbe-dn-rrkjus_psl.1.0.0.UPF"
    ["Ni.pbe-nd-rrkjus.UPF"]="https://pseudopotentials.quantum-espresso.org/upf_files/Ni.pbe-nd-rrkjus.UPF"
)

ALL_OK=true
for PP in "${!PP_FILES[@]}"; do
    if [ -f "${PSEUDO_DIR}/${PP}" ]; then
        echo "  ✓ ${PP}"
    else
        echo "  ✗ ${PP} — downloading..."
        wget -q "${PP_FILES[$PP]}" -O "${PSEUDO_DIR}/${PP}" 2>/dev/null
        [ -f "${PSEUDO_DIR}/${PP}" ] && [ -s "${PSEUDO_DIR}/${PP}" ] && echo "    ✓ Downloaded" || { echo "    ✗ FAILED"; ALL_OK=false; }
    fi
done

# ── 6. GPU test ───────────────────────────────────────────────
echo ""; echo "── GPU Test ──────────────────────────────────────────"
TEST_DIR=$(mktemp -d)
mkdir -p ${TEST_DIR}/pseudo ${TEST_DIR}/tmp
cp ${PSEUDO_DIR}/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF ${TEST_DIR}/pseudo/
cat > ${TEST_DIR}/test.in << 'TESTEOF'
&CONTROL
    calculation = 'scf', prefix = 'gpu_test',
    pseudo_dir = './pseudo/', outdir = './tmp/'
/
&SYSTEM
    ibrav = 3, celldm(1) = 5.42, nat = 1, ntyp = 1,
    ecutwfc = 30.0, ecutrho = 240.0,
    nspin = 2, starting_magnetization(1) = 0.5
/
&ELECTRONS
    conv_thr = 1.0d-6
/
ATOMIC_SPECIES
Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
Fe 0.0 0.0 0.0
K_POINTS automatic
4 4 4 0 0 0
TESTEOF

cd ${TEST_DIR}
timeout 180 $PW_BIN < test.in > test.out 2>&1 || true
if grep -q "JOB DONE" test.out; then
    WALL=$(grep "WALL" test.out | tail -1)
    echo "  ✓ GPU test PASSED"
    [ -n "$WALL" ] && echo "    ${WALL}"
else
    echo "  ✗ GPU test FAILED"
    tail -5 test.out
    ALL_OK=false
fi
cd - > /dev/null; rm -rf ${TEST_DIR}

# ── 7. Summary ────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
if [ -n "$PW_BIN" ] && $ALL_OK; then
    echo "  ✓ System ready"
    echo ""
    echo "  QE:      $PW_BIN"
    echo "  GPU:     ${GPU_NAME} (sm_${CUDA_CC})"
    echo "  CUDA:    ${CUDA_VER}"
    echo "  Pseudos: ${PSEUDO_DIR}/"
    echo ""
    echo "  Run:"
    echo "    nohup bash scripts/run_elastic_grid.sh > elastic_grid.log 2>&1 &"
else
    echo "  ✗ Setup incomplete — check errors above"
fi
echo "══════════════════════════════════════════════════════"
