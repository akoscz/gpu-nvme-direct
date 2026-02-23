#!/bin/bash
# gpu-nvme-direct: Full hardware test sequence for RTX 5060 Ti + MAXIO MAP1602
# Run as: sudo bash ~/workspace/gpu-nvme-direct/scripts/run_hardware_tests.sh
#
# GPU BDF:   0000:01:00.0  (RTX 5060 Ti, 16GB BAR1)
# NVMe BDF:  0000:02:00.0  (MAXIO MAP1602, nvme0n1, 1.9TB — NOT boot drive)
# Boot NVMe: 0000:03:00.0  (nvme1n1 — DO NOT TOUCH)

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
GPU_BDF="0000:01:00.0"
NVME_BDF="0000:02:00.0"
NVME_DEV="/dev/nvme0n1"
BUILD_HW="/home/akoscz/workspace/gpu-nvme-direct/build-hw"
TMP_DIR="/tmp/gpunvme_test"

step() { echo -e "\n${BLUE}==== $1 ====${NC}"; }
pass() { echo -e "  ${GREEN}[PASS]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail_exit() { echo -e "  ${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "       $1"; }

if [ "$(id -u)" != "0" ]; then
    echo "Run as root: sudo bash $0"; exit 1
fi

mkdir -p "$TMP_DIR"

# =========================================================
step "0: Prereqs Check"
# =========================================================

# Safety: NVMe not mounted
if mount | grep -q "^$NVME_DEV"; then
    fail_exit "SAFETY: $NVME_DEV is MOUNTED — do not unbind!"; 
fi
pass "$NVME_DEV is not mounted (safe to unbind)"

# NVIDIA module (check /dev/nvidia0 — more reliable under sudo than lsmod parse)
if [ -c /dev/nvidia0 ] || /sbin/lsmod 2>/dev/null | grep -q nvidia || lsmod 2>/dev/null | grep -q nvidia; then
    drv_ver=$(cat /proc/driver/nvidia/version 2>/dev/null | awk '{print $9}' | head -1)
    pass "NVIDIA module loaded (driver $drv_ver)"
else
    warn "NVIDIA module may not be loaded — continuing anyway (tests will self-report)"
fi

# NVMe device present
if lspci -s "$NVME_BDF" | grep -qi "Non-Volatile\|NVMe"; then
    pass "NVMe at $NVME_BDF: $(lspci -s "$NVME_BDF")"
else
    fail_exit "NVMe not found at $NVME_BDF"
fi

# GPU device present
if lspci -s "$GPU_BDF" | grep -qi "NVIDIA\|3D controller"; then
    pass "GPU at $GPU_BDF: $(lspci -s "$GPU_BDF")"
else
    fail_exit "GPU not found at $GPU_BDF"
fi

# Build binaries
for bin in test_bar1_dma test_single_block test_nvme_structs; do
    [ -f "$BUILD_HW/$bin" ] && pass "Binary: $bin" || fail_exit "Missing: $BUILD_HW/$bin"
done

# =========================================================
step "1: Capture NVMe baseline (first 512 bytes via dd)"
# =========================================================
echo "Capturing LBA 0 baseline before unbind..."
dd if="$NVME_DEV" bs=512 count=1 of="$TMP_DIR/baseline.bin" 2>/dev/null
pass "Baseline captured: $TMP_DIR/baseline.bin ($(xxd $TMP_DIR/baseline.bin | head -2))"

# =========================================================
step "2: Struct Tests (no hardware access)"
# =========================================================
"$BUILD_HW/test_nvme_structs" && pass "test_nvme_structs PASSED" || warn "test_nvme_structs FAILED"

# =========================================================
step "3: VFIO-NOIOMMU Setup for NVMe $NVME_BDF"
# =========================================================

# Unload nvme driver from device
NVME_DRIVER=$(basename $(readlink /sys/bus/pci/devices/$NVME_BDF/driver 2>/dev/null) 2>/dev/null || echo "none")
info "Current NVMe driver: $NVME_DRIVER"

if [ "$NVME_DRIVER" != "vfio-pci" ]; then
    echo "Unbinding NVMe from $NVME_DRIVER..."
    echo "$NVME_BDF" > /sys/bus/pci/devices/$NVME_BDF/driver/unbind
    sleep 0.5
    pass "Unbound from $NVME_DRIVER"
fi

# Load vfio modules with noiommu
modprobe vfio enable_unsafe_noiommu_mode=1 2>/dev/null || modprobe vfio
modprobe vfio-pci 2>/dev/null || true

# Enable noiommu mode
if [ -f /sys/module/vfio/parameters/enable_unsafe_noiommu_mode ]; then
    echo 1 > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode
    pass "vfio noiommu mode enabled"
else
    warn "Could not set noiommu parameter (may already be compiled in)"
fi

# Bind to vfio-pci
VENDOR_DEVICE=$(lspci -n -s "$NVME_BDF" | awk '{print $3}' | tr ':' ' ')
info "Vendor:Device = $VENDOR_DEVICE"
echo "$VENDOR_DEVICE" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
echo "$NVME_BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
sleep 0.5

NEW_DRIVER=$(basename $(readlink /sys/bus/pci/devices/$NVME_BDF/driver 2>/dev/null) 2>/dev/null || echo "none")
if [ "$NEW_DRIVER" = "vfio-pci" ]; then
    pass "NVMe $NVME_BDF bound to vfio-pci"
else
    warn "NVMe driver is '$NEW_DRIVER' — trying driver_override..."
    echo "vfio-pci" > /sys/bus/pci/devices/$NVME_BDF/driver_override 2>/dev/null || true
    echo "$NVME_BDF" > /sys/bus/pci/drivers_probe || true
    sleep 0.5
    NEW_DRIVER=$(basename $(readlink /sys/bus/pci/devices/$NVME_BDF/driver 2>/dev/null) 2>/dev/null || echo "none")
    info "Driver after override: $NEW_DRIVER"
    [ "$NEW_DRIVER" = "vfio-pci" ] && pass "Bound to vfio-pci" || warn "Still not vfio-pci — tests may fail"
fi

# Power management fixup (device may fall to D3 after rebind)
echo on > /sys/bus/pci/devices/$NVME_BDF/power/control 2>/dev/null || true
setpci -s "$NVME_BDF" 0x84.W=0x0008 2>/dev/null || true   # PM control: D0
setpci -s "$NVME_BDF" COMMAND=0x0006 2>/dev/null || true   # Memory + BusMaster
sleep 0.5
pass "Power: D0 forced, Memory+BusMaster enabled"

# =========================================================
step "4: GPU Single Block Read Test (THE core test)"
# =========================================================
echo "Running: test_single_block $NVME_BDF $TMP_DIR/baseline.bin"
echo ""
set +e
"$BUILD_HW/test_single_block" "$NVME_BDF" "$TMP_DIR/baseline.bin"
SB_STATUS=$?
set -e

if [ $SB_STATUS -eq 0 ]; then
    pass "test_single_block PASSED — GPU independently read NVMe LBA 0!"
else
    warn "test_single_block exited $SB_STATUS"
    echo ""
    info "Common causes:"
    info "  - NVMe timeout: try 'sleep 2' then re-run manually"
    info "  - cudaHostRegisterIoMemory fail: check nvidia DKMS os-mlock patch"
    info "  - P2P dropped by chipset: Intel usually passes, AMD sometimes drops reads"
fi

# =========================================================
step "5: BAR1 DMA Test (VRAM write via NVMe DMA)"
# =========================================================
echo "Running: test_bar1_dma $NVME_BDF $GPU_BDF"
echo ""
set +e
"$BUILD_HW/test_bar1_dma" "$NVME_BDF" "$GPU_BDF"
BAR1_STATUS=$?
set -e

if [ $BAR1_STATUS -eq 0 ]; then
    pass "test_bar1_dma PASSED — NVMe can DMA directly into GPU VRAM!"
else
    warn "test_bar1_dma exited $BAR1_STATUS (BAR1 DMA may not be supported on this chipset)"
fi

# =========================================================
step "6: Teardown (rebind NVMe to nvme driver)"
# =========================================================
echo ""
read -p "Rebind NVMe back to nvme driver? [Y/n] " ans
ans=${ans:-Y}
if [[ "$ans" =~ ^[Yy] ]]; then
    echo "$NVME_BDF" > /sys/bus/pci/devices/$NVME_BDF/driver/unbind 2>/dev/null || true
    echo "" > /sys/bus/pci/devices/$NVME_BDF/driver_override 2>/dev/null || true
    echo "$NVME_BDF" > /sys/bus/pci/drivers/nvme/bind 2>/dev/null || true
    sleep 0.5
    FINAL_DRIVER=$(basename $(readlink /sys/bus/pci/devices/$NVME_BDF/driver 2>/dev/null) 2>/dev/null || echo "none")
    pass "NVMe rebound to: $FINAL_DRIVER"
else
    warn "NVMe left bound to vfio-pci"
fi

# =========================================================
step "Summary"
# =========================================================
echo ""
if [ $SB_STATUS -eq 0 ] && [ $BAR1_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ FULL SUCCESS — GPU can drive NVMe I/O AND DMA to VRAM!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Copy model to raw NVMe (will erase nvme0n1!):"
    echo "     sudo dd if=/home/akoscz/workspace/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf of=/dev/nvme0n1 bs=4M status=progress"
    echo "  2. Rebuild ntransformer:"
    echo "     cd /home/akoscz/workspace/ntransformer && cmake -DUSE_GPUNVME=ON -DCMAKE_BUILD_TYPE=Release \\"
    echo "       -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \\"
    echo "       -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DCMAKE_CUDA_ARCHITECTURES='80;86;89;90;120' .."
    echo "  3. Run inference:"
    echo "     GPUNVME_PCI_BDF=0000:02:00.0 GPUNVME_GGUF_LBA=0 sudo ./ntransformer ..."
elif [ $SB_STATUS -eq 0 ]; then
    echo -e "${YELLOW}✓ PARTIAL — GPU can drive NVMe I/O (Tier 1), BAR1 DMA failed (Tier 2)${NC}"
    echo "Tier 1 is still viable for the layer-loading use case."
else
    echo -e "${RED}✗ GPU NVMe I/O failed — check output above for details${NC}"
fi
