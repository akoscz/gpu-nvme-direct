# BAR1 VRAM Address Mapping on Blackwell (RTX 5060 Ti / GB206)

## Summary

Tier 2 NVMe→VRAM DMA via BAR1 requires knowing the PCIe-visible physical address
of a VRAM allocation. Two approaches were investigated on Blackwell (RTX 5060 Ti,
driver 590.48.01, kernel 6.19.3):

## What Works

### cuPointerGetAttribute(PHYSICAL_ADDRESS)
Returns the VRAM-relative physical address. Confirmed `0x400000` for a 4MB
`cudaMalloc` allocation — this is a VRAM offset, NOT yet the BAR1 PCIe address.

### BAR1 base address
From `/sys/bus/pci/devices/0000:01:00.0/resource` line 1:
```
BAR1 phys: 0x4000000000–0x43ffffffff (16 GB, ResizableBAR fully enabled)
```

## What Doesn't Work

### CPU reads of resource1
`mmap(resource1)` succeeds but returns all zeros regardless of offset.
The NVIDIA open kernel module blocks CPU reads of VRAM through BAR1 on Blackwell.
Pattern-scan approach (`bar1_resolve` original design) is NOT viable.

### Simple BAR1 offset calculation
`pcIe_addr = BAR1_phys + cuPointerGetAttribute(PHYSICAL_ADDRESS)`
**Does not work.** The NVMe DMA at this address does not reach the VRAM location
that the GPU VA maps to. L2 eviction and fresh allocation tests confirm this.

The NVIDIA driver reserves an architecture-specific offset at the start of BAR1.
VRAM is only BAR1-accessible starting at `BAR1[VRAM_DRIVER_OFFSET]`, where
`VRAM_DRIVER_OFFSET` differs per GPU family:
- RTX 3090 (GA102): `0x20000000` (512 MB) — hardcoded in original code
- RTX 5060 Ti (GB206): unknown, cannot determine without `nvidia_p2p_get_pages()`

## The Fix Applied (this branch)

1. `bar1_resolve`: Use `cuPointerGetAttribute` as fast path (correct VRAM-relative
   offset extracted); fall back to pattern scan with PAGE_SIZE mmap windows instead
   of 256MB (large mmap of resource1_wc returns EINVAL)
2. `bar1_init`: Read actual BAR1 size from resource file (was hardcoded to 24GB)
3. `bar1_resolve`: Scan from offset 0 (not 512MB static hint)
4. `test_bar1_dma`: Use `ld.volatile.global` for cache-bypassing verification reads

## Required for Full Tier 2 Support

To get the correct BAR1-accessible physical address for `cudaMalloc` buffers,
the NVIDIA `nvidia_p2p` kernel module API is required:

```c
nvidia_p2p_page_table_t *page_table;
nvidia_p2p_get_pages(0, 0, (uint64_t)vram_ptr, size, &page_table, callback, NULL);
// page_table->pages[i]->physical_address = correct PCIe physical address
```

This API establishes the BAR1 mapping AND returns the correct PCIe-visible
physical address, handling all driver-internal offsets.

Alternatively, `cuFile` (GPUDirect Storage) handles this transparently.

## Test Results on RTX 5060 Ti

| Test | Result |
|------|--------|
| test_nvme_structs 26/26 | ✅ PASS |
| test_single_block (Tier 1, GPU→NVMe) | ✅ PASS — 5500+ MB/s, data verified |
| test_bar1_dma BAR1 init | ✅ PASS |
| test_bar1_dma bar1_resolve (cuPointerGetAttribute) | ✅ PASS (offset extracted) |
| test_bar1_dma NVMe→VRAM DMA data arrival | ❌ FAIL — wrong PCIe address |
| L2 eviction test (Experiment 2) | ❌ Still zeros after L2 flush |
| Fresh alloc test (Experiment 3) | ❌ GPU-written pattern survives DMA |

**Tier 1 is production-ready. Tier 2 requires nvidia_p2p kernel module.**
