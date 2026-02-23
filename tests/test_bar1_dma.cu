/*
 * test_bar1_dma: NVMe DMA directly to GPU VRAM via BAR1 (Tier 2)
 *
 * Tests:
 * 1. BAR1 init (read GPU BAR1 physical base)
 * 2. BAR1 resolve (find VRAM physical offset via pattern scan)
 * 3. NVMe read to VRAM (DMA through BAR1, verify data)
 *
 * Usage: sudo ./test_bar1_dma <nvme_bdf> [gpu_bdf]
 *   nvme_bdf: NVMe PCI address (e.g., 0000:01:00.0)
 *   gpu_bdf:  GPU PCI address (default: 0000:0a:00.0)
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gpunvme/layer_loader.h>

#define STATIC_BAR1_OFFSET 0x0ULL  /* 0 = scan from BAR1 start (auto-detect, recommended) */

/* GPU kernel to zero a buffer */
__global__ void gpu_memset_zero(uint8_t *buf, size_t n) {
    size_t i = (size_t)threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    if (i < n) buf[i] = 0;
}

/* Read a uint64_t bypassing ALL GPU caches (L1 + L2).
 * PTX ld.volatile.global reads directly from DRAM — essential for verifying
 * externally DMA'd data that bypassed GPU cache hierarchy (e.g. NVMe BAR1 writes). */
__device__ __forceinline__ uint64_t ldcv_u64(const uint64_t *ptr) {
    uint64_t val;
    asm volatile("ld.volatile.global.u64 %0, [%1];"
                 : "=l"(val)
                 : "l"((const unsigned long long *)ptr)
                 : "memory");
    return val;
}

/* GPU kernel to read words from VRAM, bypassing all caches */
__global__ void gpu_read_words(const uint64_t *buf, uint64_t *out, int count) {
    int i = threadIdx.x;
    if (i < count) out[i] = ldcv_u64(buf + i);
}

/* GPU kernel: copy N bytes from VRAM to host-mapped buffer using volatile loads.
 * Each thread handles one uint64_t. Use for DMA-write verification when L2 may be stale. */
__global__ void gpu_extract_nocache(const uint64_t *src, uint64_t *dst, int n_words) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n_words) dst[i] = ldcv_u64(src + i);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <nvme_bdf> [gpu_bdf]\n", argv[0]);
        return 1;
    }

    const char *nvme_bdf = argv[1];
    const char *gpu_bdf = (argc >= 3) ? argv[2] : "0000:0a:00.0";
    size_t test_size_mb = (argc >= 4) ? (size_t)atoi(argv[3]) : 4;

    printf("=== NVMe DMA-to-VRAM via BAR1 (Tier 2) ===\n");
    printf("NVMe: %s, GPU: %s, Size: %zu MB\n\n", nvme_bdf, gpu_bdf, test_size_mb);

    /* Test 1: Initialize layer loader + BAR1 */
    printf("--- Test 1: BAR1 Init ---\n");

    gpunvme_layer_loader_t loader;
    size_t test_size = test_size_mb * 1024 * 1024;

    gpunvme_err_t err = gpunvme_layer_loader_init(&loader, nvme_bdf, test_size, 32);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "FAIL: layer_loader_init: %d\n", err);
        return 1;
    }

    err = gpunvme_bar1_init(&loader, gpu_bdf, STATIC_BAR1_OFFSET);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "FAIL: bar1_init: %d\n", err);
        gpunvme_layer_loader_destroy(&loader);
        return 1;
    }
    printf("PASS: BAR1 init OK (phys=0x%llx)\n\n",
           (unsigned long long)loader.gpu_bar1_phys);

    /* Test 2: Allocate VRAM and resolve BAR1 address */
    printf("--- Test 2: BAR1 Resolve ---\n");

    void *vram_buf = nullptr;
    cudaError_t cerr = cudaMalloc(&vram_buf, test_size);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaMalloc: %s\n", cudaGetErrorString(cerr));
        gpunvme_layer_loader_destroy(&loader);
        return 1;
    }
    printf("VRAM buffer: %p (%zu bytes)\n", vram_buf, test_size);

    uint64_t bar1_phys = 0;
    err = gpunvme_bar1_resolve(&loader, vram_buf, test_size, &bar1_phys);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "FAIL: bar1_resolve: %d\n", err);
        cudaFree(vram_buf);
        gpunvme_layer_loader_destroy(&loader);
        return 1;
    }
    printf("PASS: BAR1 resolve OK → phys=0x%llx\n\n",
           (unsigned long long)bar1_phys);

    /* Test 3: NVMe read directly to VRAM via BAR1 */
    printf("--- Test 3: NVMe DMA to VRAM ---\n");

    /* Zero the VRAM buffer first */
    int blocks = (test_size + 255) / 256;
    gpu_memset_zero<<<blocks, 256>>>((uint8_t *)vram_buf, test_size);
    cudaDeviceSynchronize();

    /* Read from LBA 0 (whatever is on the NVMe) */
    err = gpunvme_load_layer_vram(&loader, 0, test_size, bar1_phys);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "FAIL: load_layer_vram: %d\n", err);
        cudaFree(vram_buf);
        gpunvme_layer_loader_destroy(&loader);
        return 1;
    }

    /* Read first 8 words from VRAM to verify non-zero data arrived */
    uint64_t *h_check;
    cudaMallocHost(&h_check, 64);
    gpu_read_words<<<1, 8>>>((const uint64_t *)vram_buf, h_check, 8);
    cudaDeviceSynchronize();

    int nonzero = 0;
    printf("VRAM content (first 64 bytes):\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] 0x%016llx\n", i, (unsigned long long)h_check[i]);
        if (h_check[i] != 0) nonzero++;
    }

    if (nonzero >= 4) {
        printf("PASS: NVMe DMA to VRAM via BAR1 — %d/8 words non-zero\n\n", nonzero);
    } else {
        printf("FAIL: NVMe DMA to VRAM — only %d/8 words non-zero (data didn't arrive?)\n\n", nonzero);
    }

    /* Test 4: Compare with Tier 1 (host pinned) read */
    printf("--- Test 4: Verify vs Tier 1 ---\n");

    void *host_buf;
    cudaMallocHost(&host_buf, test_size);
    err = gpunvme_load_layer(&loader, 0, test_size, host_buf);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "FAIL: Tier 1 load_layer: %d\n", err);
    } else {
        /* Copy VRAM to host for comparison using cache-bypassing reads.
         * cudaMemcpy(D2H) uses L2 which may have stale zeros from gpu_memset_zero.
         * gpu_extract_nocache uses ld.volatile.global to read past all caches. */
        void *vram_copy;
        cudaMallocHost(&vram_copy, test_size);
        {
            uint64_t *vram_copy_dev = nullptr;
            cudaMalloc(&vram_copy_dev, test_size);
            int n_words = (int)(test_size / sizeof(uint64_t));
            int threads = 256;
            int blocks = (n_words + threads - 1) / threads;
            gpu_extract_nocache<<<blocks, threads>>>((const uint64_t*)vram_buf, vram_copy_dev, n_words);
            cudaDeviceSynchronize();
            cudaMemcpy(vram_copy, vram_copy_dev, test_size, cudaMemcpyDeviceToHost);
            cudaFree(vram_copy_dev);
        }

        int mismatches = 0;
        const uint8_t *a = (const uint8_t *)host_buf;
        const uint8_t *b = (const uint8_t *)vram_copy;
        for (size_t i = 0; i < test_size; i++) {
            if (a[i] != b[i]) {
                if (mismatches < 5) {
                    printf("  Mismatch at byte %zu: host=0x%02x vram=0x%02x\n", i, a[i], b[i]);
                }
                mismatches++;
            }
        }

        if (mismatches == 0) {
            printf("PASS: Tier 1 vs Tier 2 — %zu bytes match perfectly!\n", test_size);
        } else {
            printf("FAIL: %d mismatches out of %zu bytes\n", mismatches, test_size);
        }

        cudaFreeHost(vram_copy);
    }

    /* Cleanup */
    cudaFreeHost(h_check);
    cudaFreeHost(host_buf);
    cudaFree(vram_buf);
    gpunvme_layer_loader_destroy(&loader);

    printf("\n=== Done ===\n");
    return 0;
}
