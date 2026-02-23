/*
 * gpu-nvme-direct: CPU Doorbell Polling Thread
 *
 * Provides a safe alternative to cudaHostRegisterIoMemory for NVMe BAR0
 * doorbell writes.  On NVIDIA Blackwell (sm_120) and other architectures
 * where registering NVMe MMIO with CUDA corrupts GSP firmware, this thread
 * takes over the doorbell responsibility:
 *
 *   - The GPU kernel builds SQ entries in host pinned memory (unchanged).
 *   - Instead of directly writing BAR0 MMIO, the GPU sets a flag in a
 *     small pinned control struct (gpunvme_cpu_db_state_t).
 *   - This thread spins on that struct and performs the actual BAR0 MMIO
 *     write using normal CPU stores, then clears the flag.
 *
 * Throughput impact is small: NVMe DMA time per command dominates latency.
 * The added cost is one CPU spin-loop iteration (< 1 µs at ~5 GHz) per
 * doorbell, which is negligible against the ~94 µs per 512 KB MDTS command.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <gpunvme/cpu_doorbell.h>

/* Internal context owned by the thread */
struct gpunvme_cpu_db_ctx {
    gpunvme_cpu_db_state_t *db;         /* shared state (pinned host mem)   */
    volatile uint32_t      *bar0;       /* CPU-mapped NVMe BAR0             */
    uint32_t                sq_db_off;  /* SQ tail doorbell offset in BAR0  */
    uint32_t                cq_db_off;  /* CQ head doorbell offset in BAR0  */
    pthread_t               thread;
};

/* ---------------------------------------------------------------------------
 * CPU polling thread
 * --------------------------------------------------------------------------*/

static void *cpu_doorbell_thread(void *arg) {
    struct gpunvme_cpu_db_ctx *ctx = (struct gpunvme_cpu_db_ctx *)arg;
    gpunvme_cpu_db_state_t   *db  = ctx->db;
    volatile uint32_t        *bar0 = ctx->bar0;
    volatile uint32_t        *sq_doorbell =
        (volatile uint32_t *)((uint8_t *)bar0 + ctx->sq_db_off);
    volatile uint32_t        *cq_doorbell =
        (volatile uint32_t *)((uint8_t *)bar0 + ctx->cq_db_off);

    fprintf(stderr, "cpu_doorbell: thread started (sq_db_off=0x%x, cq_db_off=0x%x)\n",
            ctx->sq_db_off, ctx->cq_db_off);

    while (!db->shutdown) {
        /*
         * SQ tail doorbell request
         *
         * Memory order:
         *   GPU:  __threadfence_system() → write sq_tail → __threadfence_system()
         *         → write sq_pending=1  → __threadfence_system()
         *   CPU:  read sq_pending (cache-coherent, sees GPU's write)
         *         → read sq_tail (already visible per above)
         *         → __sync_synchronize() (x86 TSO: sfence before MMIO write)
         *         → write BAR0 MMIO doorbell
         *         → write sq_pending=0
         */
        if (db->sq_pending) {
            uint32_t tail = db->sq_tail;
            /* Full barrier: ensure we've loaded sq_tail before the MMIO write */
            __sync_synchronize();
            *sq_doorbell = tail;
            /* Ensure the MMIO write is flushed before signalling the GPU */
            __sync_synchronize();
            db->sq_pending = 0;
            /* Release barrier: GPU spin-loop must see this clear */
            __sync_synchronize();
        }

        /*
         * CQ head doorbell request
         */
        if (db->cq_pending) {
            uint32_t head = db->cq_head;
            __sync_synchronize();
            *cq_doorbell = head;
            __sync_synchronize();
            db->cq_pending = 0;
            __sync_synchronize();
        }

        /* No sleep / yield — latency matters more than CPU burn here.
         * The GPU kernel blocks until we respond; keeping this thread
         * pinned gives the lowest possible doorbell round-trip time.
         * The thread is only alive during active I/O (init→destroy). */
    }

    fprintf(stderr, "cpu_doorbell: thread exiting\n");
    return NULL;
}

/* ---------------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------------*/

/**
 * gpunvme_cpu_db_start - Allocate context and launch the polling thread.
 *
 * @db:        Shared state (allocated with cudaMallocHost by caller)
 * @bar0:      CPU-accessible BAR0 mmap
 * @sq_db_off: SQ tail doorbell offset in BAR0
 * @cq_db_off: CQ head doorbell offset in BAR0
 * @ctx_out:   Receives the allocated context (caller must free via _stop)
 *
 * Returns 0 on success, -1 on failure.
 */
int gpunvme_cpu_db_start(gpunvme_cpu_db_state_t *db,
                          volatile void          *bar0,
                          uint32_t                sq_db_off,
                          uint32_t                cq_db_off,
                          gpunvme_cpu_db_ctx_t  **ctx_out) {
    if (!db || !bar0 || !ctx_out)
        return -1;

    struct gpunvme_cpu_db_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return -1;

    ctx->db       = db;
    ctx->bar0     = (volatile uint32_t *)bar0;
    ctx->sq_db_off = sq_db_off;
    ctx->cq_db_off = cq_db_off;

    /* Ensure db is zeroed and shutdown is clear before thread starts */
    db->sq_tail    = 0;
    db->sq_pending = 0;
    db->cq_head    = 0;
    db->cq_pending = 0;
    db->shutdown   = 0;
    __sync_synchronize();

    if (pthread_create(&ctx->thread, NULL, cpu_doorbell_thread, ctx) != 0) {
        fprintf(stderr, "cpu_doorbell: pthread_create failed\n");
        free(ctx);
        return -1;
    }

    *ctx_out = ctx;
    return 0;
}

/**
 * gpunvme_cpu_db_stop - Signal shutdown and join the thread, then free ctx.
 *
 * Safe to call with NULL ctx.
 */
void gpunvme_cpu_db_stop(gpunvme_cpu_db_ctx_t *ctx) {
    if (!ctx) return;

    /* Signal the thread to exit */
    ctx->db->shutdown = 1;
    __sync_synchronize();

    pthread_join(ctx->thread, NULL);
    free(ctx);
}
