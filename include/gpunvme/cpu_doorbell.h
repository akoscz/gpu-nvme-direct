/*
 * gpu-nvme-direct: CPU Doorbell Fallback
 *
 * Shared state between the GPU kernel and a CPU polling thread.
 * Used when cudaHostRegisterIoMemory is unavailable or unsafe (e.g.,
 * Blackwell GPUs where registering NVMe BAR0 corrupts GSP firmware).
 *
 * Protocol:
 *   1. GPU fills SQ entry, calls __threadfence_system()
 *   2. GPU writes desired tail/head to sq_tail/cq_head
 *   3. GPU sets sq_pending/cq_pending = 1, calls __threadfence_system()
 *   4. GPU spin-waits until pending == 0
 *   5. CPU thread observes pending, writes BAR0 MMIO doorbell, clears pending
 *
 * This allows the NVMe controller to receive doorbell writes without the
 * GPU ever touching BAR0 MMIO directly.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_CPU_DOORBELL_H
#define GPUNVME_CPU_DOORBELL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Shared doorbell request state (allocated with cudaMallocHost so both
 * CPU and GPU can access it coherently).
 *
 * Cache-line aligned and padded to avoid false sharing between the two
 * independent (sq, cq) control pairs and the shutdown flag.
 */
typedef struct {
    /* SQ tail doorbell request (64-byte cache line) */
    volatile uint32_t sq_tail;      /* GPU writes: desired SQ tail value    */
    volatile uint32_t sq_pending;   /* GPU: set to 1; CPU: clear after ring */
    uint32_t _sq_pad[14];

    /* CQ head doorbell request (64-byte cache line) */
    volatile uint32_t cq_head;      /* GPU writes: desired CQ head value    */
    volatile uint32_t cq_pending;   /* GPU: set to 1; CPU: clear after ring */
    uint32_t _cq_pad[14];

    /* Shutdown flag (64-byte cache line) */
    volatile uint32_t shutdown;     /* CPU thread exits when set to 1       */
    uint32_t _sd_pad[15];
} gpunvme_cpu_db_state_t;

/* Opaque context for the CPU polling thread (host-only) */
typedef struct gpunvme_cpu_db_ctx gpunvme_cpu_db_ctx_t;

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_CPU_DOORBELL_H */
