/*
 * gpu-nvme-direct: GPU-Side SQ Entry Submission
 *
 * Builds NVMe commands in the submission queue from GPU threads and
 * rings the doorbell. All operations use proper memory ordering.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_SQ_SUBMIT_CUH
#define GPUNVME_SQ_SUBMIT_CUH

#include <cstdint>
#include <gpunvme/nvme_regs.h>
#include "queue_state.cuh"
#include "doorbell.cuh"

/*
 * Build and submit an NVMe READ command from the GPU.
 *
 * q:          GPU-side queue state
 * slba:       Starting logical block address
 * nlb_0based: Number of blocks minus 1 (0 = 1 block)
 * prp1:       Physical address of destination buffer
 * prp2:       PRP2 (0 for single-page reads, PRP list addr for multi-page)
 *
 * Returns the command ID assigned to this command.
 *
 * IMPORTANT: Only call from a single thread per queue at a time.
 * Multi-thread queue access requires external synchronization.
 */
__device__ __forceinline__
uint16_t sq_submit_read(gpu_nvme_queue *q,
                        uint64_t slba,
                        uint16_t nlb_0based,
                        uint64_t prp1,
                        uint64_t prp2) {
    uint16_t cid = gpu_nvme_next_cid(q);
    uint16_t slot = gpu_nvme_advance_sq_tail(q);

    /* Build the SQ entry directly in queue memory */
    volatile nvme_sq_entry_t *sqe = &q->sq[slot];

    /* Zero and fill — we write each dword explicitly to avoid
     * a memset on volatile memory which may not compile correctly. */
    /* CDW0: opcode=0x02 (READ), cid */
    uint32_t cdw0 = 0x02u | ((uint32_t)cid << 16);
    sqe->opc  = 0x02;  /* This writes the packed bitfield */
    sqe->fuse = 0;
    sqe->rsvd0 = 0;
    sqe->psdt = 0;
    sqe->cid  = cid;

    /* We need to write CDW0 as a single 32-bit word for atomicity.
     * Use a union reinterpret to write it properly. */
    *(volatile uint32_t *)sqe = cdw0;

    sqe->nsid = q->nsid;
    sqe->cdw2 = 0;
    sqe->cdw3 = 0;
    sqe->mptr = 0;
    sqe->prp1 = prp1;
    sqe->prp2 = prp2;
    sqe->cdw10 = (uint32_t)(slba & 0xFFFFFFFFu);
    sqe->cdw11 = (uint32_t)(slba >> 32);
    sqe->cdw12 = (uint32_t)nlb_0based;
    sqe->cdw13 = 0;
    sqe->cdw14 = 0;
    sqe->cdw15 = 0;

    /* Ensure all SQ entry writes are globally visible before doorbell */
    __threadfence_system();

    /* PCIe write flush: read from BAR0 to ensure SQ entry writes
     * have reached DRAM before the doorbell write reaches the NVMe.
     * Without this, the root complex may deliver the doorbell (posted write
     * to NVMe BAR) before the SQ entry (posted write to host DRAM),
     * causing the NVMe to DMA-read stale SQ data. */
    if (q->pcie_flush_addr) {
        volatile uint32_t __attribute__((unused)) flush = mmio_read32(q->pcie_flush_addr);
    }

    if (q->doorbell_sq) {
        /* Direct GPU doorbell: GPU writes BAR0 MMIO via cudaHostRegisterIoMemory */
        doorbell_write_sq_tail(q->doorbell_sq, q->sq_tail);
        __threadfence_system();
    } else if (q->cpu_db) {
        /* CPU doorbell fallback: hand off to the CPU polling thread.
         *
         * __threadfence_system() above already flushed the SQ entry to DRAM.
         * Now write sq_tail so the CPU thread knows what value to ring, then
         * set sq_pending = 1.  A second __threadfence_system() ensures the
         * CPU observes sq_tail before sq_pending (prevents reorder). */
        q->cpu_db->sq_tail = q->sq_tail;
        __threadfence_system();
        q->cpu_db->sq_pending = 1;
        __threadfence_system();

        /* Spin until CPU thread acks the doorbell write (~1 s GPU-cycle timeout) */
        uint64_t t_db = clock64();
        while (q->cpu_db->sq_pending != 0) {
            if (clock64() - t_db > 3400000000ULL)
                break;  /* CPU thread died — proceed anyway, will likely timeout on CQ */
        }
        __threadfence_system();  /* re-establish ordering after spin */
    }

    return cid;
}

/*
 * Build and submit an NVMe WRITE command from the GPU.
 */
__device__ __forceinline__
uint16_t sq_submit_write(gpu_nvme_queue *q,
                         uint64_t slba,
                         uint16_t nlb_0based,
                         uint64_t prp1,
                         uint64_t prp2) {
    uint16_t cid = gpu_nvme_next_cid(q);
    uint16_t slot = gpu_nvme_advance_sq_tail(q);

    volatile nvme_sq_entry_t *sqe = &q->sq[slot];

    uint32_t cdw0 = 0x01u | ((uint32_t)cid << 16);  /* opcode 0x01 = WRITE */
    *(volatile uint32_t *)sqe = cdw0;

    sqe->nsid = q->nsid;
    sqe->cdw2 = 0;
    sqe->cdw3 = 0;
    sqe->mptr = 0;
    sqe->prp1 = prp1;
    sqe->prp2 = prp2;
    sqe->cdw10 = (uint32_t)(slba & 0xFFFFFFFFu);
    sqe->cdw11 = (uint32_t)(slba >> 32);
    sqe->cdw12 = (uint32_t)nlb_0based;
    sqe->cdw13 = 0;
    sqe->cdw14 = 0;
    sqe->cdw15 = 0;

    __threadfence_system();
    if (q->pcie_flush_addr) {
        volatile uint32_t __attribute__((unused)) flush = mmio_read32(q->pcie_flush_addr);
    }

    if (q->doorbell_sq) {
        doorbell_write_sq_tail(q->doorbell_sq, q->sq_tail);
        __threadfence_system();
    } else if (q->cpu_db) {
        q->cpu_db->sq_tail = q->sq_tail;
        __threadfence_system();
        q->cpu_db->sq_pending = 1;
        __threadfence_system();
        uint64_t t_db = clock64();
        while (q->cpu_db->sq_pending != 0) {
            if (clock64() - t_db > 3400000000ULL)
                break;
        }
        __threadfence_system();
    }

    return cid;
}

#endif /* GPUNVME_SQ_SUBMIT_CUH */
