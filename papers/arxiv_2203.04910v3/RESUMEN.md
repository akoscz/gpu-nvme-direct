# BaM: A Case for Enabling Fine-grain High Throughput GPU-Orchestrated Access to Storage

## Metadata

- **Authors:** Zaid Qureshi, Vikram Mailthody, Isaac Gelado, Seung Won Min, Amna Masood, Jeongmin Park, Jinjun Xiong, CJ Newburn, Dmitri Vainbrand, I-Hsin Chung, Michael Garland, William Dally, Wen-mei Hwu
- **Affiliations:** NVIDIA, UIUC, University at Buffalo, IBM Research, Stanford
- **Venue:** ASPLOS 2023 (28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2), March 25-29, 2023, Vancouver, BC, Canada
- **DOI:** 10.1145/3575693.3575748
- **arXiv:** 2203.04910v3
- **Code:** https://github.com/ZaidQureshi/bam (BSD-2-Clause)

## Key Findings

### 1. GPU-Initiated Storage Access Eliminates CPU Bottlenecks (Section 1, Introduction; Section 3, Background)

BaM ("Big accelerator Memory") is the first **accelerator-centric** system architecture where GPU threads can create on-demand accesses to data on NVMe SSDs **without relying on the CPU to initiate or trigger the accesses**. The paper demonstrates that CPU-centric approaches (proactive tiling and reactive page faults via UVM) are fundamentally bottlenecked: the UVM page fault handler on the CPU saturates at approximately 500K IOPs, which is only half the peak throughput of a single consumer-grade Samsung 980 Pro SSD. BaM eliminates the CPU from the storage control path, removing costly CPU-GPU synchronization, OS kernel crossings, and software overhead that limits achievable storage throughput.

### 2. Architecture Design: NVMe Queues and Buffers in GPU Memory (Section 4, System and Architecture; Section 5, Prototype)

BaM provisions NVMe I/O submission queues (SQ), completion queues (CQ), and DMA I/O buffers directly in GPU memory (HBM). It uses GPUDirect RDMA APIs to pin and map these structures in GPU memory, enabling SSDs to perform peer-to-peer (P2P) data reads and writes to GPU memory. BaM uses GPUDirect Async to map NVMe SSD doorbell registers into the CUDA address space so GPU threads can ring doorbells on demand. A custom Linux driver creates a character device per NVMe SSD; applications use BaM APIs to open the device. The prototype uses an NVIDIA A100-80GB PCIe GPU with a Supermicro AS-4124GS-TNR server (2x AMD EPYC 7702, 1TB DDR4), with a PCIe expansion chassis (H3 Platform Falcon-4016) providing PCIe switches for low-latency, high-throughput P2P access to up to 10 U.2 NVMe SSDs per drawer.

### 3. High-Throughput I/O Queue Design for Massive GPU Parallelism (Section 4.3, High-Throughput I/O Queues)

BaM addresses the serialization challenge of NVMe doorbell writes with a novel fine-grained synchronization mechanism. It uses an atomic ticket counter, a turn_counter array, and a mark bit-vector per SQ. Threads atomically increment the ticket counter to get a unique queue entry assignment and a turn value. The turn_counter array creates ordered sub-queues per physical queue entry, allowing up to queue-size threads to copy their I/O commands into the SQ in parallel. A single lock-holder thread coalesces doorbell writes for multiple threads by scanning the mark bit-vector and ringing the doorbell once for all consecutive new entries. This design amortizes the expensive PCIe doorbell writes across many concurrent threads. CQ head management uses a similar mechanism. Little's Law analysis shows that sustaining peak PCIe Gen4 x16 bandwidth (~26 GBps) requires queue depths of 561 requests (Optane, 512B) to 16,524 requests (Samsung 980 Pro, 512B), which BaM's massive GPU thread parallelism easily satisfies.

### 4. Software Cache Design with Warp Coalescing (Section 4.4, Software Cache)

BaM features a configurable software cache in GPU memory that minimizes redundant I/O requests to storage. Key design choices: (a) all virtual and physical memory for the cache is pre-allocated at application startup, reducing critical sections to only lock-on-insert/evict; (b) per-cache-line locking prevents multiple threads from issuing duplicate I/O requests for the same data; (c) a clock-based replacement algorithm with a global counter allows concurrent threads to evict unique cache slots in parallel; (d) warp coalescing uses the `__match_any_sync` warp primitive so threads in a warp accessing the same cache line elect a single leader to probe the cache, then broadcast the result via `__shfl_sync`. This is significantly faster than prior serialized coalescing approaches (e.g., ActivePointers/GPUfs using `__ballot_sync` in a loop). The naive cache alone provides 11.9x (BFS) and 12.65x (CC) speedup over no-cache; adding warp coalescing and reference reuse provides an additional 6.07x (BFS) and 11.24x (CC) speedup.

### 5. NVMe Communication Mechanism and GPUDirect RDMA Consistency (Section 5.1, Enable Direct NVMe Access; Section 5.4, Discussion)

GPU threads communicate with NVMe SSDs by: (1) preparing an NVMe I/O command, (2) copying it into a SQ entry in GPU memory, (3) ringing the SSD's doorbell register (mapped into CUDA address space), (4) polling the CQ in GPU memory for the completion entry. The SSD's DMA engine reads the SQ entry over PCIe, accesses its media, then writes data directly into the I/O buffer in GPU memory (for reads) or reads from it (for writes). A critical implementation detail: GPUDirect RDMA over PCIe does not guarantee write ordering from the GPU thread's perspective when a third-party device writes into GPU memory. BaM addresses this with a shared global virtual queue where threads whose first I/O completes race to submit a second "fence" I/O request on behalf of coalesced threads, reducing the consistency overhead from 100% to less than 8%.

### 6. Performance Results: Competitive with DRAM at 21.7x Lower Cost (Section 6, Evaluation)

Key performance results on the prototype system:
- **Raw throughput:** With 10 Intel Optane SSDs, BaM achieves 45.8M random read IOPs (22.9 GBps, 90% of peak PCIe Gen4 x16) and 10.6M random write IOPs at 512B granularity. Only 16K-64K GPU threads needed to saturate a single SSD.
- **Graph analytics (BFS/CC):** With 4 Intel Optane SSDs and 8GB cache, BaM achieves 1.0x (BFS) and 1.49x (CC) end-to-end speedup over the DRAM-only target system (which includes file-loading time), scaling 3.48x (BFS) and 4x (CC) from 1 to 4 SSDs.
- **Data analytics:** BaM achieves up to 5.3x speedup over RAPIDS (state-of-the-art GPU data analytics) on the NYC taxi dataset with 4 Optane SSDs, due to eliminated I/O amplification (RAPIDS suffers 2-6x amplification) and removed CPU orchestration overhead.
- **vs. GDS/ActivePointers:** BaM saturates PCIe bandwidth at 4KB I/O granularity while GDS only reaches 23.6% at 4KB. BaM achieves 17 MIOPs at 512B vs. ActivePointers' 823 KIOPs. Hot cache bandwidth: 430 GBps (BaM) vs. 38.4 GBps (ActivePointers).
- **Cost:** Consumer-grade Samsung 980 Pro SSDs provide 21.8x cost-per-GB advantage over DRAM (even including PCIe expansion hardware), though 3.21x slower for BFS than Optane SSDs.

### 7. Programming Abstraction and Applicability (Section 4.5, Abstraction and Software APIs; Section 7, Discussion)

BaM provides a `bam::array<T>` C++ abstraction with an overloaded subscript operator that transparently handles cache probing, warp coalescing, I/O request generation, and data return. Porting existing GPU applications requires minimal code changes. BaM supports writes via a write-back cache with flush APIs, and can be extended to AMD GPUs (via HIP RoCm), deep learning accelerators, and future CXL-based storage interfaces. The paper notes that performance can be further enhanced with improved GPU-scoped atomics, polling-aware warp scheduling, hardware-accelerated NVMe queues, and higher-throughput SSD protocols.

## Comparison with gpu-nvme-direct (Differences in Approach)

BaM and gpu-nvme-direct both enable GPU threads to directly submit NVMe commands and ring SSD doorbells without CPU involvement. However, they differ fundamentally in hardware platform, P2P capability, and scope:

| Aspect | BaM | gpu-nvme-direct |
|--------|-----|-----------------|
| **GPU class** | Enterprise/datacenter (NVIDIA A100-80GB PCIe) | Consumer (NVIDIA GeForce RTX 3090/4090) |
| **P2P support** | Full bidirectional P2P via GPUDirect RDMA: SSD can read SQ entries from GPU memory AND write data/CQ entries to GPU memory | Consumer GPUs lack full P2P; only **posted writes** from GPU to device BAR are reliable |
| **NVMe queues location** | GPU HBM (mapped via GPUDirect RDMA) | Cannot place queues in GPU memory (SSD cannot read from GPU); must use alternative mechanisms |
| **Data path** | SSD DMA engine reads/writes directly to/from GPU HBM (true P2P) | SSD cannot DMA into GPU memory; must use host memory bounce buffers or GPU-side polling |
| **Doorbell mechanism** | GPU threads write to SSD BAR doorbells mapped via GPUDirect Async + cudaHostRegister | GPU threads write to SSD BAR doorbells mapped via CUDA APIs (posted writes work on consumer GPUs) |
| **Consistency model** | Requires extra I/O request as PCIe fence (GPUDirect RDMA write ordering issue); mitigated with coalesced fencing (~8% overhead) | Simpler consistency since data does not arrive via P2P DMA to GPU memory |
| **Hardware cost** | High (A100 ~$10K+, PCIe expansion chassis, datacenter SSDs) | Low (GeForce ~$500-1500, standard consumer motherboard, consumer NVMe SSDs) |
| **Software stack** | Custom Linux NVMe driver + GPUDirect RDMA/Async APIs + software cache + high-throughput queue library | Lighter-weight: direct BAR mapping of NVMe registers, custom queue management from GPU |
| **Cache** | Sophisticated GPU-side software cache with warp coalescing, clock replacement, per-line locking | Typically no GPU-side software cache (or simpler approaches) |
| **Target use case** | Production datacenter workloads (graph analytics, data analytics, recommender systems) with TB-scale datasets | Research/exploration of GPU-NVMe direct access on commodity hardware; proof-of-concept for consumer platforms |
| **Scalability** | Demonstrated with up to 10 NVMe SSDs via PCIe expansion chassis; linear IOPs scaling | Typically 1-2 NVMe SSDs on consumer motherboard |

The fundamental distinction is that BaM relies on **full P2P DMA** (GPUDirect RDMA) where the SSD can both read submission queue entries from GPU memory and write completion/data directly to GPU memory. This requires enterprise GPUs (Tesla/datacenter SKUs) that expose their memory to PCIe peers. Consumer GeForce GPUs do not support this: while they can perform posted writes to device BARs (sufficient for ringing doorbells), third-party devices cannot reliably DMA into GeForce GPU memory. gpu-nvme-direct works within this constraint by finding alternative paths for data movement, making it accessible on commodity hardware at the cost of some performance and architectural complexity.
