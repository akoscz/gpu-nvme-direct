# Related Work: GPU-Initiated NVMe I/O on Consumer Hardware

## Overview

This document surveys the landscape of work related to gpu-nvme-direct,
organized into four categories: (A) GPU-initiated storage access systems,
(B) LLM inference on memory-constrained hardware, (C) NVMe/PCIe infrastructure,
and (D) complementary techniques (quantization, KV-cache compression).

gpu-nvme-direct occupies a unique position at the intersection of (A) and (B):
it is the first demonstration of GPU-initiated NVMe I/O on **consumer hardware**
(GeForce RTX 3090, AMD B550), targeting practical LLM inference on commodity
systems costing ~$2000 total.

---

## A. GPU-Initiated Storage Access Systems

### A1. BaM — GPU-Orchestrated Fine-Grained Storage Access
- **Ref:** Qureshi et al., ASPLOS 2023 — `arxiv:2203.04910`
- **Full text:** `papers/arxiv_2203.04910v3/BaM_full_text.txt`
- **Core idea:** First accelerator-centric system where GPU threads create
  on-demand NVMe I/O requests without CPU involvement.
- **Key results:** 45.8M IOPs with 10 SSDs; 1.0x/1.49x BFS/CC speedup over
  DRAM at 21.7x lower cost; saturates PCIe Gen4 x16 at 4KB granularity.
- **Hardware:** NVIDIA A100-80GB + PCIe expansion chassis + Intel Optane SSDs.
- **Mechanism:** NVMe queues in GPU HBM via GPUDirect RDMA; doorbell writes
  via GPUDirect Async; software cache with warp coalescing.
- **Gap vs gpu-nvme-direct:** Requires enterprise GPU (A100) with full P2P
  DMA support. Consumer GeForce cannot expose HBM to PCIe peers for reads.
  gpu-nvme-direct works within this constraint using only posted writes
  (Tier 1 approach).

### A2. GIDS — GPU-Initiated Direct Storage for GNNs
- **Ref:** Park et al., VLDB 2024 — `arxiv:2306.16384`
- **Full text:** `papers/arxiv_2306.16384v2/GIDS_full_text.txt`
- **Core idea:** Extension of BaM to GNN training. GPU threads fetch feature
  vectors directly from NVMe SSDs during graph sampling.
- **Key results:** Up to 582x over DGL mmap; 3.09x over BaM alone on
  terabyte-scale graphs.
- **Contributions relevant to us:**
  - Mathematical model for concurrent storage access to saturate SSD bandwidth
  - Hybrid data placement (NVMe bulk / CPU hot / GPU cached)
  - Window buffering for application-defined eviction
  - Performance data: Optane 1.5M IOPS/11us, Samsung 980 Pro 700K IOPS/324us
    when accessed from GPU
- **Gap:** Same enterprise hardware requirement as BaM.

### A3. SPIN — Seamless P2P DMA between SSDs and GPUs
- **Ref:** Markussen et al., ATC 2019 — `doi:10.1145/3309987`
- **Web:** Found via SearXNG
- **Core idea:** OS-level integration of P2P DMA between NVMe and GPU via
  PCIe switch. Tunnels GPU virtual addresses through NVMe driver.
- **Relevance:** Requires PCIe switch topology. Does not work on consumer
  platforms where GPU and NVMe are on different root ports.

### A4. Phoenix — Refactored I/O Stack for GPU Direct Storage
- **Ref:** ACM 2025 — `doi:10.1145/3712285.3759862`
- **Web:** Found via SearXNG (Google Scholar)
- **Core idea:** Redesigned GDS I/O stack without "phony buffers." Maps GPU
  memory into PCIe BAR space for direct NVMe-GPU data transfer.
- **Relevance:** Shows ongoing work to improve GDS internals; GDS now supports
  P2P DMA for NVMe via upstream kernel PCI P2PDMA (CUDA 12.8+).

### A5. libnvm / ssd-gpu-dma
- **Ref:** Markussen — `github.com/enfiskutensykkel/ssd-gpu-dma`
- **Core idea:** C library for GPU DMA to NVMe using a custom kernel module
  for P2P setup.
- **Gap:** Requires kernel module; designed for Tesla GPUs with P2P support.
  gpu-nvme-direct needs no kernel module for Tier 1.

### A6. US Patent: Peer-to-Peer PCIe Storage Transfers
- **Ref:** US Patent 9304690B2 (2016)
- **Web:** Found via SearXNG (Google Patents)
- **Core idea:** System/method for a device (e.g., GPU) to directly invoke
  NVMe storage commands via PCIe peer-to-peer.
- **Relevance:** Prior art establishing the concept. Our implementation is
  novel in demonstrating it on consumer hardware with documented workarounds.

---

## B. LLM Inference on Memory-Constrained / Consumer Hardware

### B1. FlexGen — High-Throughput LLM Inference with a Single GPU
- **Ref:** Sheng et al., ICML 2023 — `arxiv:2303.06865`
- **Full text:** `papers/arxiv_2303.06865v2/FlexGen_full_text.txt`
- **Core idea:** Offloads weights, activations, and KV cache across GPU/CPU/disk
  using linear programming to find optimal offloading strategy.
- **Key results:** OPT-175B on a single 16GB GPU at 1 tok/s (batch=144).
- **Mechanism:** CPU-mediated disk I/O; all transfers go through CPU.
- **Gap:** CPU is in the critical data path. gpu-nvme-direct could replace
  the disk→CPU→GPU pipeline with direct NVMe→host_pinned→GPU.

### B2. PowerInfer — Fast LLM Serving on Consumer-grade GPU
- **Ref:** Song et al., SOSP 2024 — `arxiv:2312.12456`
- **Full text:** `papers/arxiv_2312.12456v2/PowerInfer_full_text.txt`
- **Core idea:** Exploits activation sparsity (power-law distribution) in LLMs.
  Hot neurons preloaded to GPU; cold neurons computed on CPU.
- **Key results:** Up to 11.69x over llama.cpp; OPT-175B on RTX 4090;
  OPT-30B on RTX 4090 matches 82% of A100 throughput.
- **Relevance:** Same target platform (consumer RTX). Complementary approach:
  PowerInfer reduces what needs to be loaded, gpu-nvme-direct accelerates
  the loading itself. Combined: hot neurons in GPU, cold neurons streamed
  from NVMe without CPU involvement.

### B3. PIPO — Pipelined Offloading on Consumer Devices
- **Ref:** Liu et al., 2025 — `arxiv:2504.03664`
- **Full text:** `papers/arxiv_2504.03664v2/PIPO_full_text.txt`
- **Core idea:** Fine-grained offloading pipeline between CPU and GPU for
  consumer devices. Optimizes data transfer and computation concurrency.
- **Key results:** GPU utilization from <40% to >90%; 3.1x throughput on
  RTX 3060 (6GB).
- **Relevance:** Same target platform. Shows that pipeline design is critical
  for consumer devices. gpu-nvme-direct could provide the NVMe→GPU leg of
  the pipeline without CPU staging.

### B4. HeadInfer — Head-wise KV Cache Offloading
- **Ref:** Luo et al., 2025 — `arxiv:2502.12574`
- **Full text:** `papers/arxiv_2502.12574v1/HeadInfer_full_text.txt`
- **Core idea:** Offloads KV cache to CPU RAM at per-attention-head granularity.
  Only selective heads remain on GPU.
- **Key results:** 4M-token inference with Llama 3-8B on a single RTX 4090
  (24GB). KV cache reduced from 128GB to 1GB on GPU.
- **Relevance:** Demonstrates extreme memory efficiency on consumer GPU.
  For models that don't fit in RAM either, gpu-nvme-direct could extend
  this to NVMe-backed KV cache offloading.

### B5. InstInfer — In-Storage Attention Offloading
- **Ref:** 2024 — `arxiv:2409.04992`
- **Full text:** `papers/arxiv_2409.04992v1/InstInfer_full_text.txt`
- **Core idea:** Offloads attention computation to computational storage
  devices (CSDs). Heterogeneous GPU-CSD inference.
- **Relevance:** Most radical approach — puts compute near storage.
  Complementary to gpu-nvme-direct which keeps compute on GPU but brings
  data faster.

### B6. ServerlessLLM — Multi-Tier Checkpoint Loading
- **Ref:** Fu et al., OSDI 2024 — `arxiv:2401.14351`
- **Full text:** `papers/arxiv_2401.14351v2/ServerlessLLM_full_text.txt`
- **Core idea:** Exploits near-GPU storage hierarchy (DRAM + NVMe) for fast
  LLM checkpoint loading. Loading-optimized format + multi-tier pipeline.
- **Key results:** OPT-30B in 7.5s (28x over Ray Serve); saturates RAID0-NVMe
  at 12 GB/s with 4 CPU cores.
- **Relevance:** Optimized everything around the data path (format, scheduling,
  migration). gpu-nvme-direct targets the remaining bottleneck: the
  NVMe→DRAM→GPU data path itself. Their checkpoint format is already
  GDS-compatible (sequential, chunk-based, word-aligned).

### B7. Petals — Collaborative Inference
- **Ref:** Borzunov et al., 2022 — `arxiv:2209.01188`
- **Core idea:** Distributed inference across consumer GPUs over the network.
  BLOOM-176B at ~1 step/s on consumer GPUs.
- **Relevance:** Alternative approach (network) vs gpu-nvme-direct (local storage).

### B8. NEO — CPU Offloading for Online LLM Inference
- **Ref:** 2024 — `arxiv:2411.01142`
- **Core idea:** Offloads attention compute and KV cache from GPU to CPU.
  Asymmetric GPU-CPU pipelining.
- **Key results:** Up to 7.5x throughput on T4.
- **Relevance:** CPU-mediated. gpu-nvme-direct removes CPU from data path.

### B9. Dovetail — CPU/GPU Heterogeneous Speculative Decoding
- **Ref:** Zhang et al., 2024 — `arxiv:2412.18934`
- **Core idea:** Draft model on GPU, target model on CPU. Speculative decoding
  across heterogeneous hardware.
- **Key results:** 1.79x-10.1x speedup on consumer GPUs for 13B models.
- **Relevance:** Another approach to use both CPU and GPU on consumer hardware.

### B10. MoE CPU-GPU Collaborative Inference
- **Ref:** Huang et al., ASP-DAC 2026 — `arxiv:2512.16473`
- **Core idea:** Expert caching on GPU + CPU computation for cache misses
  in Mixture-of-Experts models on consumer hardware.
- **Relevance:** Similar theme of CPU-GPU collaboration on consumer devices.

---

## C. NVMe / PCIe Infrastructure

### C1. NVIDIA GPUDirect Storage (GDS)
- **Ref:** `docs.nvidia.com/gpudirect-storage/`
- **Status:** As of CUDA 12.8, GDS supports P2P DMA for NVMe devices using
  upstream kernel PCI P2PDMA infrastructure (x86_64).
- **Limitation:** Requires Tesla/datacenter GPUs. Not available on GeForce.
- **Relevance:** The "official" solution that gpu-nvme-direct bypasses for
  consumer hardware.

### C2. SPDK — Storage Performance Development Kit
- **Ref:** `spdk.io/doc/nvme.html`
- **Core idea:** Intel's userspace NVMe driver. Polls from CPU threads.
- **Gap:** CPU-only. Cannot be initiated from GPU kernels.

### C3. Linux P2PDMA Subsystem
- **Ref:** LWN.net article on 2023 LSFMM Summit; kernel documentation
- **Status:** Growing kernel support for device-to-device DMA. Used by GDS
  internally.
- **Relevance:** gpu-nvme-direct's Tier 1 approach sidesteps the need for
  full P2PDMA (which requires IOMMU cooperation on AMD platforms).

### C4. CXL Memory Pools Replacing PCIe Switches
- **Ref:** Zhong et al., 2025 — `arxiv:2503.23611`
- **Core idea:** CXL memory pools can serve as I/O buffers for PCIe devices,
  enabling software-defined device pooling without hardware PCIe switches.
- **Relevance:** Future direction. CXL could provide a shared memory region
  accessible by both GPU and NVMe, simplifying the data path.

### C5. I/O Characterization of LLM Offloading to NVMe
- **Ref:** CHEOPS 2025 — `doi:10.1145/3719330.3721230`
- **Key findings:**
  - LLM model offloading dominated by **128 KiB sequential reads**
  - libaio-based tensor offloading delivers higher bandwidth than POSIX
  - Model offloading **does not saturate NVMe SSDs** (key opportunity!)
  - KV cache offloading: reads at 2.0 GiB/s, writes at only 11.0 MiB/s
- **Relevance:** Directly validates gpu-nvme-direct's premise: current LLM
  offloading underutilizes NVMe bandwidth because CPU is the bottleneck.

---

## D. Complementary Techniques

### D1. NanoQuant — Sub-1-Bit Quantization
- **Ref:** Chong et al., 2026 — `arxiv:2602.06694`
- **Full text:** `papers/arxiv_2602.06694v1/NanoQuant_full_text.txt`
- **Core idea:** Compresses LLMs to binary/sub-1-bit levels via low-rank
  binary factorization. Llama2-70B compressed 25.8x → fits in 8GB GPU.
- **Relevance:** Reduces model size dramatically, making NVMe streaming
  even more viable. Combined with gpu-nvme-direct: a 70B model compressed
  to ~3GB could be streamed from NVMe at 2.7 GB/s in ~1 second per full
  model pass.

### D2. KVTC — KV Cache Transform Coding
- **Ref:** Staniszewski et al., 2025 — `arxiv:2511.01815`
- **Core idea:** Compresses KV caches up to 20-40x using PCA + adaptive
  quantization + entropy coding.
- **Relevance:** Reduces KV cache storage on NVMe when offloading. Combined
  with gpu-nvme-direct, compressed KV caches could be streamed efficiently.

### D3. MemAscend — System Memory Optimization for SSD Offloading
- **Ref:** 2025 — `arxiv:2505.23254`
- **Core idea:** Tackles system memory bottlenecks in ZeRO-Infinity SSD
  offloading: fragmentation, pinned buffer allocation, peak CPU overhead.
- **Key results:** 55.7% reduction in peak system memory consumption.
- **Relevance:** Addresses the host-side bottlenecks that gpu-nvme-direct
  also encounters when using host pinned memory as staging.

---

## Positioning Map

```
                    Enterprise HW              Consumer HW
                    ─────────────              ───────────
GPU-initiated       BaM (ASPLOS'23)            gpu-nvme-direct (ours)
storage I/O         GIDS (VLDB'24)
                    SPIN (ATC'19)
                    GDS (NVIDIA)

CPU-mediated        ServerlessLLM (OSDI'24)    FlexGen (ICML'23)
storage I/O         MemAscend (2025)           PIPO (2025)
                    CHEOPS I/O study (2025)    Dovetail (2024)

Compute             NEO (2024)                 PowerInfer (SOSP'24)
offloading          HeadInfer (2025)           Petals (2022)
(CPU/GPU hybrid)    InstInfer (2024)           MoE collab (ASP-DAC'26)

Model compression   ---                        NanoQuant (2026)
                                               KVTC (2025)
```

**gpu-nvme-direct is the only entry in the "GPU-initiated + Consumer HW" cell.**

---

## Key Gaps in Existing Literature

1. **No GPU-initiated NVMe I/O on consumer hardware**: All prior work (BaM,
   GIDS, SPIN, GDS) requires enterprise GPUs with full P2P DMA support.

2. **No characterization of PCIe posted writes from consumer GPUs**: The
   Tier 1 approach (GPU writes doorbells, NVMe DMAs to host pinned memory)
   is undocumented in literature.

3. **No AMD consumer platform P2P analysis**: Detailed documentation of what
   works (posted writes) and what doesn't (non-posted reads) on AMD B550/Ryzen.

4. **No integration of GPU-initiated NVMe with LLM inference on consumer HW**:
   The streaming-layer-from-NVMe approach for 70B models on 24GB VRAM is novel.

5. **PTX MMIO instructions for NVMe**: Using `st.relaxed.mmio.sys` /
   `ld.relaxed.mmio.sys` instead of volatile C++ for correct PCIe transactions
   is undocumented.

6. **cudaHostRegisterIoMemory on GeForce**: Requires driver patch on kernel
   6.12+ (follow_pfn removal). Not documented anywhere.

---

## Papers Downloaded (Full Text Available)

| Paper | arXiv ID | File |
|-------|----------|------|
| BaM | 2203.04910 | `papers/arxiv_2203.04910v3/BaM_full_text.txt` |
| GIDS | 2306.16384 | `papers/arxiv_2306.16384v2/GIDS_full_text.txt` |
| ServerlessLLM | 2401.14351 | `papers/arxiv_2401.14351v2/ServerlessLLM_full_text.txt` |
| FlexGen | 2303.06865 | `papers/arxiv_2303.06865v2/FlexGen_full_text.txt` |
| PowerInfer | 2312.12456 | `papers/arxiv_2312.12456v2/PowerInfer_full_text.txt` |
| PIPO | 2504.03664 | `papers/arxiv_2504.03664v2/PIPO_full_text.txt` |
| InstInfer | 2409.04992 | `papers/arxiv_2409.04992v1/InstInfer_full_text.txt` |
| HeadInfer | 2502.12574 | `papers/arxiv_2502.12574v1/HeadInfer_full_text.txt` |
| NanoQuant | 2602.06694 | `papers/arxiv_2602.06694v1/NanoQuant_full_text.txt` |

---

## Suggested Citation Groups for Paper

**GPU-initiated storage (our lineage):**
BaM [ASPLOS'23], GIDS [VLDB'24], SPIN [ATC'19], GDS [NVIDIA], ssd-gpu-dma

**LLM inference on constrained hardware (our application):**
FlexGen [ICML'23], PowerInfer [SOSP'24], PIPO [2025], HeadInfer [2025],
ServerlessLLM [OSDI'24], Petals [2022], NEO [2024], Dovetail [2024]

**NVMe/PCIe infrastructure (our mechanism):**
SPDK [Intel], Linux P2PDMA, CHEOPS I/O study [2025], CXL pools [2025]

**Complementary techniques (future integration):**
NanoQuant [2026], KVTC [2025], MemAscend [2025]
