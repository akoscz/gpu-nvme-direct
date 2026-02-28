# GIDS: GNN-Informed Direct Storage Access for Training Graph Neural Networks on Large Graphs

## Paper Metadata

- **Authors:** Jeongmin Brian Park (UIUC), Vikram Sharma Mailthody (NVIDIA), Zaid Qureshi (NVIDIA), Wen-mei Hwu (NVIDIA/UIUC)
- **Venue:** PVLDB (Proceedings of the VLDB Endowment), 2024
- **arXiv:** [2306.16384v2](https://arxiv.org/abs/2306.16384)

## Key Findings

### 1. CPU-initiated data preparation is the bottleneck in large-scale GNN training (Section 2.3, Figure 3)

Profiling shows that CPUs can only generate 4.1 million feature vector requests per second (even with 16 threads), while the GPU training kernels consume features at 29 million requests/second. The GPU, by contrast, can generate 77 million feature requests/second. When using memory-mapped files for datasets exceeding CPU memory, OS page-fault overhead makes the sampling and feature aggregation stages dominate total training time by orders of magnitude -- for terabyte-scale graphs like IGB-Full and IGBH-Full, the GPU training stage becomes "barely visible" in execution time profiles.

### 2. GPU-initiated direct storage access eliminates CPU-GPU synchronization overhead (Sections 3.1, 2.4)

GIDS leverages the BaM (Big-data Accelerator for Memory) system to allow GPU threads to directly fetch feature vectors from NVMe SSDs without going through the CPU or OS page-fault handler. The massive GPU thread-level parallelism (thousands of concurrent threads) naturally overlaps the long storage access latencies. This bypasses the CPU software stack entirely, eliminating the page-fault handling overhead that cripples memory-mapped file approaches.

### 3. Dynamic storage access accumulator ensures peak SSD bandwidth utilization (Sections 3.2, 5.3)

A mathematical model estimates the required number of concurrent storage accesses to reach peak SSD bandwidth: N_access = IOP_achieved * (T_i + T_s + T_t) * N_ssd. The accumulator exploits the logical independence of graph sampling from model training to merge consecutive iterations, dynamically maintaining enough overlapping storage requests. With the accumulator on two Intel Optane SSDs, bandwidth improves from 7.6 GB/s to 9.8 GB/s at batch size 32. When combined with the constant CPU buffer and window buffering, the accumulator delivers 1.31x to 1.95x speedup depending on batch size.

### 4. Hybrid data placement across GPU memory, CPU memory, and NVMe storage (Sections 3.1, 3.3, 3.5)

GIDS uses a tiered approach: (a) graph structure data (typically ~5% of total dataset) is pinned in CPU memory and accessed via UVA zero-copy -- this avoids I/O amplification since structure data has fine-grained 4-8 byte access patterns; (b) a configurable "constant CPU buffer" pins hot node features (selected by weighted reverse PageRank) in CPU memory to amplify effective bandwidth beyond SSD limits; (c) node feature data (the bulk, 68-96% of dataset) resides on NVMe SSDs. With 20% of features in the CPU buffer using reverse PageRank selection, effective bandwidth jumps from 6.6 GB/s to 23.4 GB/s on a single Optane SSD.

### 5. GPU software cache with window buffering exploits cross-mini-batch locality (Sections 3.4, 5.5)

GIDS introduces a "window buffering" technique on top of BaM's application-defined GPU software cache. It pre-runs graph sampling for future iterations and stores sampled node IDs in a window buffer. During feature aggregation, nodes that will be reused in upcoming iterations are marked "USE" (not evictable), while nodes with no future reuse are marked "Safe to Evict." With a window depth of 8 and an 8 GB GPU cache on IGB-Full, the cache hit ratio improves 2.19x over random eviction, reducing aggregation time by 1.13x. Even a 4 GB cache with window buffering outperforms a 16 GB cache without it.

### 6. End-to-end speedups of up to 582x on terabyte-scale graphs (Sections 5.6, 5.7)

On Samsung 980 Pro SSDs, GIDS achieves up to 582x speedup over the DGL mmap baseline, 10.62x over Ginex, and 3.09x over BaM alone. On Intel Optane SSDs, speedups reach 8.3x over DGL mmap, 37.21x over Ginex, and 3.23x over BaM. The performance gain is largest for datasets exceeding CPU memory (IGB-Full at 1 TB, IGBH-Full at 2.7 TB). With layer-wise sampling (LADIES), GIDS achieves 412x speedup over DGL and 1.92x over BaM.

### 7. Single-GPU, single-machine training is viable for terabyte-scale graphs (Section 6, Conclusion)

GIDS demonstrates that a single NVIDIA A100 GPU with 512 GB CPU memory and NVMe SSDs can train on datasets more than an order of magnitude larger than CPU memory capacity. This eliminates the need for expensive multi-node/multi-GPU distributed setups. The system evaluated with IGB-Full (1 TB) and IGBH-Full (2.77 TB) on a single machine with one or two NVMe SSDs.

## Relevance to gpu-nvme-direct Project

This paper is directly relevant to the gpu-nvme-direct project in several ways:

1. **Core mechanism:** GIDS builds on BaM to enable GPU threads to issue storage I/O directly to NVMe SSDs, bypassing the CPU and OS entirely. This is the same GPU-initiated direct storage access paradigm that gpu-nvme-direct investigates.

2. **Latency hiding through parallelism:** The paper provides a concrete mathematical model and practical technique (the dynamic storage access accumulator) for determining how many concurrent GPU-to-NVMe requests are needed to saturate SSD bandwidth. This is directly applicable to any GPU-NVMe direct access workload.

3. **Hybrid memory hierarchy design:** The tiered placement strategy (NVMe for bulk data, CPU memory for hot data, GPU memory for cached data) with application-defined eviction policies offers a reusable architectural pattern for gpu-nvme-direct applications beyond GNNs.

4. **Performance characterization of NVMe from GPU:** The paper provides empirical measurements of Intel Optane (1.5M IOPS, 11us latency) and Samsung 980 Pro (700K IOPS, 324us latency) when accessed directly from GPU threads via PCIe, which are valuable reference data points.

5. **Software cache design:** The window buffering technique for application-defined GPU software caches demonstrates how domain-specific knowledge about future access patterns can dramatically improve cache efficiency when the working set far exceeds cache capacity -- a general technique for any GPU-NVMe application with predictable access patterns.

6. **Implementation reference:** GIDS is implemented as an extension to DGL on top of BaM, providing a concrete reference architecture for building GPU-initiated direct storage access systems for production ML workloads.
