# ServerlessLLM: Low-Latency Serverless Inference for Large Language Models

## Paper Metadata

- **Title:** ServerlessLLM: Low-Latency Serverless Inference for Large Language Models
- **Authors:** Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, Luo Mai
- **Affiliations:** University of Edinburgh, NTU Singapore
- **Venue:** OSDI 2024
- **arXiv:** [2401.14351v2](https://arxiv.org/abs/2401.14351)
- **Code:** [github.com/ServerlessLLM/ServerlessLLM](https://github.com/ServerlessLLM/ServerlessLLM)

---

## Key Findings

### 1. Multi-Tier Storage Hierarchy is Massively Underutilized (Section 3.1)

GPU inference servers contain a rich storage hierarchy -- up to 4 TB DRAM, 64 TB NVMe SSDs, and 192 TB SATA SSDs on a contemporary 8-GPU server -- yet current serverless systems (KServe, Ray Serve) barely use host memory and minimally employ SSDs for caching. ServerlessLLM exploits this full hierarchy for local checkpoint storage. An 8-GPU server with PCIe 5.0 can deliver **512 GB/s aggregate bandwidth** from DRAM to GPUs and **~60 GB/s** from NVMe SSDs (RAID 0) to DRAM.

### 2. Loading-Optimized Checkpoint Format Enables Sequential, Chunk-Based I/O (Section 4.1)

ServerlessLLM converts standard checkpoints into a loading-optimized format with two critical properties:
- **Sequential chunk-based reading:** tensors are pre-partitioned per target GPU into binary files (no metadata interleaved), enabling large sequential I/O.
- **Direct tensor addressing:** a tensor index file maps tensor names to (GPU id, offset, size) tuples with memory-word alignment, so tensors can be directly addressed in GPU memory via `base + offset` (using CUDA IPC handles) without deserialization overhead.

The loading process is decoupled from inference initialization -- the model manager loads binary data while the inference process initializes the model object, and they synchronize only at the end.

### 3. Multi-Tier Loading Pipeline Saturates Storage Bandwidth (Section 4.2)

The multi-tier loading subsystem incorporates several key NVMe-to-GPU optimizations:
- **Direct I/O (`O_DIRECT`):** bypasses kernel page cache, avoiding excessive copies. This alone provides a **2.1x throughput improvement** over standard reads. ServerlessLLM explicitly rejects `mmap` (used by Safetensors) because it causes ~112K page faults on cold start and lacks predictable performance guarantees.
- **Pinned memory pool:** eliminates redundant DRAM-to-GPU copies by enabling GPU DMA transfers with minimal CPU involvement. Contributes a **1.4x throughput improvement**.
- **Multi-thread intra-tier concurrency:** multiple I/O threads per storage tier exploit SSD internal parallelism. Contributes a **2.3x throughput improvement**.
- **Flexible pipeline across tiers:** task-queue-based pipeline avoids synchronization barriers between storage tiers. Contributes a **1.5x improvement**.
- **Parallel PCIe links:** checkpoint partitions are loaded concurrently across multiple GPUs, each through its own PCIe link.

Combined result: ServerlessLLM fully saturates available storage bandwidth on all tested media (SATA, NVMe, RAID0-NVMe), achieving normalized bandwidth utilization close to 100% (measured against FIO optimal baselines). With RAID0-NVMe (12 GB/s throughput), ServerlessLLM is **3.6-8.2x faster** than PyTorch and Safetensors. The system requires only **4 CPU cores** to achieve maximum bandwidth utilization.

### 4. Dramatic Cold-Start Latency Reductions in End-to-End Evaluation (Section 6.3)

In real-world serverless workloads (Azure Trace):
- **OPT-6.7B:** ServerlessLLM starts in **0.8 seconds** vs. Ray Serve's 12.1s and Ray Serve+Cache's 8.2s (>10x improvement).
- **OPT-30B:** ServerlessLLM starts in **7.5 seconds** vs. Ray Serve's 213s and Ray Serve+Cache's 199.2s (**28x improvement**). Parallel PCIe links are especially effective for large multi-GPU models loaded from pinned memory.
- **Overall:** 10-200x latency improvement across LLM inference workloads. ServerlessLLM fulfills 89% of requests within 300s timeout with OPT-30B vs. only 26% for Ray Serve+Cache.

### 5. LoRA Adapter Loading Also Benefits (Section 6.1)

For a LoRA adapter (rank=32, 1GB) of LLaMA-70B, ServerlessLLM achieves **83.5ms** loading latency -- **4.4x faster** than Safetensors (370ms), demonstrating that the optimized loader design is effective even for small checkpoints.

### 6. Live Migration of LLM Inference Outperforms Preemption (Sections 5, 6.2)

ServerlessLLM introduces token-based live migration: instead of transferring the large KV-cache (1-10 GB), it migrates only the tokens (10-100s KB) and recomputes the KV-cache at the destination. KV-cache recomputation for 1000 tokens takes the same time as generating ~100 new tokens, ensuring convergence. Combined with the locality-aware scheduler, ServerlessLLM achieves **1.27-1.95x lower P99 latency** compared to preemption-based (Shepherd*) and availability-driven (Serverless) schedulers.

### 7. Startup-Time-Optimized Scheduling With Accurate Cost Models (Section 5)

The cluster scheduler estimates model loading time as `q + n/b` (queue time + model size / bandwidth), using the slowest tier in the pipeline as the bottleneck. GPU time estimation error is bounded at 5ms, SSD loading estimation error at 40ms. The scheduler uses dynamic programming to select the optimal server, considering whether migration to free a locality-rich server is faster than loading from a lower tier.

---

## Relevance to gpu-nvme-direct

ServerlessLLM represents the **current state of the art** in optimizing the NVMe-to-GPU loading pipeline for LLM checkpoints. Its architecture reveals precisely where gpu-nvme-direct could provide further acceleration:

1. **ServerlessLLM's data path is: NVMe -> DRAM (pinned) -> GPU.** Even with `O_DIRECT` and pinned memory, data still transits through host DRAM and requires CPU involvement to orchestrate I/O threads and DMA transfers. gpu-nvme-direct (using GPUDirect Storage / GDS) could **eliminate the DRAM hop entirely**, enabling NVMe -> GPU transfers via PCIe peer-to-peer, potentially doubling effective bandwidth for the NVMe-to-GPU path.

2. **ServerlessLLM's pipeline is bottlenecked by NVMe-to-DRAM bandwidth** (~12 GB/s for RAID0-NVMe in their testbed). With GDS, the aggregate NVMe bandwidth could feed directly into GPU HBM without competing for DRAM bandwidth, and the parallel-PCIe-link optimization that ServerlessLLM already uses for DRAM-to-GPU could be applied to direct NVMe-to-GPU paths.

3. **ServerlessLLM's loading-optimized checkpoint format is already ideally structured for GDS:** sequential, chunk-based binary files with word-aligned offsets. This format requires zero modification to work with `cuFile` APIs -- the tensor binary partitions can be read directly into GPU memory with `cuFileRead()`.

4. **The 4-CPU-core requirement** for achieving maximum bandwidth could be reduced or eliminated with GDS offloading I/O orchestration to the GPU/DMA engines.

5. **ServerlessLLM's cost model (`q + n/b`)** would need to be extended with a new bandwidth parameter for the NVMe-GPU direct path, but the scheduling framework is already tier-aware and extensible.

In summary, ServerlessLLM has done the systems work to optimize everything *around* the storage-to-GPU path (checkpoint format, pipeline structure, scheduling, migration). gpu-nvme-direct targets the remaining bottleneck: the data path itself. Combining both would yield a system where NVMe bandwidth feeds directly into GPU memory through an already-optimized checkpoint format and scheduling framework.
