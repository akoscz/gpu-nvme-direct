# Related Work — Detailed Paper Analysis
## Verified 2026-03-19

### Venue Corrections Applied to Paper
- Phoenix: SC'25 (not EuroSys 2025)
- CXL Pool: HotOS'25 (not OSDI 2025)
- InstInfer: HPCA 2025 (not just arxiv)

---

## 1. GIDS (VLDB 2024) — arxiv:2306.16384v2
- Authors: Park, Mailthody, Qureshi, Hwu (UIUC/NVIDIA)
- Extends BaM to GNN training: 582x over DGL, 3.09x over BaM alone
- Samsung 980 Pro: ~3.2 GB/s peak, 700K IOPS
- Requires A100 + custom kernel modules
- **vs gpu-nvme-direct**: comparable raw SSD BW on enterprise vs consumer HW

## 2. Phoenix (SC'25)
- Authors: Yan et al. (multiple Chinese universities)
- 70.3% reduction in GDS I/O overhead, 2.29x for small I/O, 4.11x for large
- Eliminates GDS "phony buffer" using ZONE_DEVICE
- Still requires enterprise GPU ecosystem
- **vs gpu-nvme-direct**: both identify bounce buffer as unnecessary; different approaches

## 3. PIPO (arxiv:2504.03664v2)
- Authors: Liu, Li, Li (Nanjing University)
- 3.1x over FlexGen, GPU util 40%->90%, RTX 3060 (6GB)
- CPU-mediated pipeline: Disk -> CPU -> GPU
- **vs gpu-nvme-direct**: exactly the CPU bottleneck we eliminate

## 4. HeadInfer (arxiv:2502.12574v1)
- Authors: Luo et al. (Caltech, CMU, UW-Madison, Microsoft, Rutgers)
- KV cache 128GB -> 1GB for 1M tokens on RTX 4090
- Context: 25K -> 4,096K tokens
- **vs gpu-nvme-direct**: complementary (weight loading vs KV cache offloading)

## 5. ServerlessLLM (OSDI 2024) — arxiv:2401.14351v2
- Authors: Fu et al. (Edinburgh, NTU Singapore)
- RAID0-NVMe: ~12 GB/s, 10-200x vs serverless baselines
- O_DIRECT + pinned memory + multi-thread pipeline
- **vs gpu-nvme-direct**: optimizes CPU path to near-perfection; we bypass CPU entirely

## 6. InstInfer (HPCA 2025) — arxiv:2409.04992v1
- Authors: Pan et al. (Peking U, Xiamen U, CAS, Huawei, HUST)
- In-storage attention on FPGA CSD, 6.85-11.1x over FlexGen
- CSD internal BW: 11.2 GB/s
- **vs gpu-nvme-direct**: orthogonal (compute near storage vs GPU-initiated I/O)

## 7. NanoQuant (arxiv:2602.06694v1)
- Authors: Chong, Kim, Kim, Choi (Samsung Research, Seoul)
- Sub-1-bit: Llama-2-70B from 138GB to 5.35GB (25.8x compression)
- 70B on 8GB GPU at 20.11 tok/s
- **vs gpu-nvme-direct**: highly complementary. NanoQuant + gpu-nvme-direct =
  ~0.5 tok/s from NVMe alone (42GB -> 8GB reduces I/O by 5x)

## 8. CHEOPS 2025
- Authors: Ren et al. (VU Amsterdam, IBM Research)
- LLM offloading does NOT saturate SSDs: 128KiB reads, 2.0 GiB/s
- **vs gpu-nvme-direct**: validates our motivation. CPU I/O path is the bottleneck.

## 9. CXL Pool (HotOS'25)
- Authors: Zhong et al. (Columbia, Microsoft)
- CXL memory: 2-3x DDR5 latency, 30 GB/s per x8 link
- ~$600/host vs ~$80K for PCIe switches
- **vs gpu-nvme-direct**: CXL could replace host pinned memory as staging area
