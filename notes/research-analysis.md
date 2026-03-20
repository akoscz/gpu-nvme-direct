# Research Analysis: gpu-nvme-direct Paper
## Status: v0.2 (updated 2026-03-19)

## 1. Paper Status

**Draft v0.2** — 18 pages, arxiv format, 9 figures, 9 tables, 23 references.
Compiles cleanly. All figures embedded via \includegraphics.

### Key claims
1. First GPU-initiated NVMe I/O on consumer hardware (GeForce + AMD consumer)
2. 3,350 MB/s sustained on SN740 (99% of PCIe Gen3 x4 link)
3. 2.2x throughput over CPU baselines (using slower SSD)
4. 78-99% link utilization vs 20-25% for CPU methods (bandwidth normalization)
5. 0.06 tok/s Llama 70B NVMe-only, 0.50 tok/s tiered (measured, ntransformer)
6. cuFile/GDS provides zero benefit on GeForce (= cpu_pinned)

### Venue considerations
- **arxiv preprint** — ready to submit immediately
- **USENIX ATC** — best fit (systems, GPU+storage, consumer HW)
- **EuroSys** — good fit for hardware hacking angle
- **ASPLOS** — where BaM was published (high competition)

## 2. Strengths

- Novel: no prior work does GPU-initiated NVMe on consumer HW
- Practical: $2,000 platform, real LLM inference results
- Rigorous: 4 baselines, 5 block sizes, 4 QDs, 3 runs each
- Two NVMe devices (SN530 + SN740) validate device-independence
- Bandwidth normalization analysis handles device asymmetry fairly
- Little's Law analysis explains *why* QD scaling matters
- 7 bugs documented for reproducibility
- Extensive related work (23 refs including 2024-2026 papers)

## 3. Weaknesses to Address

1. **Device asymmetry**: gpu_direct uses Gen3 SSD, CPU baselines use Gen4.
   **Mitigated**: bandwidth normalization shows 3-4x better link utilization.
   SN740 at 99% confirms device-independence.

2. **Only Tier 1**: No Tier 2/3 demonstrated.
   **Mitigated**: Tier 1 throughput limited by NVMe not extra hop.

3. **Sequential only**: No random I/O benchmarks.
   **Mitigated**: sequential is the LLM inference pattern.

4. **Single platform**: Only AMD B450 + RTX 3090.
   **Mitigated**: PCIe posted writes are universal; portability argued.

5. **Driver patch fragility**: Tied to kernel 6.17 + driver 590.48.01.
   **Acknowledged** in limitations section.

## 4. Related Work Summary (verified 2026-03-19)

### GPU-Initiated Storage
- **BaM** [ASPLOS'23]: Enterprise GPU, native P2P, 45.8M IOPS across 10 SSDs
- **GIDS** [VLDB'24]: Extends BaM to GNN, 582x over DGL, Samsung 980 Pro ~3.2 GB/s
- **Phoenix** [SC'25]: Refactored GDS I/O stack, 70% overhead reduction, still enterprise

### LLM Inference + Storage
- **FlexGen** [ICML'23]: OPT-175B on single GPU, CPU-mediated offloading
- **PowerInfer** [SOSP'24]: Activation sparsity, 11.69x over llama.cpp on RTX 4090
- **PIPO** [arxiv]: Fine-grained pipeline, 3.1x over FlexGen, RTX 3060
- **HeadInfer** [arxiv]: KV cache head-wise offloading, 4M tokens on RTX 4090
- **ServerlessLLM** [OSDI'24]: 12 GB/s RAID0-NVMe, optimized CPU pipeline
- **InstInfer** [HPCA'25]: In-storage attention on FPGA CSD
- **NanoQuant** [arxiv]: Sub-1-bit PTQ, 70B -> 5.35 GB, 20 tok/s on 8GB GPU

### Key differentiator
gpu-nvme-direct is the ONLY system combining: GPU-initiated I/O + no CPU in
data path + consumer hardware + no kernel module + no enterprise GPU.

## 5. Publication Checklist

- [x] Paper LaTeX complete (v0.2)
- [x] All figures generated and embedded
- [x] Citations verified (venues corrected: Phoenix->SC'25, CXL->HotOS'25, InstInfer->HPCA'25)
- [x] PDF compiles cleanly (18 pages, 408KB)
- [x] GitHub Pages created (index.html + paper.html + PDF)
- [x] naranjositos.tech page exists and PDF updated
- [ ] Push to GitHub (docs/ with Pages content)
- [ ] Upload to naranjositos R2
- [ ] arxiv submission
