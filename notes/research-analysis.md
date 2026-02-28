# Research Analysis: gpu-nvme-direct Paper

## Status: Draft v0.1 (2026-02-19)

---

## 1. Resumen de Datos Experimentales

### 1.1 Dataset

- **4 metodos**: gpu_direct, cpu_memcpy, cpu_pinned, cufile
- **5 block sizes**: 4K, 16K, 64K, 256K, 512K
- **4 queue depths**: 1, 4, 16, 32
- **3 runs por configuracion** = 60 filas por metodo
- **Patron**: secuencial

### 1.2 Hallazgos Clave

#### Throughput (MB/s, QD=32, promedio de 3 runs)

| Block Size | gpu_direct | cpu_memcpy | cpu_pinned | cufile | Ventaja gpu_direct |
|-----------|-----------|-----------|-----------|--------|-------------------|
| 4K | **336** | 180 | 153 | 151 | 1.87x vs memcpy |
| 16K | **1,393** | 626 | 516 | 510 | 2.22x vs memcpy |
| 64K | **2,111** | 1,373 | 1,152 | 1,144 | 1.54x vs memcpy |
| 256K | **2,666** | 1,731 | 1,409 | 1,396 | 1.54x vs memcpy |
| 512K | **2,634** | 2,008 | 1,700 | 1,696 | 1.31x vs memcpy |

**CRITICO**: gpu_direct usa el SN530 (PCIe 3.0 x4, ~3.5 GB/s). Los baselines
CPU usan el 980 PRO (PCIe 4.0 x4, ~7 GB/s). La ventaja de gpu_direct es
conservadora — con el mismo dispositivo seria aun mayor.

#### Queue Depth Scaling (Throughput QD=32 / QD=1)

| Block Size | gpu_direct | cpu_memcpy | cpu_pinned | cufile |
|-----------|-----------|-----------|-----------|--------|
| 4K | **1.95x** | 1.08x | 0.96x | 1.01x |
| 16K | **2.27x** | 1.00x | 0.98x | 1.00x |
| 64K | **1.41x** | 1.10x | 0.98x | 1.00x |
| 256K | **1.65x** | 1.00x | 0.99x | 1.01x |
| 512K | **1.98x** | 1.02x | 1.00x | 0.99x |

**Conclusion**: Queue depth scaling es la ventaja fundamental. Los metodos CPU
NO escalan con QD porque serializan operaciones.

#### Latencia Mediana (us)

**gpu_direct, por QD:**

| Block Size | QD=1 | QD=4 | QD=16 | QD=32 |
|-----------|------|------|-------|-------|
| 4K | 10.6 | 1.70 | 1.49 | 1.50 |
| 16K | 14.2 | 15.5 | 1.48 | 1.47 |
| 64K | 29.0 | 33.4 | 13.0 | 14.0 |
| 256K | 126 | 69 | 63 | 63 |
| 512K | 412 | 154 | 141 | 141 |

Sub-2us inter-completion a 4K/QD=32 demuestra polling extremadamente eficiente.

#### CPU Utilization

| Metodo | Media | Min | Max |
|--------|-------|-----|-----|
| gpu_direct | 8.5% | 5.6% | 15.0% |
| cpu_memcpy | 4.3% | 0.0% | 12.2% |
| cpu_pinned | 4.7% | 0.0% | 12.5% |
| cufile | 5.1% | 2.1% | 10.4% |

El CPU de gpu_direct es overhead de CUDA runtime, NO I/O. El CPU esta libre
para computacion.

### 1.3 Observaciones Notables

1. **cufile ≈ cpu_pinned**: Dentro de 3% en todas las configs. GDS no hace
   nada en GeForce.

2. **Outlier cpu_memcpy 64K/QD=1 run 1**: max_us=4197 (scheduler interrupt o
   page fault). Throughput cae a 1001 MB/s en ese run.

3. **gpu_direct 4K/QD=4 alta varianza**: 183, 262, 215 MB/s. Zona de
   transicion del pipeline.

4. **Bimodalidad en latencia a QD=4-16**: min_us ~0.8-1.5, max_us ~100-250.
   Algunas completions llegan casi inmediatamente (batched).

---

## 2. Figuras para el Paper

### Figura 1: Throughput vs Queue Depth (HERO FIGURE)
- **Tipo**: Line plot, 5 paneles (uno por block size)
- **Ejes**: X=QD (1,4,16,32), Y=Throughput (MB/s)
- **Series**: 4 lineas por metodo, colores distintos
- **Mensaje**: gpu_direct escala con QD; CPU metodos NO escalan
- **Panel mas dramatico**: 16K (2.22x gap a QD=32)

### Figura 2: Throughput vs Block Size a QD=32
- **Tipo**: Line plot, una linea por metodo
- **Anotaciones**: Peak (2666 MB/s a 256K), limite teorico PCIe 3.0 x4
- **Nota**: CPU usa SSD mas rapido pero pierde

### Figura 3: Latencia CDF o Box Plot a 4K/QD=1
- **Tipo**: CDF o box plot
- **Datos**: Distribucion per-op latency
- **Mensaje**: gpu_direct 10.6us vs cpu_memcpy 19.6us (46% menos)

### Figura 4: CPU Utilization
- **Tipo**: Bar chart, media +/- stddev por metodo
- **Nota**: Explicar que gpu_direct CPU es runtime, no I/O

### Tabla 1: Peak Performance
- Mejor throughput, IOPS, latencia por metodo
- Speedup ratio

### Tabla 2: QD Scaling Factor
- gpu_direct vs todos los baselines

### Figura 5 (opcional): Latencia vs Throughput scatter
- Punto por cada (metodo, block_size, QD)
- Color por metodo, size por block_size
- Pareto frontier

---

## 3. Related Work — Papers Identificados

### Tier 1: Directamente Relacionados
1. **BaM (ASPLOS 2023)** — El mas cercano. GPU-orchestrated storage. Enterprise HW.
2. **GPUDirect Storage (NVIDIA)** — Solucion oficial. Enterprise only.
3. **SPDK (Intel)** — NVMe userspace, CPU-driven.
4. **libnvm/SmartIO (Markussen 2021)** — GPU DMA NVMe. Kernel module req.

### Tier 2: GPU-Centric I/O
5. **GPUfs (Silberstein 2014)** — GPU file API, CPU in data path.
6. **DRAGON (Markthub 2018)** — GPU NVM via page faults. Enterprise.
7. **SPIN (Bergman 2017)** — OS P2P DMA integration.

### Tier 3: LLM Inference + Storage
8. **FlexGen (ICML 2023)** — Single GPU, CPU-mediated offload.
9. **DeepSpeed-Inference (SC 2022)** — Multi-GPU, ZeRO-Infinity NVMe.
10. **PowerInfer (2023)** — Consumer GPU, sparsity-based.
11. **LLM in a Flash (Apple 2023)** — Flash storage inference.
12. **vLLM (SOSP 2023)** — PagedAttention (KV cache, no storage offload).

### Tier 4: PCIe P2P
13. **SmartIO/PCIe networking** — P2P characterization.
14. **tinygrad P2P patches** — GeForce GPU-GPU P2P.

---

## 4. Claim Principal del Paper

### Thesis Statement
> A consumer GPU (GeForce RTX 3090) can act as an autonomous I/O processor,
> directly driving an NVMe controller via PCIe MMIO writes, achieving 78% of
> the link bandwidth without any CPU involvement in the data path.

### Novelty Claims
1. **Primera demostracion** de GPU-initiated NVMe I/O en hardware consumer
   (GeForce + AMD consumer platform).
2. **Tiered approach** que degrada gracefully segun P2P disponible.
3. **Caracterizacion** de P2P asimetrico en AMD B550 (writes OK, reads fail).
4. **Benchmarks comprehensivos** vs 4 baselines con metricas detalladas.
5. **Aplicacion practica** a LLM inference.
6. **Catalogo de bugs** para reproducibilidad.

---

## 5. Venue Target

### Opciones principales (en orden de fit):
1. **USENIX ATC** — Systems paper, buen fit para la combinacion
   GPU+storage+consumer-HW.
2. **EuroSys** — Systems venue europeo, acepta bien papers de hardware hacking.
3. **ASPLOS** — Donde se publico BaM. Alto impacto, alta competencia.
4. **FAST** — Storage-focused, pero el angulo GPU es menos natural.

### Workshop alternatives:
- **HotStorage** — Workshop de USENIX, buen fit para version corta.
- **GPGPU (workshop en ASPLOS/PPoPP)** — GPU computing.
- **ATC Poster/WIP** — Para feedback temprano.

---

## 6. Debilidades a Abordar

1. **Dispositivos asimetricos**: gpu_direct usa SN530 (Gen3), baselines usan
   980 PRO (Gen4). Mitigacion: la ventaja es conservadora; mismo dispositivo
   mostraria gap mayor.

2. **Solo Tier 1**: No demostramos Tier 2/3. Mitigacion: Tier 1 prueba el
   concepto; throughput esta limitado por NVMe no por el hop extra.

3. **Solo sequential**: No hay random I/O benchmarks. Mitigacion: sequential
   es el patron de LLM inference; random es future work.

4. **Single platform**: Solo AMD B550 + RTX 3090. Mitigacion: documentamos
   la portabilidad esperada.

5. **No end-to-end LLM**: Solo proyeccion, no integracion real con
   ntransformer. Mitigacion: los datos de throughput son medidos, solo la
   inferencia tok/s es proyectada.

---

## 7. TODO para completar el paper

- [ ] Generar figuras (Python/matplotlib) desde los CSV
- [ ] Verificar todas las cifras del draft contra los CSV
- [ ] Agregar figuras al LaTeX
- [ ] Escribir abstract mas conciso (actualmente ~250 palabras, target 200)
- [ ] Revisar related work contra papers mas recientes (2025-2026)
- [ ] Agregar diagrama de arquitectura (TikZ o similar)
- [ ] Integrar resultados de test_large_read
- [ ] Formatear para venue target (USENIX ATC template)
- [ ] Agregar acknowledgments si aplica
- [ ] Revisar licencia del codigo (BSD-2-Clause) para open-source claim
