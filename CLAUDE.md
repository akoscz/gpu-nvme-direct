# CLAUDE.md --- gpu-nvme-research

## Objetivo de este repositorio

Este directorio contiene los datos experimentales, documentacion tecnica y
borradores para escribir papers e investigacion sobre **gpu-nvme-direct**:
un sistema que permite a la GPU iniciar operaciones de I/O NVMe de forma
autonoma, sin intervencion del CPU en el data path.

**El codigo fuente esta en `../gpu-nvme-direct`.**
Este directorio es solo para la investigacion, datos y documentacion.

## Que se ha logrado (estado al 2026-02-19)

### Resultado principal

Se demostro que una **GPU GeForce RTX 3090** (hardware de consumo) puede
actuar como procesador autonomo de I/O, emitiendo comandos NVMe directamente
al controlador de almacenamiento via MMIO a traves del bus PCIe, **sin ninguna
intervencion del CPU durante las operaciones de lectura**.

Esto se logro en una plataforma AMD consumer (B550/Ryzen 5800X) donde:
- Los **writes PCIe peer-to-peer funcionan** (posted MemWr via AMD data fabric)
- Los **reads PCIe peer-to-peer NO funcionan** (non-posted, AMD root complex los descarta)
- Se usa un enfoque Tier 1: GPU escribe doorbells al NVMe, datos van a host pinned memory

### Throughput medido

| Metodo | Block Size | Queue Depth | Throughput |
|--------|-----------|-------------|------------|
| gpu_direct | 4K | 1 | ~177 MB/s |
| gpu_direct | 4K | 32 | ~350 MB/s |
| gpu_direct | 64K | 32 | ~2100 MB/s |
| gpu_direct | 256K | 32 | ~2660 MB/s |
| gpu_direct | 512K | 32 | ~2640 MB/s |
| gpu_direct (large seq, 128MB chunks) | 512K | 32 | **2734 MB/s** (pico) |
| gpu_direct (669MB, 1 layer) | 512K | 32 | **2127 MB/s** |

El SN530 es PCIe 3.0 x4 (~3.5 GB/s teorico). Alcanzamos ~78% del maximo
teorico del link, lo cual indica que el overhead del GPU-initiated I/O es minimo.

### Latencia per-op

| Block Size | QD=1 (roundtrip) | QD=32 (inter-completion) |
|-----------|-------------------|--------------------------|
| 4K | ~11 us | ~1.5 us |
| 64K | ~45 us | ~5 us |
| 256K | ~145 us | ~84 us |
| 512K | ~370 us | ~180 us |

### CPU utilization

gpu_direct muestra **6-11% CPU** durante las operaciones (solo el overhead de
la GPU runtime y el kernel launch). El CPU NO participa en el data path de I/O.
En contraste, los metodos CPU (memcpy, pinned) muestran utilizacion significativamente
mayor porque el CPU media cada transferencia.

## Motivacion aplicada: inferencia LLM

### ntransformer (../ntransformer)

Inference engine custom que corre modelos Llama 70B en 24GB VRAM usando SLEP
(Streaming Layer Execution Pipeline). Lee layers del modelo desde almacenamiento
de uno en uno, ejecuta en GPU con double buffering.

**Pipeline actual (con CPU bottleneck):**
```
NVMe -> page cache -> CPU memcpy -> pinned staging -> H2D DMA -> GPU compute
         (mmap)      (worker thread)   (1.3 GB x 2)     (PCIe)
```
Resultado: **0.02 tok/s en 70B**. El memcpy del CPU es el bottleneck.

**Pipeline objetivo (GPU autonomo):**
```
GPU doorbell write -> NVMe DMA -> host pinned buffer -> GPU compute
  (MMIO a BAR0)      (autonomo)   (sin CPU memcpy)    (lee directo)
```

**Proyeccion de rendimiento:**

| Ruta de datos | BW | 1 layer (669MB Q6_K) | 80 layers | tok/s |
|---------------|------|---------------------|-----------|-------|
| mmap+memcpy+H2D (actual) | ~1.5-2 GB/s | ~400ms | 32s | 0.03 |
| **gpu-nvme-direct (SN530 Gen3)** | **2.1 GB/s** | **315ms** | **25s** | **0.04** |
| gpu-nvme-direct pico | 2.7 GB/s | ~248ms | ~20s | 0.05 |
| Con NVMe Gen4 x4 | ~4-6 GB/s | ~130ms | 10s | 0.10 |
| Warm page cache + H2D | ~13 GB/s | ~52ms | 4.1s | 0.24 |

Para **Q8_0** (70B = ~70GB total): NO CABE en 48GB RAM, streaming desde NVMe es
obligatorio. gpu-nvme-direct es el unico camino viable sin ampliar RAM.

## Hardware de prueba

| Componente | Detalle |
|---|---|
| GPU | NVIDIA RTX 3090 (GA102, sm_86, 24GB, Ampere) at 0000:0a:00.0 |
| CPU | AMD Ryzen 7 5800X (Zen 3, 8C/16T, 3.8-4.7GHz) |
| Plataforma | AMD B550, PCIe 4.0 (root ports CPU-direct) |
| RAM | 48GB DDR4-3200 |
| NVMe test (VFIO) | WD SN530 1TB, 0000:0b:00.0, PCIe 3.0 x4, NVMe 1.4.0 |
| NVMe sistema | Samsung 980 PRO 500GB, /dev/nvme0n1, PCIe 4.0 x4 |
| OS | Ubuntu 25.10 (kernel 6.17.0-14-generic) |
| CUDA | 13.1 |
| Driver NVIDIA | 590.48.01 (open kernel modules, DKMS, patcheado) |
| Compilador | gcc-14 + nvcc (gcc-15 incompatible con CUDA 13.1) |

### Topologia PCIe
```
Root Complex (AMD Matisse/Vermeer)
+-- Root Port 03.1 -> GPU 0a:00.0     (PCIe 4.0 x16, CPU-direct)
+-- Root Port 03.4 -> NVMe 0b:00.0    (PCIe 3.0 x4, via chipset B550)
```

**IMPORTANTE**: El NVMe de test (SN530) esta conectado al chipset B550, no
directamente al CPU. Esto introduce latencia adicional del fabric, pero el
throughput de PCIe 3.0 x4 (~3.5 GB/s) es el factor limitante principal.

## Datos experimentales

### Archivos CSV en data/

- `gpu_direct_results.csv` -- gpu-nvme-direct benchmark completo
  - Block sizes: 4K, 16K, 64K, 256K, 512K
  - Queue depths: 1, 4, 16, 32
  - Pattern: sequential
  - 3 runs per configuracion
  - 60 filas de datos

- `cpu_memcpy_results.csv` -- CPU read + memcpy (O_DIRECT + aio)
- `cpu_pinned_results.csv` -- CPU read a pinned memory + H2D (cuMemcpy)
- `cufile_results.csv` -- NVIDIA GPUDirect Storage (cuFile API)

**NOTA**: Los benchmarks de CPU usan el NVMe del sistema (Samsung 980 PRO,
PCIe 4.0 x4), que es un dispositivo fisico DIFERENTE al SN530. Esto es porque
el SN530 esta bajo VFIO (no tiene /dev/ path). La comparacion es valida para
medir el **overhead del metodo de I/O** y la **utilizacion de CPU**, no para
comparar el throughput crudo de dispositivos.

### Formato CSV

```
method,block_size,queue_depth,pattern,num_ops,
min_us,max_us,mean_us,median_us,p99_us,p999_us,stddev_us,
throughput_mbs,iops,cpu_util_pct,total_sec
```

- `method`: gpu_direct, cpu_memcpy, cpu_pinned, cufile
- `block_size`: bytes por operacion de I/O
- `queue_depth`: operaciones en vuelo simultaneas
- `pattern`: seq (secuencial) o rand (aleatorio)
- `*_us`: latencias en microsegundos (del GPU clock para gpu_direct,
  de clock_gettime para CPU)
- `throughput_mbs`: MB/s (1 MB = 1048576 bytes)
- `iops`: operaciones por segundo
- `cpu_util_pct`: porcentaje de CPU (/proc/stat, todos los cores)

### Test de large sequential read (test_large_read)

Resultados previos (no en CSV, ejecutados manualmente):

| Size | Throughput | Pipeline depth |
|------|-----------|---------------|
| 4 MB | 2122 MB/s | 32 |
| 128 MB | 2734 MB/s (pico) | 32 |
| 669 MB (1 layer Q6_K) | 2127 MB/s | 32 |

## Arquitectura tecnica

### Innovaciones clave

1. **GPU-initiated NVMe I/O en hardware de consumo**: Primera demostracion
   (que sepamos) de un kernel CUDA en una GeForce RTX que emite comandos NVMe
   directamente, sin SPDK, sin GPUDirect Storage, sin hardware enterprise.

2. **Tier 1 approach**: Solo necesita GPU writes (posted PCIe MemWr) que
   funcionan en plataformas AMD consumer. No necesita P2P DMA completo.

3. **PTX MMIO instructions**: Usa `st.relaxed.mmio.sys` y `ld.relaxed.mmio.sys`
   en lugar de volatile C++ (que no genera las instrucciones PCIe correctas).

4. **cudaHostRegisterIoMemory en GeForce**: Funciona tras parchear el driver
   NVIDIA para kernel 6.12+ (follow_pfn() removido).

5. **Pipeline depth optimization**: Un solo GPU thread mantiene 32 comandos
   NVMe en vuelo simultaneamente, saturando el ancho de banda del SSD.

### Como funciona

```
                    GPU (RTX 3090)                    NVMe (SN530)
                    +------------+                    +----------+
                    |            |                    |          |
                    | CUDA kernel|  <=== MMIO ===>   | BAR0     |
                    | (1 thread) |  doorbell writes   | registers|
                    |            |                    |          |
                    +------+-----+                    +----+-----+
                           |                               |
                           v                               v
                    +------+-----+                    +----+-----+
                    | Host pinned|  <=== DMA ====    | NVMe DMA |
                    | memory     |  (data transfer)  | engine   |
                    | (SQ/CQ/    |                    |          |
                    |  data buf) |                    +----------+
                    +------------+

Flujo:
1. GPU escribe SQ entry (64 bytes) en host pinned memory
2. GPU escribe doorbell via MMIO (PTX st.mmio.sys) al NVMe BAR0
3. NVMe DMA-lee el SQ entry desde host memory
4. NVMe ejecuta la lectura del flash
5. NVMe DMA-escribe los datos al buffer en host pinned memory
6. NVMe escribe CQ entry (16 bytes) en host pinned memory
7. GPU polea el CQ entry y detecta la completacion via phase bit
8. GPU escribe CQ head doorbell al NVMe

El CPU NO interviene en ningun paso de este flujo.
```

### Memory ordering critico

```
SQ entry writes -> __threadfence_system() -> doorbell MMIO write -> __threadfence_system()
```

Sin `__threadfence_system()`, las escrituras al SQ entry pueden llegar al NVMe
DESPUES del doorbell, causando que el NVMe lea datos stale del SQ.

### PRP lists para transfers >4KB

Para leer >4KB (una pagina), NVMe requiere PRP (Physical Region Page) lists:
- PRP1: direccion fisica de la primera pagina de datos
- PRP2: direccion fisica de la segunda pagina (2 paginas) O puntero a una
  lista de direcciones fisicas (>2 paginas)

Las PRP lists se pre-construyen en el CPU usando /proc/self/pagemap y se
pasan al kernel GPU como arrays.

## Bugs criticos descubiertos y resueltos

| # | Bug | Impacto | Causa raiz | Solucion |
|---|-----|---------|-----------|----------|
| 1 | cudaHostRegisterIoMemory falla | No se puede mapear BAR0 al GPU | follow_pfn() removido en kernel 6.12+ | Parche os-mlock.c: PFN desde vm_pgoff |
| 2 | GPU reads retornan 0xffffffff | GPU no puede leer registros NVMe | AMD root complex descarta non-posted P2P TLPs | Tier 1: solo usar writes |
| 3 | Admin commands cuelgan | No se puede inicializar el controller | Admin CQ no page-aligned (cudaMallocHost suballocator) | Allocations >= 4096 bytes |
| 4 | Pipeline depth >=4 timeout | No escala con profundidad de cola | NVMe completions out-of-order + cq_poll_for_cid descartaba CQEs | cq_poll_completion (acepta cualquier CID) |
| 5 | Queue state reset post-warmup | Benchmark ops timeout tras warmup | NVMe controller internal pointers desincronizados | No resetear queue state; dejar rolling |
| 6 | GPU reads crashean NVMe link | Dispositivo desaparece del PCIe bus | Error acumulado de failed reads | Evitar GPU reads; power cycle si ocurre |
| 7 | 64-bit MMIO write corrupta | Registros NVMe reciben valores incorrectos | Write no atomica en BAR0 | Dos writes de 32 bits (low first) |

## Papers y referencias clave

- **BaM (ASPLOS 2023)**: https://arxiv.org/abs/2203.04910
  GPU-initiated on-demand storage access. El mas cercano a nuestro trabajo.
  Diferencias: BaM usa hardware Tesla/A-series con P2P nativo, nosotros
  demostramos que funciona en GeForce consumer sin P2P completo.

- **libnvm/ssd-gpu-dma**: https://github.com/enfiskutensykkel/ssd-gpu-dma
  Implementacion C de GPU DMA a NVMe. Requiere kernel module para P2P.
  Nosotros no necesitamos kernel module para Tier 1.

- **GPUDirect Storage (GDS)**: https://docs.nvidia.com/gpudirect-storage/
  Solucion oficial de NVIDIA. Requiere hardware enterprise (Tesla/A/H series)
  y drivers especificos. No disponible en GeForce.

- **SPDK**: https://spdk.io/doc/nvme.html
  NVMe userspace driver de Intel. Solo CPU, no GPU-initiated.

- **tinygrad P2P patch**: https://github.com/tinygrad/open-gpu-kernel-modules
  Patch para habilitar P2P en GeForce. Complementario a nuestro trabajo.

## Contribuciones potenciales para paper

1. **Demostracion de factibilidad**: GPU-initiated NVMe I/O en hardware consumer
   (GeForce + AMD platform), sin ninguna dependencia de hardware enterprise.

2. **Analisis de P2P en plataformas consumer**: Documentacion detallada de que
   funciona y que no en AMD B550/Ryzen con GeForce RTX 3090.

3. **Tiered approach**: Propuesta de 3 niveles que degrada gracefully segun
   el soporte de P2P disponible.

4. **Benchmarks formales**: Comparacion cuantitativa vs cuFile, CPU memcpy,
   CPU pinned, con metricas de latencia, throughput, IOPS, y CPU utilization.

5. **Aplicacion practica a LLM inference**: Integracion con ntransformer para
   servir modelos 70B en hardware de consumo (~$2000 total).

6. **Catalogo de bugs y workarounds**: Documentacion practica para quien
   quiera replicar en hardware similar.

## Como usar este directorio

### Para analizar datos
```bash
# Los CSV estan en data/
# Formato: method,block_size,queue_depth,pattern,num_ops,min_us,max_us,...

# Ejemplo: extraer throughput de gpu_direct por queue_depth
awk -F, '$1=="gpu_direct" {print $2, $3, $13}' data/gpu_direct_results.csv
```

### Para generar figuras
Usa Python con matplotlib/seaborn. Los scripts de visualizacion estan en
`../gpu-nvme-direct/bench/plot_results.py` (copiarlo aqui si se necesita).

### Para escribir el paper
El borrador va en `drafts/`. Usa LaTeX o Markdown segun preferencia.
Los datos clave para figuras estan en los CSV. Las arquitecturas y diagramas
se pueden generar desde las descripciones en este archivo.

### Para reproducir los benchmarks
```bash
cd ../gpu-nvme-direct/build-hw

# Prerequisito: NVMe setup (VFIO)
sudo modprobe vfio enable_unsafe_noiommu_mode=1
sudo modprobe vfio-pci
sudo bash scripts/setup_vfio.sh 0000:0b:00.0
sudo sh -c 'echo on > /sys/bus/pci/devices/0000:0b:00.0/power/control'
sudo setpci -s 0000:0b:00.0 0x84.W=0x0008
sudo setpci -s 0000:0b:00.0 COMMAND=0x0006

# GPU direct benchmark
sudo ./bench/bench_gpu_direct --device 0000:0b:00.0 \
  --block-size 512K --queue-depth 32 --num-ops 200 --pattern seq \
  --output results.csv

# CPU baselines (sistema NVMe)
sudo ./bench/bench_cpu_memcpy --device /dev/nvme0n1 \
  --block-size 512K --queue-depth 1 --num-ops 200 --pattern seq \
  --output results.csv

# Sweep completo
python3 bench/bench_sweep.py --bdf 0000:0b:00.0 --device /dev/nvme0n1 \
  --block-sizes 4K 16K 64K 256K 512K --queue-depths 1 4 16 32 \
  --patterns seq --runs 3 --output-dir bench_results
```

## Estructura de este directorio

```
gpu-nvme-research/
  CLAUDE.md              -- Este archivo (configuracion e instrucciones)
  data/                  -- CSVs con resultados de benchmarks
    gpu_direct_results.csv
    cpu_memcpy_results.csv
    cpu_pinned_results.csv
    cufile_results.csv
  figures/               -- Graficos generados (PNG/PDF)
  drafts/                -- Borradores de papers
  notes/                 -- Notas de investigacion
```
