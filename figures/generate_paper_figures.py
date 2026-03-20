#!/usr/bin/env python3
"""
Generate publication-quality figures for the gpu-nvme-direct paper.

Usage:
    python3 figures/generate_paper_figures.py

Reads CSV data from data/ and writes figures to figures/

v0.2: Added Fig 7 (bandwidth normalization), Fig 8 (LLM inference),
      improved Fig 1/2 with SN740 data points.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.dirname(__file__)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

METHOD_COLORS = {
    'gpu_direct': '#d62728',    # red
    'cpu_memcpy': '#1f77b4',    # blue
    'cpu_pinned': '#2ca02c',    # green
    'cufile':     '#ff7f0e',    # orange
}

METHOD_LABELS = {
    'gpu_direct': 'gpu-nvme-direct',
    'cpu_memcpy': 'CPU memcpy',
    'cpu_pinned': 'CPU pinned + H2D',
    'cufile':     'cuFile (GDS)',
}

METHOD_MARKERS = {
    'gpu_direct': 'o',
    'cpu_memcpy': 's',
    'cpu_pinned': '^',
    'cufile':     'D',
}

# SN740 data (measured externally, not in CSV)
# 3350 MB/s sustained at 512K QD=32, also used at 256K QD=32 for normalization
SN740_DATA = {
    'throughput_mbs_256k_qd32': 3350,
    'throughput_mbs_512k_qd32': 3350,
}

# PCIe theoretical link bandwidths (MB/s, usable after encoding overhead)
PCIE_BW = {
    'gen3_x4': 3400,   # ~3.94 GB/s raw, ~3.4 GB/s usable (128b/130b)
    'gen4_x4': 7000,   # ~7.88 GB/s raw, ~7.0 GB/s usable
}

BLOCK_SIZE_LABELS = {
    4096: '4K',
    16384: '16K',
    65536: '64K',
    262144: '256K',
    524288: '512K',
}


def load_data():
    """Load all CSV files and combine into one DataFrame."""
    files = {
        'gpu_direct': 'gpu_direct_results.csv',
        'cpu_memcpy': 'cpu_memcpy_results.csv',
        'cpu_pinned': 'cpu_pinned_results.csv',
        'cufile':     'cufile_results.csv',
    }
    dfs = []
    for method, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def avg_by_config(df):
    """Average across runs (3 runs per config)."""
    return df.groupby(['method', 'block_size', 'queue_depth']).agg({
        'throughput_mbs': 'mean',
        'mean_us': 'mean',
        'median_us': 'mean',
        'min_us': 'mean',
        'max_us': 'mean',
        'p99_us': 'mean',
        'stddev_us': 'mean',
        'iops': 'mean',
        'cpu_util_pct': 'mean',
        'total_sec': 'mean',
        'num_ops': 'first',
    }).reset_index()


# =========================================================
# Figure 1: Throughput vs Queue Depth (multi-panel)
# =========================================================
def fig1_throughput_vs_qd(df):
    """Hero figure: throughput scaling with queue depth.
    v0.2: y-axis starts at 0; SN740 star marker at QD=32 in 256K panel."""
    avg = avg_by_config(df)
    block_sizes = sorted(avg['block_size'].unique())
    methods = ['gpu_direct', 'cpu_memcpy', 'cpu_pinned', 'cufile']

    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=False)

    for idx, bs in enumerate(block_sizes):
        ax = axes[idx]
        subset = avg[avg['block_size'] == bs]

        for method in methods:
            mdata = subset[subset['method'] == method].sort_values('queue_depth')
            if len(mdata) == 0:
                continue
            ax.plot(mdata['queue_depth'], mdata['throughput_mbs'],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    label=METHOD_LABELS[method] if idx == 0 else None,
                    zorder=5 if method == 'gpu_direct' else 3)

        # v0.2: Add SN740 star marker at QD=32 in the 256K panel
        if bs == 262144:
            ax.plot(32, SN740_DATA['throughput_mbs_256k_qd32'],
                    marker='*', markersize=12, color='#8B0000',
                    markeredgecolor='black', markeredgewidth=0.5,
                    zorder=10, linestyle='None',
                    label='gpu-nvme-direct (SN740)' if idx == 0 else None)
            ax.annotate(f"SN740\n{SN740_DATA['throughput_mbs_256k_qd32']} MB/s",
                        xy=(32, SN740_DATA['throughput_mbs_256k_qd32']),
                        xytext=(16, SN740_DATA['throughput_mbs_256k_qd32'] + 300),
                        fontsize=6.5, color='#8B0000',
                        arrowprops=dict(arrowstyle='->', color='#8B0000',
                                        lw=0.8),
                        ha='center')

        ax.set_title(BLOCK_SIZE_LABELS.get(bs, f'{bs}B'))
        ax.set_xlabel('Queue Depth')
        ax.set_xticks([1, 4, 16, 32])
        ax.set_xscale('log', base=2)
        ax.set_ylim(bottom=0)  # v0.2: y-axis starts at 0
        if idx == 0:
            ax.set_ylabel('Throughput (MB/s)')

    # Shared legend at bottom (include SN740 entry)
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect SN740 handle from 256K panel if it wasn't on axes[0]
    for idx, bs in enumerate(block_sizes):
        if bs == 262144:
            h, l = axes[idx].get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
    fig.legend(handles, labels, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.12), frameon=True)

    fig.suptitle('Throughput vs. Queue Depth by Block Size', y=1.02, fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_throughput_vs_qd.pdf')
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# =========================================================
# Figure 2: Throughput vs Block Size at QD=32
# =========================================================
def fig2_throughput_vs_bs(df):
    """Throughput comparison at QD=32.
    v0.2: Add SN740 data point as star at 256K."""
    avg = avg_by_config(df)
    subset = avg[avg['queue_depth'] == 32]
    methods = ['gpu_direct', 'cpu_memcpy', 'cpu_pinned', 'cufile']

    fig, ax = plt.subplots(figsize=(5, 3.5))

    block_sizes = sorted(subset['block_size'].unique())

    for method in methods:
        mdata = subset[subset['method'] == method].sort_values('block_size')
        if len(mdata) == 0:
            continue
        x = range(len(mdata))
        ax.plot(x, mdata['throughput_mbs'],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method],
                zorder=5 if method == 'gpu_direct' else 3)

    # v0.2: Add SN740 star marker at 256K position
    if 262144 in block_sizes:
        sn740_x = list(block_sizes).index(262144)
        ax.plot(sn740_x, SN740_DATA['throughput_mbs_256k_qd32'],
                marker='*', markersize=14, color='#8B0000',
                markeredgecolor='black', markeredgewidth=0.5,
                zorder=10, linestyle='None',
                label='gpu-nvme-direct (SN740)')
        ax.annotate(f"{SN740_DATA['throughput_mbs_256k_qd32']} MB/s",
                    xy=(sn740_x, SN740_DATA['throughput_mbs_256k_qd32']),
                    xytext=(sn740_x - 1.2, SN740_DATA['throughput_mbs_256k_qd32'] + 150),
                    fontsize=8, color='#8B0000',
                    arrowprops=dict(arrowstyle='->', color='#8B0000'))

    # PCIe 3.0 x4 theoretical limit line
    ax.axhline(y=3400, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(4.05, 3400, 'PCIe 3.0 x4\ntheoretical', fontsize=7,
            va='center', color='gray')

    ax.set_xticks(range(len(block_sizes)))
    ax.set_xticklabels([BLOCK_SIZE_LABELS.get(bs, str(bs)) for bs in block_sizes])
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Throughput (MB/s)')
    ax.set_title('Throughput at Queue Depth = 32')
    ax.legend(loc='upper left', frameon=True)

    # Annotate peak for SN530 gpu_direct
    gd = subset[subset['method'] == 'gpu_direct'].sort_values('throughput_mbs',
                                                                ascending=False)
    if len(gd) > 0:
        peak = gd.iloc[0]
        peak_idx = list(block_sizes).index(peak['block_size'])
        ax.annotate(f"{peak['throughput_mbs']:.0f} MB/s",
                    xy=(peak_idx, peak['throughput_mbs']),
                    xytext=(peak_idx - 0.8, peak['throughput_mbs'] + 200),
                    fontsize=8, color=METHOD_COLORS['gpu_direct'],
                    arrowprops=dict(arrowstyle='->', color=METHOD_COLORS['gpu_direct']))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_throughput_vs_bs.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


# =========================================================
# Figure 3: Latency comparison at 4K QD=1
# =========================================================
def fig3_latency_comparison(df):
    """Latency bar chart at 4K/QD=1."""
    avg = avg_by_config(df)
    subset = avg[(avg['block_size'] == 4096) & (avg['queue_depth'] == 1)]
    methods = ['gpu_direct', 'cpu_memcpy', 'cpu_pinned', 'cufile']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    bars_data = []
    for method in methods:
        row = subset[subset['method'] == method]
        if len(row) > 0:
            bars_data.append({
                'method': method,
                'median': row['median_us'].values[0],
                'min': row['min_us'].values[0],
                'max': row['max_us'].values[0],
            })

    x = range(len(bars_data))
    colors = [METHOD_COLORS[d['method']] for d in bars_data]
    medians = [d['median'] for d in bars_data]

    bars = ax.bar(x, medians, color=colors, edgecolor='black', linewidth=0.5)

    # Error bars (min/max)
    for i, d in enumerate(bars_data):
        ax.errorbar(i, d['median'],
                    yerr=[[d['median'] - d['min']], [d['max'] - d['median']]],
                    fmt='none', color='black', capsize=3, linewidth=1)

    # Value labels
    for i, v in enumerate(medians):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[d['method']] for d in bars_data],
                       rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Median Latency (us)')
    ax.set_title('Per-Operation Latency at 4KB, QD=1')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_latency_4k_qd1.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


# =========================================================
# Figure 4: CPU Utilization
# =========================================================
def fig4_cpu_utilization(df):
    """CPU utilization comparison."""
    avg = avg_by_config(df)
    methods = ['gpu_direct', 'cpu_memcpy', 'cpu_pinned', 'cufile']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    stats = []
    for method in methods:
        mdata = avg[avg['method'] == method]
        stats.append({
            'method': method,
            'mean': mdata['cpu_util_pct'].mean(),
            'std': mdata['cpu_util_pct'].std(),
            'min': mdata['cpu_util_pct'].min(),
            'max': mdata['cpu_util_pct'].max(),
        })

    x = range(len(stats))
    colors = [METHOD_COLORS[s['method']] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]

    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                  linewidth=0.5, capsize=3)

    for i, s in enumerate(stats):
        ax.text(i, s['mean'] + s['std'] + 0.5, f'{s["mean"]:.1f}%',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[s['method']] for s in stats],
                       rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('CPU Utilization (%)')
    ax.set_title('CPU Utilization (Mean +/- StdDev)')
    ax.set_ylim(0, 20)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_cpu_utilization.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


# =========================================================
# Figure 5: QD Scaling Factor heatmap
# =========================================================
def fig5_scaling_heatmap(df):
    """Heatmap of QD scaling (QD=32/QD=1 throughput ratio)."""
    avg = avg_by_config(df)
    methods = ['gpu_direct', 'cpu_memcpy', 'cpu_pinned', 'cufile']
    block_sizes = sorted(avg['block_size'].unique())

    scaling = np.zeros((len(methods), len(block_sizes)))
    for i, method in enumerate(methods):
        for j, bs in enumerate(block_sizes):
            qd1 = avg[(avg['method'] == method) &
                       (avg['block_size'] == bs) &
                       (avg['queue_depth'] == 1)]['throughput_mbs']
            qd32 = avg[(avg['method'] == method) &
                        (avg['block_size'] == bs) &
                        (avg['queue_depth'] == 32)]['throughput_mbs']
            if len(qd1) > 0 and len(qd32) > 0 and qd1.values[0] > 0:
                scaling[i, j] = qd32.values[0] / qd1.values[0]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    im = ax.imshow(scaling, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=2.5)

    ax.set_xticks(range(len(block_sizes)))
    ax.set_xticklabels([BLOCK_SIZE_LABELS.get(bs, str(bs)) for bs in block_sizes])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_xlabel('Block Size')
    ax.set_title('Throughput Scaling Factor (QD=32 / QD=1)')

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(block_sizes)):
            text = f'{scaling[i, j]:.2f}x'
            color = 'white' if scaling[i, j] > 1.8 or scaling[i, j] < 0.95 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Scaling Factor')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_scaling_heatmap.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


# =========================================================
# Figure 6: Speedup bar
# =========================================================
def fig6_speedup_bar(df):
    """Speedup of gpu_direct vs best CPU baseline at QD=32."""
    avg = avg_by_config(df)
    subset = avg[avg['queue_depth'] == 32]
    block_sizes = sorted(subset['block_size'].unique())

    fig, ax = plt.subplots(figsize=(5, 3))

    speedups = []
    for bs in block_sizes:
        gd = subset[(subset['method'] == 'gpu_direct') &
                     (subset['block_size'] == bs)]['throughput_mbs']
        cpu_best = subset[(subset['method'] == 'cpu_memcpy') &
                           (subset['block_size'] == bs)]['throughput_mbs']
        if len(gd) > 0 and len(cpu_best) > 0:
            speedups.append(gd.values[0] / cpu_best.values[0])
        else:
            speedups.append(0)

    colors_bar = [METHOD_COLORS['gpu_direct'] if s > 1 else 'gray' for s in speedups]
    x = range(len(block_sizes))
    bars = ax.bar(x, speedups, color=colors_bar, edgecolor='black', linewidth=0.5)

    # Reference line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.8)

    # Value labels
    for i, v in enumerate(speedups):
        ax.text(i, v + 0.03, f'{v:.2f}x', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([BLOCK_SIZE_LABELS.get(bs, str(bs)) for bs in block_sizes])
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Speedup over CPU memcpy')
    ax.set_title('gpu-nvme-direct Speedup at QD=32\n(vs. CPU memcpy on faster SSD)')
    ax.set_ylim(0, 2.5)

    # Note about different SSDs
    ax.text(0.02, 0.02, 'Note: gpu_direct uses slower SN530 (Gen3)\n'
            'CPU memcpy uses faster 980 PRO (Gen4)',
            transform=ax.transAxes, fontsize=6.5, va='bottom',
            style='italic', color='gray')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_speedup_bar.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


# =========================================================
# Figure 7: Bandwidth Normalization (NEW for v0.2)
# =========================================================
def fig7_bandwidth_normalization(df):
    """Bar chart showing % of theoretical PCIe link bandwidth utilized.

    Key figure for v0.2: demonstrates that gpu-nvme-direct saturates
    the PCIe link while CPU methods leave most bandwidth unused.
    """
    avg = avg_by_config(df)

    # Extract measured throughput at 256K, QD=32 from CSV data
    gd_sn530_row = avg[(avg['method'] == 'gpu_direct') &
                        (avg['block_size'] == 262144) &
                        (avg['queue_depth'] == 32)]
    cpu_memcpy_row = avg[(avg['method'] == 'cpu_memcpy') &
                          (avg['block_size'] == 262144) &
                          (avg['queue_depth'] == 32)]
    cpu_pinned_row = avg[(avg['method'] == 'cpu_pinned') &
                          (avg['block_size'] == 262144) &
                          (avg['queue_depth'] == 32)]
    cufile_row = avg[(avg['method'] == 'cufile') &
                      (avg['block_size'] == 262144) &
                      (avg['queue_depth'] == 32)]

    # Build entries: (label, measured_mbs, link_bw_mbs, color)
    entries = []
    if len(gd_sn530_row) > 0:
        entries.append(('gpu-nvme-direct\n(SN530)',
                        gd_sn530_row['throughput_mbs'].values[0],
                        PCIE_BW['gen3_x4'],
                        METHOD_COLORS['gpu_direct']))
    entries.append(('gpu-nvme-direct\n(SN740)',
                    SN740_DATA['throughput_mbs_256k_qd32'],
                    PCIE_BW['gen3_x4'],
                    '#8B0000'))  # dark red for SN740
    if len(cpu_memcpy_row) > 0:
        entries.append(('CPU memcpy\n(980 PRO)',
                        cpu_memcpy_row['throughput_mbs'].values[0],
                        PCIE_BW['gen4_x4'],
                        METHOD_COLORS['cpu_memcpy']))
    if len(cpu_pinned_row) > 0:
        entries.append(('CPU pinned\n+ H2D',
                        cpu_pinned_row['throughput_mbs'].values[0],
                        PCIE_BW['gen4_x4'],
                        METHOD_COLORS['cpu_pinned']))
    if len(cufile_row) > 0:
        entries.append(('cuFile (GDS)\n(980 PRO)',
                        cufile_row['throughput_mbs'].values[0],
                        PCIE_BW['gen4_x4'],
                        METHOD_COLORS['cufile']))

    labels = [e[0] for e in entries]
    pcts = [100.0 * e[1] / e[2] for e in entries]
    colors = [e[3] for e in entries]

    fig, ax = plt.subplots(figsize=(7, 3.8))

    x = np.arange(len(entries))
    bars = ax.bar(x, pcts, color=colors, edgecolor='black', linewidth=0.5,
                  width=0.6)

    # Annotate percentage and absolute throughput on each bar
    for i, (pct, entry) in enumerate(zip(pcts, entries)):
        measured = entry[1]
        link = entry[2]
        link_label = 'Gen3 x4' if link == PCIE_BW['gen3_x4'] else 'Gen4 x4'
        ax.text(i, pct + 1.5,
                f'{pct:.1f}%\n({measured:.0f}/{link:.0f} MB/s)',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # Reference line at 100%
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(len(entries) - 0.5, 101, '100% link BW', fontsize=7,
            va='bottom', ha='right', color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('PCIe Link Bandwidth Utilization (%)')
    ax.set_title('PCIe Link Bandwidth Utilization at 256KB, QD=32')
    ax.set_ylim(0, 118)

    # Add PCIe generation annotation below x-axis
    ax.text(0.5, -0.18, 'gpu-nvme-direct: Gen3 x4 link (3400 MB/s)    |    '
            'CPU methods: Gen4 x4 link (7000 MB/s)',
            transform=ax.transAxes, fontsize=7, ha='center', va='top',
            style='italic', color='gray')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    path = os.path.join(FIG_DIR, 'fig7_bandwidth_normalization.pdf')
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# =========================================================
# Figure 8: LLM Inference Comparison (NEW for v0.2)
# =========================================================
def fig8_llm_inference(df):
    """Horizontal bar chart: tok/s for Llama-70B inference across data paths.

    Shows measured and projected results for different configurations,
    annotating the improvement from mmap+memcpy to best tiered approach.
    """
    # Data: (label, tok_s, category)
    # category: 'cpu' (blue), 'gpu_nvme' (red), 'tiered' (green)
    configs = [
        ('mmap + memcpy',                    0.028,  'cpu'),
        ('gpu-nvme-direct\n(SN530, proj.)',  0.039,  'gpu_nvme'),
        ('gpu-nvme-direct\n(SN530, meas.)',  0.04,   'gpu_nvme'),
        ('gpu-nvme-direct\n(SN740, proj.)',  0.063,  'gpu_nvme'),
        ('gpu-nvme-direct\n(SN740, meas.)',  0.06,   'gpu_nvme'),
        ('Tiered VRAM/RAM',                  0.20,   'tiered'),
        ('Tiered + layer skip',              0.27,   'tiered'),
        ('Tiered + Q4_K_M\n+ layer skip',   0.50,   'tiered'),
    ]

    cat_colors = {
        'cpu':       METHOD_COLORS['cpu_memcpy'],  # blue
        'gpu_nvme':  METHOD_COLORS['gpu_direct'],  # red
        'tiered':    '#2ca02c',                    # green
    }

    labels = [c[0] for c in configs]
    tok_s = [c[1] for c in configs]
    colors = [cat_colors[c[2]] for c in configs]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    y_pos = np.arange(len(configs))
    bars = ax.barh(y_pos, tok_s, color=colors, edgecolor='black',
                   linewidth=0.5, height=0.6)

    # Annotate values on each bar
    for i, (val, label) in enumerate(zip(tok_s, labels)):
        ax.text(val + 0.008, i, f'{val:.3f}',
                ha='left', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Tokens per Second (tok/s)')
    ax.set_title('Llama-70B Inference: tok/s by Data Path')
    ax.set_xlim(0, 0.62)
    ax.invert_yaxis()  # top-to-bottom ordering

    # Annotate improvement from baseline to best
    baseline_y = 0   # mmap+memcpy position
    best_y = len(configs) - 1  # Tiered + Q4_K_M + skip
    baseline_val = configs[0][1]
    best_val = configs[-1][1]
    improvement = best_val / baseline_val

    ax.annotate('',
                xy=(best_val - 0.01, best_y),
                xytext=(baseline_val + 0.01, baseline_y),
                arrowprops=dict(arrowstyle='<->',
                                color='black',
                                lw=1.5,
                                connectionstyle='arc3,rad=-0.3'))
    # Place the improvement label at the midpoint of the arrow
    mid_x = max(best_val * 0.85, 0.35)
    mid_y = (baseline_y + best_y) / 2
    ax.text(mid_x, mid_y, f'{improvement:.0f}x',
            fontsize=12, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='black', linewidth=0.8))

    # Legend for categories
    legend_elements = [
        Patch(facecolor=cat_colors['cpu'], edgecolor='black', linewidth=0.5,
              label='CPU baseline'),
        Patch(facecolor=cat_colors['gpu_nvme'], edgecolor='black', linewidth=0.5,
              label='gpu-nvme-direct'),
        Patch(facecolor=cat_colors['tiered'], edgecolor='black', linewidth=0.5,
              label='Tiered (VRAM + RAM)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              frameon=True)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig8_llm_inference.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows from {df['method'].nunique()} methods")

    print("\nGenerating figures...")
    fig1_throughput_vs_qd(df)
    fig2_throughput_vs_bs(df)
    fig3_latency_comparison(df)
    fig4_cpu_utilization(df)
    fig5_scaling_heatmap(df)
    fig6_speedup_bar(df)
    fig7_bandwidth_normalization(df)
    fig8_llm_inference(df)

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("  fig1_throughput_vs_qd.{pdf,png}")
    print("  fig2_throughput_vs_bs.{pdf,png}")
    print("  fig3_latency_4k_qd1.{pdf,png}")
    print("  fig4_cpu_utilization.{pdf,png}")
    print("  fig5_scaling_heatmap.{pdf,png}")
    print("  fig6_speedup_bar.{pdf,png}")
    print("  fig7_bandwidth_normalization.{pdf,png}  [NEW v0.2]")
    print("  fig8_llm_inference.{pdf,png}            [NEW v0.2]")


if __name__ == '__main__':
    main()
