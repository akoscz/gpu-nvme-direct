#!/usr/bin/env python3
"""
Generate publication-quality figures for the gpu-nvme-direct paper.

Usage:
    python3 figures/generate_paper_figures.py

Reads CSV data from data/ and writes figures to figures/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    """Hero figure: throughput scaling with queue depth."""
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

        ax.set_title(BLOCK_SIZE_LABELS.get(bs, f'{bs}B'))
        ax.set_xlabel('Queue Depth')
        ax.set_xticks([1, 4, 16, 32])
        ax.set_xscale('log', base=2)
        if idx == 0:
            ax.set_ylabel('Throughput (MB/s)')

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
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
    """Throughput comparison at QD=32."""
    avg = avg_by_config(df)
    subset = avg[avg['queue_depth'] == 32]
    methods = ['gpu_direct', 'cpu_memcpy', 'cpu_pinned', 'cufile']

    fig, ax = plt.subplots(figsize=(5, 3.5))

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

    # PCIe 3.0 x4 theoretical limit line
    ax.axhline(y=3400, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(4.05, 3400, 'PCIe 3.0 x4\ntheoretical', fontsize=7,
            va='center', color='gray')

    block_sizes = sorted(subset['block_size'].unique())
    ax.set_xticks(range(len(block_sizes)))
    ax.set_xticklabels([BLOCK_SIZE_LABELS.get(bs, str(bs)) for bs in block_sizes])
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Throughput (MB/s)')
    ax.set_title('Throughput at Queue Depth = 32')
    ax.legend(loc='upper left', frameon=True)

    # Annotate peak
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
# Figure 6: Architecture diagram (text-based for now)
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

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("  fig1_throughput_vs_qd.{pdf,png}")
    print("  fig2_throughput_vs_bs.{pdf,png}")
    print("  fig3_latency_4k_qd1.{pdf,png}")
    print("  fig4_cpu_utilization.{pdf,png}")
    print("  fig5_scaling_heatmap.{pdf,png}")
    print("  fig6_speedup_bar.{pdf,png}")


if __name__ == '__main__':
    main()
