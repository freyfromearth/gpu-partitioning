from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = Path("results/results.csv")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")
    
df = pd.read_csv(RESULTS_PATH)

df = df[df["repeat"] > 0].copy()

summary = (df.groupby(["mode", "n", "bits", "partitions", "distribution"], as_index=False).agg(
    mean_time_ms=("time_ms", "mean"),
    std_time_ms=("time_ms", "std"),
    mean_throughput_mtuples_s=("throughput_mtuples_s", "mean"),
    std_throughput_mtuples_s=("throughput_mtuples_s", "std")
))

summary.to_csv("results/summary.csv", index=False)
print(summary)

timing_columns = [
    "alloc_ms",
    "h2d_input_ms",
    "count_kernel_ms",
    "d2h_counts_ms",
    "cpu_prefix_ms",
    "h2d_offsets_ms",
    "scatter_kernel_ms",
    "d2h_output_ms",
    "free_ms"
]

timing_summary = (df[df["mode"] == "gpu"].groupby(["mode", "n", "bits", "partitions", "distribution"], as_index=False).agg({
    "alloc_ms": "mean",
    "h2d_input_ms": "mean",
    "count_kernel_ms": "mean",
    "d2h_counts_ms": "mean",
    "cpu_prefix_ms": "mean",
    "h2d_offsets_ms": "mean",
    "scatter_kernel_ms": "mean",
    "d2h_output_ms": "mean",
    "free_ms": "mean"
}))

timing_summary.to_csv("results/timing_summary.csv", index=False)

# plot runtime vs input size, fixed bits=8
bits_to_plot = 8
dist = "uniform"

plot_df = summary[(summary["bits"] == bits_to_plot) & (summary["distribution"] == dist)].copy()

plt.figure()
for mode in sorted(plot_df["mode"].unique()):
    sub = plot_df[plot_df["mode"] == mode].sort_values("n")
    plt.errorbar(
        sub["n"],
        sub["mean_time_ms"],
        yerr=sub["std_time_ms"],
        marker="o",
        capsize=3,
        label=mode.upper()
    )
    
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Input size, tuples")
plt.ylabel("Runtime, ms")
plt.title(f"Runtime vs input size ({2 ** bits_to_plot} partitions)")
plt.legend()
save_plot(FIGURE_DIR / "runtime_vs_input_size.png")

# plot throughtput vs input size, fixed bits=8
plt.figure()
for mode in sorted(plot_df["mode"].unique()):
    sub = plot_df[plot_df["mode"] == mode].sort_values("n")
    plt.errorbar(
        sub["n"],
        sub["mean_throughput_mtuples_s"],
        yerr=sub["std_throughput_mtuples_s"],
        marker="o",
        capsize=3,
        label=mode.upper(),
    )
    
plt.xscale("log")
plt.xlabel("Input size, tuples")
plt.ylabel("Throughput, million tuples/s")
plt.title(f"Throughput vs input size ({2 ** bits_to_plot} partitions)")
plt.legend()
save_plot(FIGURE_DIR / "throughput_vs_input_size.png")

# plot runtime vs number of partitions, fixed n=10 million if present
n_to_plot = 10_000_000

plot_df = summary[(summary["n"] == n_to_plot) & (summary["distribution"] == dist)].copy()

plt.figure()
for mode in sorted(plot_df["mode"].unique()):
    sub = plot_df[plot_df["mode"] == mode].sort_values("partitions")
    plt.errorbar(
        sub["partitions"],
        sub["mean_time_ms"],
        yerr=sub["std_time_ms"],
        marker="o",
        capsize=3,
        label=mode.upper(),
    )
    
plt.xscale("log", base=2)
plt.xlabel("Number of partitions")
plt.ylabel("Runtime, ms")
plt.title(f"Runtime vs partition count (n={n_to_plot:,})")
plt.legend()
save_plot(FIGURE_DIR / "runtime_vs_partitions.png")

# plot GPU speedup over CPU for input-size scaling
cpu = plot_df[plot_df["mode"] == "cpu"][["bits", "partitions", "mean_time_ms"]]
gpu = plot_df[plot_df["mode"] == "gpu"][["bits", "partitions", "mean_time_ms"]]

speedup_partitions = cpu.merge(gpu, on=["bits", "partitions"], suffixes=("_cpu", "_gpu"))

if not speedup_partitions.empty:
    speedup_partitions["speedup"] = (speedup_partitions["mean_time_ms_cpu"] / speedup_partitions["mean_time_ms_gpu"])
    
    plt.figure()
    plt.plot(speedup_partitions["partitions"], speedup_partitions["speedup"], marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.xscale("log", base=2)
    plt.xlabel("Number of partitions")
    plt.ylabel("Speedup, CPU time / GPU time")
    plt.title(f"GPU speedup vs partition count (n={n_to_plot:,})")
    save_plot(FIGURE_DIR / "speedup_vs_partitions.png")
    
# plot input size speedup at bits=8
cpu = summary[
    (summary["bits"] == bits_to_plot) & (summary["distribution"] == dist) & (summary["mode"] == "cpu")][["n", "mean_time_ms"]]

gpu = summary[
    (summary["bits"] == bits_to_plot) & (summary["distribution"] == dist) & (summary["mode"] == "gpu")][["n", "mean_time_ms"]]

speedup_input = cpu.merge(gpu, on="n", suffixes=("_cpu", "_gpu"))

if not speedup_input.empty:
    speedup_input["speedup"] = (speedup_input["mean_time_ms_cpu"] / speedup_input["mean_time_ms_gpu"])
    
    plt.figure()
    plt.plot(speedup_input["n"], speedup_input["speedup"], marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.xscale("log")
    plt.xlabel("Input size, tuples")
    plt.ylabel("Speedup, CPU time / GPU time")
    plt.title(f"GPU speedup vs input size ({2 ** bits_to_plot} partitions)")
    save_plot(FIGURE_DIR / "speedup_vs_input_size.png")
    
# plot stacked GPU timing breakdown for one representative case
n_case = 10_000_000
bits_case = 8
dist_case = "uniform"

case_df = timing_summary[(timing_summary["mode"] == "gpu")
                         & (timing_summary["n"] == n_case)
                         & (timing_summary["bits"] == bits_case)
                         & (timing_summary["distribution"] == dist_case)
]

if not case_df.empty:
    row = case_df.iloc[0]

    phase_labels = [
        "GPU mem allocation",
        "Copy input to GPU",
        "GPU count phase",
        "Copy counts to CPU",
        "CPU compute offsets",
        "Copy offsets to GPU",
        "GPU scatter phase",
        "Copy output to CPU",
        "GPU mem cleanup"
    ]
    
    phase_vals = [
        row["alloc_ms"],
        row["h2d_input_ms"],
        row["count_kernel_ms"],
        row["d2h_counts_ms"],
        row["cpu_prefix_ms"],
        row["h2d_offsets_ms"],
        row["scatter_kernel_ms"],
        row["d2h_output_ms"],
        row["free_ms"]
    ]
    
    plt.figure()
    bottom = 0.0
    
    for label, val in zip(phase_labels, phase_vals):
        plt.bar(["GPU pipeline"], [val], bottom=bottom, label=label)
        bottom += val
    
    plt.ylabel("Time, ms")
    plt.title(f"GPU timing breakdown (n={n_case:,}, {2 ** bits_case} partitions)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gpu_timing_breakdown_case.png", dpi=200)
    plt.close()
    
    print(f"Saved {FIGURE_DIR / 'gpu_timing_breakdown_case.png'}")
    
# plot GPU phase scaling vs input size for bits = 8
bits_case = 8
dist_case = "uniform"

scaling_df = timing_summary[(timing_summary["mode"] == "gpu")
                         & (timing_summary["bits"] == bits_case)
                         & (timing_summary["distribution"] == dist_case)
].sort_values("n")

if not scaling_df.empty:
    plt.figure()
    
    phase_display_labels = {
        "h2d_input_ms": "Copy input to GPU",
        "count_kernel_ms": "GPU count phase",
        "scatter_kernel_ms": "GPU scatter phase",
        "d2h_output_ms": "Copy output to CPU"
    }
    
    phases_to_plot = [
        "h2d_input_ms",
        "count_kernel_ms",
        "scatter_kernel_ms",
        "d2h_output_ms"
    ]
    
    for phase in phases_to_plot:
        plt.plot(scaling_df["n"],scaling_df[phase],marker="o",label=phase_display_labels[phase])
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Input size, tuples")
    plt.ylabel("Time, ms")
    plt.title(f"GPU phase scaling vs input size ({2 ** bits_case} partitions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gpu_phase_scaling_vs_input_size.png", dpi=200)
    plt.close()
    
    print(f"Saved {FIGURE_DIR / 'gpu_phase_scaling_vs_input_size.png'}")