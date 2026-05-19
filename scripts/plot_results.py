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