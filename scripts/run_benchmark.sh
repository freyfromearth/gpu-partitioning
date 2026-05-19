#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

OUT="results/results.csv"

echo "mode,n,bits,partitions,distribution,repeat,time_ms,throughput_mtuples_s" > "$OUT"

# we use 6 repeats because repeat 0 is treated as a gpu warmup in the plotting script
REPEATS=6

# experiment A: input size scaling with fixed partition count
NS=(100000 1000000 10000000 50000000)
BITS_INPUT_SCALING=(8)

# experiment B: partition count scaling with fixed input size
N_PARTITION_SCALING=10000000
BITS_PARTITION_SCALING=(4 8 12 16)

DISTS=("uniform")

echo "--- Experiment A: input-size scaling ---" >&2
for dist in "${DISTS[@]}"; do
    for n in "${NS[@]}"; do
        for bits in "${BITS_INPUT_SCALING[@]}"; do
            echo "CPU n=$n bits=$bits dist=$dist" >&2
            ./bench_partition cpu "$n" "$bits" "$REPEATS" "$dist" >> "$OUT"

            echo "GPU n=$n bits=$bits dist=$dist" >&2
            ./bench_partition gpu "$n" "$bits" "$REPEATS" "$dist" >> "$OUT"
        done
    done
done

echo "--- Experiment B: partition-count scaling ---" >&2
for dist in "${DISTS[@]}"; do
    for bits in "${BITS_PARTITION_SCALING[@]}"; do
        echo "CPU n=$N_PARTITION_SCALING bits=$bits dist=$dist" >&2
        ./bench_partition cpu "$N_PARTITION_SCALING" "$bits" "$REPEATS" "$dist" >> "$OUT"

        echo "GPU n=$N_PARTITION_SCALING bits=$bits dist=$dist" >&2
        ./bench_partition gpu "$N_PARTITION_SCALING" "$bits" "$REPEATS" "$dist" >> "$OUT"
    done
done

echo "Wrote $OUT" >&2