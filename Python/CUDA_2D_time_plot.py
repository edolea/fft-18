import os
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

# ---------------------
# Path Setup
# ---------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
seq_input_dir = os.path.join(script_dir, "..", "output_result/sequential/2D")
cuda_input_dir = os.path.join(script_dir, "..", "output_result/CUDA/2D")
plot_dir = os.path.join(script_dir, "..", "plot")
os.makedirs(plot_dir, exist_ok=True)

# ---------------------
# Read Sequential FFT Timings
# ---------------------
def read_seq_timings(file_path):
    timings = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                fft_size = int(parts[0])
                time = float(parts[1])
                if fft_size > 1:  # Exclude N = 1
                    timings[fft_size] = time
    return timings

iterative_file = os.path.join(seq_input_dir, "iterative_fft_timings.txt")
recursive_file = os.path.join(seq_input_dir, "recursive_fft_timings.txt")

iterative_timings = read_seq_timings(iterative_file)
recursive_timings = read_seq_timings(recursive_file)

# ---------------------
# Read CUDA FFT Timings
# ---------------------
cuda_timings_by_thread = defaultdict(dict)

def read_cuda_timings(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                fft_size = int(parts[0])
                time = float(parts[1])
                thread_count = int(parts[2])
                if fft_size > 1:
                    cuda_timings_by_thread[thread_count][fft_size] = time

timing_files = sorted([
    f for f in os.listdir(cuda_input_dir)
    if f.startswith("timings_parallel_fft_") and f.endswith(".txt")
])

for filename in timing_files:
    read_cuda_timings(os.path.join(cuda_input_dir, filename))

# ---------------------
# Plot Setup
# ---------------------
fig, ax_time = plt.subplots(figsize=(10, 6))
ax_speedup = ax_time.twinx()

# All FFT sizes in play
all_fft_sizes = sorted(set(iterative_timings.keys()))
for recs in cuda_timings_by_thread.values():
    all_fft_sizes.extend(recs.keys())
all_fft_sizes = sorted(set(all_fft_sizes) - {1})  # Exclude N = 1

# Reference line: O(N log N)
n_arr = np.array(all_fft_sizes)
nlogn = n_arr * np.log2(n_arr) / 10000
#ax_time.plot(n_arr, nlogn, linestyle="--", color="black", label="O(N log N)")

# Plot Sequential
sizes_iter, times_iter = zip(*sorted(iterative_timings.items()))
sizes_rec, times_rec = zip(*sorted(recursive_timings.items()))

ax_time.plot(sizes_iter, times_iter, marker='o', color="blue", label="Iterative FFT")
ax_time.plot(sizes_rec, times_rec, marker='s', color="orange", label="Recursive FFT")

# Plot CUDA lines and speedups
colors = plt.cm.viridis(np.linspace(0, 1, len(cuda_timings_by_thread)))
for idx, (thread_count, records) in enumerate(sorted(cuda_timings_by_thread.items())):
    sorted_records = sorted((k, v) for k, v in records.items() if k in iterative_timings)
    if not sorted_records:
        continue
    fft_sizes, times_cuda = zip(*sorted_records)
    times_seq = [iterative_timings[n] for n in fft_sizes]
    speedups = [ts / tc for ts, tc in zip(times_seq, times_cuda)]

    # Time line (right Y-axis)
    ax_time.plot(fft_sizes, times_cuda, label=f"{thread_count} threads", marker='o', color=colors[idx])

    # Speedup line (left Y-axis)
    ax_speedup.plot(fft_sizes, speedups, linestyle='--', marker='x', color=colors[idx], alpha=0.6)

# Axes config
ax_time.set_xscale('log')
ax_time.set_yscale('log')
ax_time.set_xlabel("Problem Size (N)")
ax_time.set_ylabel("Execution Time (seconds)")
ax_speedup.set_ylabel("Speedup vs Iterative FFT", color="gray")

ax_time.set_title("FFT Timings and Speedups (Log-Log Time Axis)")
ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
ax_time.legend(loc='upper left', fontsize=9, title="Implementation / Threads")

# Save
plot_path = os.path.join(plot_dir, "2D_fft_combined_with_speedup_loglog.pdf")
plt.savefig(plot_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Final plot with speedup saved to: {plot_path}")
