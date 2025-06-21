import os
import matplotlib.pyplot as plt
import numpy as np

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "output_result/sequential/1D")
plot_dir = os.path.join(script_dir, "..", "plot")
os.makedirs(plot_dir, exist_ok=True)

# Read (N, time) from a file
def read_fft_timings(file_path):
    timings = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                n = int(parts[0])
                t = float(parts[1])
                timings.append((n, t))
    return timings

# File paths
iterative_file = os.path.join(input_dir, "iterative_fft_timings.txt")
recursive_file = os.path.join(input_dir, "recursive_fft_timings.txt")

# Read timing data
iterative_timings = read_fft_timings(iterative_file)
recursive_timings = read_fft_timings(recursive_file)

# Unpack
sizes_iter, times_iter = zip(*iterative_timings)
sizes_rec, times_rec = zip(*recursive_timings)

# Reference line: O(N log N)
sizes = np.array(sorted(set(sizes_iter + sizes_rec)))
nlogn = sizes * np.log2(sizes) / 10000  # Scaled for visibility

# Print summary
def print_summary(name, sizes, times):
    print(f"{name} FFT:")
    print(f"  Max N: {max(sizes)}")
    print(f"  Avg Time: {sum(times)/len(times):.6f} s")
    print(f"  Total Time: {sum(times):.6f} s\n")

print_summary("Iterative", sizes_iter, times_iter)
print_summary("Recursive", sizes_rec, times_rec)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sizes_iter, times_iter, label="Iterative FFT", marker='o')
plt.plot(sizes_rec, times_rec, label="Recursive FFT", marker='s')
plt.plot(sizes, nlogn, label="O(N log N)", linestyle="--", color="black")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Problem Size (N)")
plt.ylabel("Time (seconds)")
plt.title("FFT Timing Comparison: Iterative vs Recursive (Log-Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save plot
plot_path = os.path.join(plot_dir, "fft_iterative_vs_recursive_loglog.pdf")
plt.savefig(plot_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Plot saved to: {plot_path}")
