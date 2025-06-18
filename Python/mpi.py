import os
import matplotlib.pyplot as plt

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
directories = [
    os.path.join(script_dir, "..", "OUTPUT_RESULT", "mpi", "1D"),
    os.path.join(script_dir, "..", "OUTPUT_RESULT", "mpi", "1D_inverse"),
    os.path.join(script_dir, "..", "OUTPUT_RESULT", "mpi", "2D"),
    os.path.join(script_dir, "..", "OUTPUT_RESULT", "mpi", "2D_inverse"),
]
plot_dir = os.path.join(script_dir, "..", "PLOT", "mpi")
os.makedirs(plot_dir, exist_ok=True)

# Function to read timings and calculate speedup from a file
def read_timings_and_speedup(file_path):
    iterative_times = []
    mpi_times = []
    speedups = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].isdigit():
                n = int(parts[0])
                sequential_time = float(parts[1])
                mpi_time = float(parts[2])
                speedup = sequential_time / mpi_time if mpi_time > 0 else 0
                iterative_times.append((n, sequential_time))
                mpi_times.append((n, mpi_time))
                speedups.append((n, speedup))
    return iterative_times, mpi_times, speedups

# Generate plots for each directory
for directory in directories:
    all_iterative_times = []
    mpi_timings = {}
    speedup_data = {}
    mpi_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    for mpi_file in mpi_files:
        file_path = os.path.join(directory, mpi_file)
        iterative_times, mpi_times, speedups = read_timings_and_speedup(file_path)
        if not all_iterative_times:
            all_iterative_times = iterative_times  # Use iterative times from the first file
        mpi_timings[mpi_file] = mpi_times
        speedup_data[mpi_file] = speedups

    # Unpack iterative times
    if all_iterative_times:
        sizes_iter, times_iter = zip(*all_iterative_times)
    else:
        raise ValueError(f"No valid iterative timings found in {directory}.")

    ## Plot size vs time
    plt.figure(figsize=(12, 8))
    plt.plot(sizes_iter, times_iter, label="Iterative FFT", marker='o', color="blue")
    for label, timings in mpi_timings.items():
        if timings:  # Ensure data is not empty
            sizes_mpi, times_mpi = zip(*timings)
            plt.plot(sizes_mpi, times_mpi, label=label.replace("_", " ").replace(".txt", ""), marker='s')

    # for now leave them in normal scale
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Time (seconds)")
    plt.title(f"Problem Size vs Time: {os.path.basename(directory)}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plot_path_time = os.path.join(plot_dir, f"{os.path.basename(directory)}_size_vs_time_comparison.pdf")
    plt.savefig(plot_path_time, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Time plot saved to: {plot_path_time}")


    ## Plot size vs speedup
    plt.figure(figsize=(12, 8))
    for label, speedups in speedup_data.items():
        if speedups:  # Ensure data is not empty
            sizes_speedup, values_speedup = zip(*speedups)
            plt.plot(sizes_speedup, values_speedup, label=label.replace("_", " ").replace(".txt", ""), marker='s')

    # for now leave them in normal scale
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Speedup")
    plt.title(f"Problem Size vs Speedup: {os.path.basename(directory)}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plot_path_speedup = os.path.join(plot_dir, f"{os.path.basename(directory)}_size_vs_speedup_comparison.pdf")
    plt.savefig(plot_path_speedup, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Speedup plot saved to: {plot_path_speedup}")