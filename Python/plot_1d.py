import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# === FFT 1D ===
def plot_fft_result(frequencies, fft_result, title="FFT Magnitude", xlabel="Frequency", ylabel="Magnitude", output_file="../Plot_result/fft_plot.png"):
    magnitudes = np.abs(fft_result)
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, magnitudes, label="FFT Magnitude")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")
    plt.close()


def load_fft_data(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=complex)
    num_points = len(data)
    frequencies = np.fft.fftfreq(num_points)
    return frequencies, data


# === SVD ===
def load_singular_values(filename):
    return np.loadtxt(filename)


def plot_singular_values(singular_values, title="Singular Values", xlabel="Index", ylabel="Value", output_file="../Plot_result/singular_values.png"):
    indices = np.arange(len(singular_values))
    base_name = os.path.splitext(output_file)[0]

    # Linear scale
    plt.figure(figsize=(10, 6))
    plt.plot(indices, singular_values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")
    plt.close()

    # Log scale
    plt.figure(figsize=(10, 6))
    plt.semilogy(indices, singular_values, marker='o', linestyle='-')
    plt.title(title + " (Log Scale)")
    plt.xlabel(xlabel)
    plt.ylabel("Log(" + ylabel + ")")
    plt.grid(True)
    plt.tight_layout()
    log_path = base_name + "_log.png"
    plt.savefig(log_path)
    print(f"Graph saved as {log_path}")
    plt.close()

    # Normalized
    normalized = singular_values / singular_values[0] if singular_values[0] != 0 else singular_values
    plt.figure(figsize=(10, 6))
    plt.plot(indices, normalized, marker='o', linestyle='-')
    plt.title(title + " (Normalized)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel + " (Normalized)")
    plt.grid(True)
    plt.tight_layout()
    norm_path = base_name + "_normalized.png"
    plt.savefig(norm_path)
    print(f"Graph saved as {norm_path}")
    plt.close()


# === ERROR PLOTS ===
def load_error_vs_threshold(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    thresholds = data[:, 0]
    errors = data[:, 1]
    return thresholds, errors


def plot_error_vs_threshold(thresholds, errors, title="Reconstruction Error vs Threshold", xlabel="Threshold", ylabel="Reconstruction Error", output_file="../Plot_result/Error_vs_Threshold.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, errors, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")
    plt.close()


def plot_error_comparison(percentages, magnitude_errors, band_errors, title="Comparison of Reconstruction Error", output_file="../Plot_result/error_comparison.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, magnitude_errors, 'o-', label="Magnitude Thresholding", color='blue')
    plt.plot(percentages, band_errors, 's-', label="Bandpass Thresholding", color='green')
    plt.title(title)
    plt.xlabel("Threshold Percentage (%)")
    plt.ylabel("Reconstruction Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")
    plt.close()


# === PSNR PLOTS ===
def load_error_and_psnr(csv_file):
    df = pd.read_csv(csv_file)
    x = df.iloc[:, 0].to_numpy()
    err = df.iloc[:, 1].to_numpy()
    if df.shape[1] > 2:
        psnr = df.iloc[:, 2].to_numpy()
    else:
        # If PSNR not precomputed, compute it here
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / (err + 1e-8))  # avoid division by zero
    return x, err, psnr


def plot_psnr_vs_threshold(x, psnr, title="PSNR vs Threshold", xlabel="Threshold", ylabel="PSNR (dB)", output_file="../Plot_result/psnr_plot.png"):
    plt.figure()
    plt.plot(x, psnr, marker='o', linestyle='-', color='tab:blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Graph saved as {output_file}")
    plt.close()


def plot_psnr_comparison(x, psnr_mag, psnr_band, title="PSNR Comparison: Magnitude vs Band", output_file="../Plot_result/psnr_comparison.png"):
    plt.figure()
    plt.plot(x, psnr_mag, marker='o', label='Magnitude', color='tab:blue')
    plt.plot(x, psnr_band, marker='s', label='Band', color='tab:orange')
    plt.title(title)
    plt.xlabel("Threshold / Percentile")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Graph saved as {output_file}")
    plt.close()