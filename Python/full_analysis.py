import numpy as np

import plot_1d
import plot_2d
import os


# --- Output directory ---
output_dir = "../output_result/Plot_result"
csv_dir = "../output_result/csv_output"
os.makedirs(output_dir, exist_ok=True)

# === SVD ===
print("Plotting Singular Values...")
singular_values = plot_1d.load_singular_values(f"{csv_dir}/singular_values.csv")
plot_1d.plot_singular_values(
    singular_values,
    title="Singular Values Decay",
    xlabel="Index",
    ylabel="Singular Value",
    output_file=os.path.join(output_dir, "singular_values.png")
)

# === FFT ORIGINALE ===
print("Plotting FFT Magnitude (2D)...")
fft_result_2d = plot_2d.load_fft_data_2d(f"{csv_dir}/fft_output_2d.csv")
plot_2d.plot_fft_result_2d(
    fft_result_2d,
    sampling_rate=1000,
    title="FFT Magnitude (2D)",
    output_file=os.path.join(output_dir, "fft_magnitude_2d.png")
)

# === PERCENTUALI COMUNI ===
percentages = [1, 5, 10, 25, 40, 50, 60, 75, 85, 90, 95, 99]

# === MAGNITUDE-FILTERED ===
print("Plotting Filtered FFT Magnitude (Magnitude Threshold)...")
for p in percentages:
    filename = f"{csv_dir}/fft_magnitude_filtered_{p}p_magnitude.csv"
    if os.path.exists(filename):
        filtered = plot_2d.load_magnitude_data_2d(filename)

        plot_2d.plot_magnitude_2d_log(
            filtered,
            title=f"Filtered FFT (Magnitude) - Log - {p}%",
            output_file=os.path.join(output_dir, f"filtered_fft_magnitude_{p}p_log.png")
        )


# === BAND-FILTERED ===
print("Plotting Filtered FFT Magnitude (Band Threshold)...")
for p in percentages:
    filename = f"{csv_dir}/fft_magnitude_filtered_{p}p_band.csv"
    if os.path.exists(filename):
        filtered = plot_2d.load_magnitude_data_2d(filename)

        plot_2d.plot_magnitude_2d_log(
            filtered,
            title=f"Filtered FFT (Band) - Log - {p}%",
            output_file=os.path.join(output_dir, f"filtered_fft_band_{p}p_log.png")
        )

# === ERROR PLOT ===
print("Plotting Reconstruction Error vs Thresholds...")

def try_plot_error(csv_name, title, image_name, xlabel):
    path = f"{csv_dir}/{csv_name}"
    if os.path.exists(path):
        x, y = plot_1d.load_error_vs_threshold(path)
        plot_1d.plot_error_vs_threshold(
            x, y,
            title=title,
            xlabel=xlabel,
            ylabel="Reconstruction Error",
            output_file=os.path.join(output_dir, image_name)
        )

try_plot_error("error_vs_threshold_magnitude_percentage.csv", "Error vs Threshold (Magnitude)", "error_vs_threshold_magnitude.png", "Threshold / Percentile")
try_plot_error("error_vs_threshold_band_percentage.csv", "Error vs Threshold (Band)", "error_vs_threshold_band.png", "Band Percentage")

# === COMPARAZIONE ERRORI MAGNITUDE vs BAND ===
print("Plotting Error Comparison between Magnitude and Band Thresholding...")

mag_path = f"{csv_dir}/error_vs_threshold_magnitude_percentage.csv"
band_path = f"{csv_dir}/error_vs_threshold_band_percentage.csv"

if os.path.exists(mag_path) and os.path.exists(band_path):
    x_mag, err_mag = plot_1d.load_error_vs_threshold(mag_path)
    x_band, err_band = plot_1d.load_error_vs_threshold(band_path)

    # Optional: check if x_mag and x_band are the same (or very close)
    if not np.allclose(x_mag, x_band):
        print("Warning: Threshold arrays differ between magnitude and band!")

    # Plot comparison
    plot_1d.plot_error_comparison(
        x_mag,
        err_mag,
        err_band,
        title="Comparison of Reconstruction Error",
        output_file=os.path.join(output_dir, "error_comparison.png")
    )

# === PSNR PLOT: MAGNITUDE ===
print("Plotting PSNR vs Threshold (Magnitude)...")

def try_plot_psnr(csv_name, title, image_name, xlabel):
    path = f"{csv_dir}/{csv_name}"
    if os.path.exists(path):
        x, _, psnr = plot_1d.load_error_and_psnr(path)
        plot_1d.plot_psnr_vs_threshold(
            x,
            psnr,
            title=title,
            xlabel=xlabel,
            ylabel="PSNR (dB)",
            output_file=os.path.join(output_dir, image_name)
        )

try_plot_psnr("error_vs_threshold_magnitude_percentage.csv", "PSNR vs Threshold (Magnitude)", "psnr_magnitude.png", "Threshold / Percentile")

# === PSNR PLOT: BAND ===
print("Plotting PSNR vs Threshold (Band)...")
try_plot_psnr("error_vs_threshold_band_percentage.csv", "PSNR vs Threshold (Band)", "psnr_band.png", "Threshold / Percentile")

# === PSNR COMPARISON PLOT ===
print("Plotting PSNR Comparison between Magnitude and Band...")

mag_path = f"{csv_dir}/error_vs_threshold_magnitude_percentage.csv"
band_path = f"{csv_dir}/error_vs_threshold_band_percentage.csv"

if os.path.exists(mag_path) and os.path.exists(band_path):
    x_mag, _, psnr_mag = plot_1d.load_error_and_psnr(mag_path)
    x_band, _, psnr_band = plot_1d.load_error_and_psnr(band_path)

    if not np.allclose(x_mag, x_band):
        print("Warning: Percentages do not match exactly between magnitude and band!")

    plot_1d.plot_psnr_comparison(
        x_mag,
        psnr_mag,
        psnr_band,
        title="PSNR Comparison: Magnitude vs Band",
        output_file=os.path.join(output_dir, "psnr_comparison.png")
    )

print("\nAll plots completed and saved in '../OUTPUT_RESULT/Plot_result' folder!")