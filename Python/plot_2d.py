import numpy as np
import matplotlib.pyplot as plt
import os

def plot_fft_result_2d(fft_result, sampling_rate, title="FFT Visualization (2D)", output_file="fft_plot_2d.png"):
    """
    Plots and saves the 2D FFT magnitude spectrum with correct frequency scaling.
    
    Args:
        fft_result (np.array): 2D FFT complex output.
        sampling_rate (float): Sampling rate in Hz.
        title (str): Title of the plot.
        output_file (str): Path to save the output plot.
    """
    # Shift the FFT result to center the low frequencies
    fft_result_shifted = np.fft.fftshift(fft_result)
    
    # Compute magnitude spectrum with log scaling
    magnitudes = np.log1p(np.abs(fft_result_shifted))

    # Get image dimensions
    Ny, Nx = fft_result.shape  # Rows = Y, Columns = X

    # Compute frequency axes in Hz
    freq_x = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0/sampling_rate))  # X-axis frequency
    freq_y = np.fft.fftshift(np.fft.fftfreq(Ny, d=1.0/sampling_rate))  # Y-axis frequency

    # Plot the FFT magnitude
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitudes, extent=[freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]], origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="Magnitude")
    
    # Labels and grid
    plt.title(title)
    plt.xlabel("Frequency X (Hz)")
    plt.ylabel("Frequency Y (Hz)")
    plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")

def load_fft_data_2d(filename):
    """
    Carica i dati FFT 2D esportati dal file CSV.

    Args:
        filename (str): Nome del file CSV.

    Returns:
        np.array: Risultati complessi della FFT 2D.
    """
    with open(filename, 'r') as file:
        data = file.readlines()

    rows = []
    for i, line in enumerate(data):
        row = []
        for j, val in enumerate(line.strip().split(',')):
            try:
                row.append(complex(val.strip()))  # Rimuove spazi extra e converte
            except ValueError:
                print(f"Valore malformato: '{val.strip()}' nella riga {i + 1}, colonna {j + 1}")
                row.append(0 + 0j)  # Sostituisce il valore malformato con 0+0j
        rows.append(row)

    return np.array(rows, dtype=complex)



# # Debug: Verifica la directory corrente
# print("Current working directory:", os.getcwd())
#
# # Percorso al file esportato dal C++
# filename = "../CUDA_FFT/2D/fft_output_2d.csv"
# 
# # Carica i dati
# fft_result_2d = load_fft_data_2d(filename)
#
# print("FFT Data Shape:", fft_result_2d.shape)
# print("Max Magnitude:", np.max(np.abs(fft_result_2d)))
# print("Min Magnitude:", np.min(np.abs(fft_result_2d)))
# print("Sample Data (first 5x5 values):")
# print(np.abs(fft_result_2d[:5, :5]))
#
#
# # Percorso per salvare il grafico
# output_file_path = "../Plot_result/fft_plot_2d.png"
#
# sampling_rate = 1000  # Hz (Adjust this based on your actual image sampling rate)
# plot_fft_result_2d(fft_result_2d, sampling_rate, title="FFT Visualization (2D)", output_file=output_file_path)


def load_magnitude_data_2d(filename):
    """
    Carica i dati della magnitudo FFT 2D da CSV (solo valori reali).

    Args:
        filename (str): Nome del file CSV.

    Returns:
        np.array: Magnitudine FFT 2D.
    """
    return np.loadtxt(filename, delimiter=',')

def plot_magnitude_2d(magnitude, title="Magnitude Spectrum (2D)", output_file="magnitude_2d.png"):
    """
    Plotta uno spettro di magnitudo 2D salvato come solo numeri reali.

    Args:
        magnitude (np.array): Magnitudine FFT 2D.
        title (str): Titolo del grafico.
        output_file (str): Path per salvare il grafico.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label="Magnitude")
    plt.title(title)
    plt.xlabel("X Frequency Bin")
    plt.ylabel("Y Frequency Bin")
    plt.grid(visible=False)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")
    plt.close()

def plot_magnitude_2d_log(magnitude, title="Log-Scaled Magnitude Spectrum (2D)", output_file="magnitude_2d_log.png"):
    """
    Plotta uno spettro di magnitudo 2D applicando scala logaritmica,
    utile per visualizzare valori molto piccoli dopo filtraggio.

    Args:
        magnitude (np.array): Magnitudine FFT 2D (reale).
        title (str): Titolo del grafico.
        output_file (str): Path per salvare il grafico.
    """
    magnitude_log = np.log1p(magnitude)

    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude_log, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label="Log(1 + Magnitude)")
    plt.title(title)
    plt.xlabel("X Frequency Bin")
    plt.ylabel("Y Frequency Bin")
    plt.grid(visible=False)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Log-scaled magnitude graph saved as {output_file}")
    plt.close()