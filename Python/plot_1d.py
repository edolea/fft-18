import os

import numpy as np

import matplotlib.pyplot as plt

def plot_fft_result(frequencies, fft_result, title="FFT Magnitude", xlabel="Frequency", ylabel="Magnitude", output_file="../Plot_result/fft_plot.png"):
    """
    Plotta e salva i risultati della FFT.

    Args:
        frequencies (list or np.array): Array con le frequenze corrispondenti.
        fft_result (list or np.array): Risultati complessi della FFT.
        title (str): Titolo del grafico.
        xlabel (str): Etichetta dell'asse x.
        ylabel (str): Etichetta dell'asse y.
        output_file (str): Nome del file per salvare il grafico (es. 'output.png').
    """
    # Calcolo della magnitudine
    magnitudes = np.abs(fft_result)

    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, magnitudes, label="FFT Magnitude")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Salva il grafico su file
    plt.savefig(output_file)
    print(f"Graph saved as {output_file}")
# Carica i dati esportati da C++
def load_fft_data(filename):
    """
    Carica i dati FFT esportati dal file CSV.

    Args:
        filename (str): Nome del file CSV.

    Returns:
        tuple: Frequenze e valori FFT complessi.
    """
    data = np.genfromtxt(filename, delimiter=',', dtype=complex)
    num_points = len(data)
    frequencies = np.fft.fftfreq(num_points)  # Frequenze normalizzate
    return frequencies, data


def load_singular_values(filename):
    """
    Carica i valori singolari da un file CSV (un valore per riga).

    Args:
        filename (str): Path del file CSV.

    Returns:
        np.array: Array di valori singolari.
    """
    data = np.loadtxt(filename)
    return data

def plot_singular_values(singular_values, title="Singular Values", xlabel="Index", ylabel="Value", output_file="../Plot_result/singular_values.png"):
    """
    Plotta i valori singolari in scala lineare, logaritmica e normalizzata, e salva i grafici.

    Args:
        singular_values (np.array): Array di valori singolari.
        title (str): Titolo del grafico.
        xlabel (str): Etichetta asse x.
        ylabel (str): Etichetta asse y.
        output_file (str): Path per salvare il grafico principale.
    """
    indices = np.arange(len(singular_values))
    base_name = os.path.splitext(output_file)[0]

    # Plot originale
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

    # Plot log-scale
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

    # Plot normalizzato
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
def load_error_vs_threshold(filename):
    """
    Carica i dati di errore vs soglia da un file CSV.

    Args:
        filename (str): Percorso al file CSV.

    Returns:
        tuple: thresholds (np.array), errors (np.array)
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    thresholds = data[:, 0]
    errors = data[:, 1]
    return thresholds, errors

def plot_error_vs_threshold(thresholds, errors, title="Reconstruction Error vs Threshold", xlabel="Threshold", ylabel="Reconstruction Error", output_file="../Plot_result/Error_vs_Threshold.png"):
    """
    Plotta l'errore di ricostruzione rispetto alla soglia.

    Args:
        thresholds (np.array): Valori di soglia.
        errors (np.array): Errori corrispondenti.
        title (str): Titolo del grafico.
        xlabel (str): Etichetta asse x.
        ylabel (str): Etichetta asse y.
        output_file (str, opzionale): Path per salvare il grafico.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, errors, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Graph saved as {output_file}")
    plt.close()

def plot_error_comparison(percentages, magnitude_errors, band_errors, title="Comparison of Reconstruction Error", output_file="../Plot_result/error_comparison.png"):
    """
    Plotta e salva il confronto tra gli errori di ricostruzione di due metodi: magnitude e bandpass.

    Args:
        percentages (list or np.array): Percentuali di soglia applicate (es. [1, 5, 10, ...]).
        magnitude_errors (list or np.array): Errori corrispondenti al filtraggio per magnitudine.
        band_errors (list or np.array): Errori corrispondenti al filtraggio per banda.
        title (str): Titolo del grafico.
        output_file (str): Path dove salvare il grafico.
    """
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