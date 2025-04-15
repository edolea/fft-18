#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "signal_generator.h"
#include "fft_transform.h"
#include "optimized_fft.h"
#include "dwt_transform.h"
#include "signal_visualizer.h"

// Function to benchmark and compare transform performance
void benchmarkCompare(TransformBase& t1, TransformBase& t2, const Signal& signal, int iterations = 20) {
    std::cout << "Comparing " << t1.getName() << " vs " << t2.getName() << std::endl;

    // Warmup
    t1.forward(signal);
    t2.forward(signal);

    // Benchmark t1
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        t1.forward(signal);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t1_time = end - start;

    // Benchmark t2
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        t2.forward(signal);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t2_time = end - start;

    // Results
    double avg_t1 = t1_time.count() / iterations;
    double avg_t2 = t2_time.count() / iterations;
    double speedup = avg_t1 / avg_t2;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << t1.getName() << ": " << avg_t1 << " ms" << std::endl;
    std::cout << t2.getName() << ": " << avg_t2 << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Validate results match
    Signal result1 = t1.forward(signal);
    Signal result2 = t2.forward(signal);

    double max_diff = 0.0;
    for (size_t i = 0; i < result1.size(); ++i) {
        double diff = std::abs(std::abs(result1[i]) - std::abs(result2[i]));
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Maximum difference in results: " << max_diff << std::endl;
    std::cout << (max_diff < 1e-10 ? "Results match!" : "WARNING: Results differ!") << std::endl;
    std::cout << std::endl;
}

int main() {
    // Create a signal generator
    SignalGenerator generator;

    // Generate some test signals
    size_t signal_size = 4096;
    Signal sine_wave = generator.generateSineWave(signal_size, signal_size/64.0);
    Signal chirp = generator.generateChirp(signal_size, signal_size/1024.0, signal_size/16.0);
    Signal complex_signal = generator.generateTestSignal(signal_size);

    // Create transforms
    FFTTransform original_fft;
    OptimizedFFT optimized_fft;

    // Compare performance
    std::cout << "===== Performance Comparison =====" << std::endl;
    benchmarkCompare(original_fft, optimized_fft, sine_wave);
    benchmarkCompare(original_fft, optimized_fft, chirp);
    benchmarkCompare(original_fft, optimized_fft, complex_signal);

    // Demonstrate visualization
    std::cout << "===== Signal Visualization =====" << std::endl;

    // ASCII plots
    std::cout << "ASCII Plot of Sine Wave:" << std::endl;
    SignalVisualizer::plotASCII(sine_wave, 80, 20, true);

    std::cout << "ASCII Plot of Chirp:" << std::endl;
    SignalVisualizer::plotASCII(chirp, 80, 20, true);

    // Export to CSV for external plotting
    SignalVisualizer::exportToCSV(sine_wave, "sine_wave.csv", signal_size);
    std::cout << "Exported sine wave to sine_wave.csv" << std::endl;

    Signal fft_result = optimized_fft.forward(sine_wave);
    SignalVisualizer::exportToCSV(fft_result, "sine_wave_spectrum.csv", signal_size, true);
    std::cout << "Exported sine wave spectrum to sine_wave_spectrum.csv" << std::endl;

    // Example of exporting a spectrogram (simplified)
    // In a real application, you'd compute the STFT properly
    std::vector<Signal> mock_spectrogram;
    for (int i = 0; i < 10; ++i) {
        Signal frame = generator.generateChirp(signal_size/4,
                                               signal_size/1024.0 * (i+1),
                                               signal_size/64.0 * (i+1));
        mock_spectrogram.push_back(optimized_fft.forward(frame));
    }

    SignalVisualizer::exportSpectrogramToCSV(mock_spectrogram, "spectrogram.csv", 0.01, signal_size);
    std::cout << "Exported example spectrogram to spectrogram.csv" << std::endl;

    // Compare window functions
    std::cout << "\n===== Window Function Comparison =====" << std::endl;

    // Create signal with harmonics
    Signal harmonics = generator.generateMultiSineWave(signal_size,
                                                       {signal_size/64.0, signal_size/32.0, signal_size/16.0},
                                                       {1.0, 0.5, 0.25});

    OptimizedFFT::Config cfg_none;
    cfg_none.window_type = OptimizedFFT::NONE;
    OptimizedFFT fft_none(cfg_none);

    OptimizedFFT::Config cfg_hanning;
    cfg_hanning.window_type = OptimizedFFT::HANNING;
    OptimizedFFT fft_hanning(cfg_hanning);

    // Compute spectra with different windows
    Signal spectrum_none = fft_none.forward(harmonics);
    Signal spectrum_hanning = fft_hanning.forward(harmonics);

    // Export for comparison
    SignalVisualizer::exportToCSV(spectrum_none, "spectrum_no_window.csv", signal_size, true);
    SignalVisualizer::exportToCSV(spectrum_hanning, "spectrum_hanning.csv", signal_size, true);

    std::cout << "Exported spectra with different windows to:" << std::endl;
    std::cout << "  - spectrum_no_window.csv" << std::endl;
    std::cout << "  - spectrum_hanning.csv" << std::endl;

    std::cout << "\nInstructions for visualizing the exported data:" << std::endl;
    std::cout << "1. You can use tools like Python with matplotlib, MATLAB, or Excel to plot the CSV files" << std::endl;
    std::cout << "2. For spectra, plot the 'Magnitude' column against 'Frequency'" << std::endl;
    std::cout << "3. For time domain signals, plot 'Real' or 'Magnitude' against 'Time'" << std::endl;
    std::cout << "4. For spectrograms, use a 3D plot or heatmap with x=Time, y=Frequency, z=Magnitude" << std::endl;

    return 0;
}