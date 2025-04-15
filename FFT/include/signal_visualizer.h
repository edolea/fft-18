#ifndef SIGNAL_VISUALIZER_H
#define SIGNAL_VISUALIZER_H

#include <vector>
#include <complex>
#include <string>
#include <fstream>

// Type alias for convenience
using Complex = std::complex<double>;
using Signal = std::vector<Complex>;

/**
 * Class for visualizing signals and transform results
 */
class SignalVisualizer {
public:
    /**
     * Writes signal data to a CSV file with time/frequency information
     * 
     * @param signal The signal to export
     * @param filename The filename to write to
     * @param sample_rate Optional sample rate (for proper time axis)
     * @param is_frequency_domain Whether this is frequency domain data
     * @return true if successful
     */
    static bool exportToCSV(const Signal& signal,
                            const std::string& filename,
                            double sample_rate = 1.0,
                            bool is_frequency_domain = false);

    /**
     * Writes a spectrogram (time-frequency representation) to a CSV file
     * 
     * @param spectrogram Vector of frequency domain signals representing STFT
     * @param filename The filename to write to
     * @param time_step Time step between frames
     * @param sample_rate Sample rate for frequency calculation
     * @return true if successful
     */
    static bool exportSpectrogramToCSV(const std::vector<Signal>& spectrogram,
                                       const std::string& filename,
                                       double time_step = 1.0,
                                       double sample_rate = 1.0);

    /**
     * Creates a simple ASCII plot of a signal in the console
     * 
     * @param signal The signal to plot
     * @param width Width of the plot in characters
     * @param height Height of the plot in characters
     * @param real_part Whether to plot the real part (true) or magnitude (false)
     */
    static void plotASCII(const Signal& signal,
                          size_t width = 80,
                          size_t height = 20,
                          bool real_part = true);

private:
    // Helper methods for ASCII plotting
    static double findMin(const Signal& signal, bool real_part);
    static double findMax(const Signal& signal, bool real_part);
};

#endif // SIGNAL_VISUALIZER_H