#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include "../include/Cooley-Tukey-parallel.hpp"
#include "../include/Cooley-Tukey.hpp"
#include "../include/MPIHelper.h"
#include "../include/VectorInitializer.h"

using namespace std;

// config of constants for VectorInitializer
constexpr int DATA_SIZE = 1 << 17;  // 2^17
constexpr unsigned int RANDOM_SEED = 95;


int main(int argc, char** argv){
    MPIHelper::instance(argc, argv);

    // Timing variables
    struct timeval t1, t2;
    double etimePar,etimeSeq;

    //Initialize solvers
    ParallelIterativeFFT ParallelFFTSolver = ParallelIterativeFFT();
    SequentialFFT SequentialFFTSolver = SequentialFFT();

    /* //creating a random input vector
    srand(95);
    std::vector<std::complex<double>>input_vector;
    for(int i=0; i<N; i++)
    {
       input_vector.push_back(std::complex<double>(rand() % RAND_MAX, rand() % RAND_MAX));
    }*/

    // Generate input data using the separate initializer
    std::vector<std::complex<double>> input_vector =
            VectorInitializer::createRandomVector(DATA_SIZE, RANDOM_SEED);

    // recursive FFT
    std::vector<std::complex<double>> recursiveResult = SequentialFFTSolver.recursive_FFT(input_vector);

    //exec and measure of SEQUENTIAL iterativeFFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> iterativeResult = SequentialFFTSolver.iterative_FFT(input_vector);
    gettimeofday(&t2, NULL);
	// etimeSeq = std::abs(t2.tv_usec - t1.tv_usec);
    etimeSeq = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);  // considers seconds too now (x1000000 for seconds)
    std::cout <<"Sequential version done, took ->  " << etimeSeq << " usec." << std::endl;

    // exec and measure of PARALLEL iterativeFFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> parallelResult = ParallelFFTSolver.findFFT(input_vector);
	gettimeofday(&t2, NULL);
    // etimePar = std::abs(t2.tv_usec - t1.tv_usec);
    etimePar = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);  // considers seconds too now (x1000000 for seconds)

    std::cout <<"Parallel version done, took ->  " << etimePar << " usec." << std::endl;
    // speedup
    std::cout<<"The parallel version is "<< etimeSeq/etimePar <<" times faster. "<<std::endl; 

    // exec and measure SEQUENTIAL INVERSE iterativeFFT
    gettimeofday(&t1, NULL);
    std::vector<std::complex<double>> iterativeInverseResult = SequentialFFTSolver.iterative_inverse_FFT(input_vector);
	gettimeofday(&t2, NULL);
    etimePar = std::abs(t2.tv_usec - t1.tv_usec);
	std::cout <<"Inverse iterative version done, took ->  " << etimePar << " usec." << std::endl;
    
/*
  double tolerance = 1e-9; // Define an acceptable tolerance
    bool inverseCheck = true;
    for (int i = 0; i < input_vector.size(); i++) {
        if (std::abs(input_vector[i] - iterativeInverseResult[i]) > tolerance) {
            cout << "Inverse Error at index: " << i 
                << " | Original: " << input_vector[i]
                << " | Inverse: " << iterativeInverseResult[i] << endl;
            inverseCheck = false;
        }
    }

    if (inverseCheck) {
        cout << "Inverse FFT validation successful!" << endl;
    } else {
        cout << "Inverse FFT validation failed!" << endl;
    }

*/
  


    //Checking if the 3 implementations give the same results 
    std::cout << "\nChecking results... " << std::endl;
    bool check = true;
    for(int i = 0; i < recursiveResult.size(); i++){
        if(recursiveResult[i]!=iterativeResult[i] && iterativeResult[i]!=parallelResult[i])
        {
            std::cout <<"Different result in line " << i << std::endl;
            check=false;
        }
    }

    if(check)
        std::cout <<"Same result for the 3 methods" << std::endl;
    else
        std::cout << "error 02394" << std::endl;

    return 0;
}