#ifndef FFT_MPI_WRAPPER_HPP
#define FFT_MPI_WRAPPER_HPP

#include <mpi.h>
#include <iostream>

class MPIWrapper {
private:
    int rank_{0};
    int size_{1};
    bool mpiInitialized_{false};

public:
    MPIWrapper() {
        int flag;
        MPI_Initialized(&flag);

        if (!flag) {
            int provided;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
            mpiInitialized_ = true;
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~MPIWrapper() {
        if (mpiInitialized_) {
            int flag;
            MPI_Finalized(&flag);
            if (!flag) {
                MPI_Finalize();
            }
        }
    }

    // Basic MPI information
    int rank() const { return rank_; }
    int size() const { return size_; }
    bool isRoot() const { return rank_ == 0; }

    // Communication operations
    template<typename T>
    void broadcast(T* data, int count, int root = 0) const {
        MPI_Bcast(data, count * sizeof(T), MPI_BYTE, root, MPI_COMM_WORLD);
    }

    template<typename T>
    void allGather(const T* sendData, int sendCount, T* recvData, int recvCount) const {
        MPI_Allgather(sendData, sendCount, MPI_BYTE, recvData, recvCount, MPI_BYTE, MPI_COMM_WORLD);
    }

    template<typename T>
    void send(const T* data, int count, int dest, int tag) const {
        MPI_Send(data, count * sizeof(T), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
    }

    template<typename T>
    void recv(T* data, int count, int source, int tag) const {
        MPI_Recv(data, count * sizeof(T), MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    void barrier() const {
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

#endif // FFT_MPI_WRAPPER_HPP

/*
// Singleton pattern to provide a global MPIWrapper instance
inline MPIWrapper& GetSharedMPI() {
    static MPIWrapper instance;
    return instance;
}



class MPIStream {
private:
    MPIWrapper& mpi_;
    std::ostream& stream_;

public:
    MPIStream(MPIWrapper& mpi, std::ostream& stream = std::cout)
        : mpi_(mpi), stream_(stream) {}

    template<typename T>
    MPIStream& operator<<(const T& value) {
        if (mpi_.isRoot()) {
            stream_ << value;
        }
        return *this;
    }

    // Special handling for manipulators like std::endl
    MPIStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (mpi_.isRoot()) {
            stream_ << manip;
        }
        return *this;
    }
};

// Global MPI-aware output stream that only prints on rank 0
inline MPIStream mpi_cout(GetSharedMPI());
*/