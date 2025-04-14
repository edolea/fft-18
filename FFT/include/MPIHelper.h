//
// Created by Edoardo Leali on 26/03/25.
//
#ifndef MPIHELPER_HPP
#define MPIHELPER_HPP

#include <mpi.h>

/**
 * \class MPIHelper
 * \brief Singleton class to manage MPI initialization and finalization.
 *
 * The MPIHelper class ensures that MPI is initialized at the start of the program
 * and finalized at the end. It uses the singleton pattern to provide a single
 * instance of the class.
 */
class MPIHelper {
public:
    /**
     * \brief Get the singleton instance of MPIHelper.
     * \param argc Reference to the argument count.
     * \param argv Reference to the argument vector.
     * \return Reference to the singleton instance of MPIHelper.
     */
    static MPIHelper& instance(int& argc, char**& argv) {
        static MPIHelper singleton(argc, argv);
        return singleton;
    }

private:
    bool initializedHere_;

    /**
     * \brief Constructor that initializes MPI if not already initialized.
     * \param argc Reference to the argument count.
     * \param argv Reference to the argument vector.
     */
    MPIHelper(int& argc, char**& argv) : initializedHere_(false) {
        int flag;
        MPI_Initialized(&flag);
        if (!flag) {
            MPI_Init(&argc, &argv);
            initializedHere_ = true;
        }
    }

    /**
     * \brief Destructor that finalizes MPI if it was initialized by this instance.
     */
    ~MPIHelper() {
        if (initializedHere_)
            MPI_Finalize();
    }

    // Delete copy constructor and assignment operator to prevent copying.
    MPIHelper(const MPIHelper&) = delete;
    MPIHelper& operator=(const MPIHelper&) = delete;
};

#endif // MPIHELPER_HPP
