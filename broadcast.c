#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int rang;
    int a = 0;


    MPI_Init(&argc, &argv);


    MPI_Comm_rank(MPI_COMM_WORLD, &rang);


    if (rang == 0) {
        a = 22;
    }


    printf("%d : avant le MPI_BCAST, a = %d\n", rang, a);
    fflush(stdout);

    MPI_Bcast(&a, 1, MPI_INT, 2, MPI_COMM_WORLD);
    sleep(1);

    printf("%d : apres le MPI_BCAST, a = %d\n", rang, a);
    fflush(stdout);

    MPI_Finalize();

    return 0;
}