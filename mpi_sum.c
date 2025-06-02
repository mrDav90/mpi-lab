#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TOTAL_NUMBERS 1000000

int main(int argc, char** argv) {
    int rank, size;
    long long i;
    long long local_sum = 0;
    long long global_sum = 0;
    int* all_numbers = NULL;
    int* local_numbers = NULL;
    int numbers_per_process;
    int remainder_numbers; // Pour gérer le cas où TOTAL_NUMBERS n'est pas divisible par size

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    numbers_per_process = TOTAL_NUMBERS / size;
    remainder_numbers = TOTAL_NUMBERS % size;

    // Le processus 0 (maître) génère les nombres et les distribue
    if (rank == 0) {
        all_numbers = (int*)malloc(TOTAL_NUMBERS * sizeof(int));
        if (all_numbers == NULL) {
            fprintf(stderr, "Erreur d'allocation mémoire pour all_numbers\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        //printf("Génération de %d nombres aléatoires par le processus 0...\n", TOTAL_NUMBERS);
        srand(time(NULL)); // Initialiser le générateur de nombres aléatoires
        for (i = 0; i < TOTAL_NUMBERS; i++) {
            all_numbers[i] = rand() % 100; // Nombres entre 0 et 99 pour simplifier
        }
    }

    // Calculer le nombre d'éléments pour ce processus
    int current_process_chunk_size = numbers_per_process + (rank < remainder_numbers ? 1 : 0);
    local_numbers = (int*)malloc(current_process_chunk_size * sizeof(int));
    if (local_numbers == NULL) {
        fprintf(stderr, "Processus %d: Erreur d'allocation mémoire pour local_numbers\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Définir les counts et les déplacements pour MPI_Scatterv
    int* sendcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int current_displ = 0;
        for (int p = 0; p < size; p++) {
            sendcounts[p] = numbers_per_process + (p < remainder_numbers ? 1 : 0);
            displs[p] = current_displ;
            current_displ += sendcounts[p];
        }
    }

    // Distribuer les données avec MPI_Scatterv
    // MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm)
    MPI_Scatterv(all_numbers, sendcounts, displs, MPI_INT,
                 local_numbers, current_process_chunk_size, MPI_INT,
                 0, MPI_COMM_WORLD);

    // Chaque processus calcule sa somme locale
    for (i = 0; i < current_process_chunk_size; i++) {
        local_sum += local_numbers[i];
    }
    //printf("Processus %d: somme locale = %lld (traitement de %d nombres)\n", rank, local_sum, current_process_chunk_size);


    // Réduire (rassembler et sommer) toutes les sommes locales vers global_sum sur le processus 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Le processus 0 affiche le résultat final
    if (rank == 0) {
        printf("Calcul distribué de la somme de %d nombres.\n", TOTAL_NUMBERS);
        printf("Nombre de processus MPI utilisés : %d\n", size);
        printf("Somme globale calculée : %lld\n", global_sum);

        // Optionnel : Vérification avec une somme séquentielle (pour de petits nombres)
        /*
        long long check_sum = 0;
        for (i = 0; i < TOTAL_NUMBERS; i++) {
            check_sum += all_numbers[i];
        }
        printf("Somme de vérification (séquentielle) : %lld\n", check_sum);
        if (global_sum == check_sum) {
            printf("Vérification OK !\n");
        } else {
            printf("ERREUR de vérification !\n");
        }
        */
        free(all_numbers);
        free(sendcounts);
        free(displs);
    }
    free(local_numbers);

    MPI_Finalize();
    return 0;
}