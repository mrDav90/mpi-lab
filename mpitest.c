#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h> // Pour srand dans une initialisation plus complexe (non utilisé ici)

#define TOTAL_NUMBERS 1000000 // 1 million de nombres
#define ROOT_RANK 0

// Fonction pour la somme séquentielle (pour comparaison)
long long sum_sequential_range(long long start, long long end) {
    long long sum = 0;
    for (long long i = start; i <= end; ++i) {
        sum += i;
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    long long local_sum_mpi = 0;
    long long global_sum_mpi = 0;

    double time_seq_start, time_seq_end;
    double time_mpi_global_start, time_mpi_global_end;
    double time_mpi_local_calc_start, time_mpi_local_calc_end;

    // Initialisation de MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtient le rang du processus courant
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // Obtient le nombre total de processus

    // --- Partie Séquentielle (exécutée uniquement par le processus root pour comparaison) ---
    if (rank == ROOT_RANK) {
        printf("--- Calcul Séquentiel (sur 1 cœur) ---\n");
        printf("Calcul de la somme des nombres de 1 à %d...\n", TOTAL_NUMBERS);

        time_seq_start = MPI_Wtime(); // Temps MPI pour la précision
        long long sum_seq_result = sum_sequential_range(1, TOTAL_NUMBERS);
        time_seq_end = MPI_Wtime();

        printf("Somme séquentielle calculée : %lld\n", sum_seq_result);
        printf("Temps d'exécution séquentiel : %f secondes\n\n", time_seq_end - time_seq_start);
    }

    // --- Partie MPI ---
    // Synchronisation de tous les processus avant de démarrer le chronomètre global MPI
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == ROOT_RANK) {
        printf("--- Calcul Parallèle MPI (avec %d processus) ---\n", num_procs);
        time_mpi_global_start = MPI_Wtime();
    }

    // Calcul de la plage de nombres pour chaque processus
    // Chaque processus va calculer la somme d'une portion des nombres.
    // TOTAL_NUMBERS doit être divisible par num_procs pour cette répartition simple.
    if (TOTAL_NUMBERS % num_procs != 0) {
        if (rank == ROOT_RANK) {
            fprintf(stderr, "ERREUR: TOTAL_NUMBERS (%d) n'est pas divisible par num_procs (%d) pour cet exemple simple.\n", TOTAL_NUMBERS, num_procs);
            fprintf(stderr, "Veuillez ajuster TOTAL_NUMBERS ou le nombre de processus.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1); // Termine tous les processus MPI
    }
    long long numbers_per_proc = TOTAL_NUMBERS / num_procs;
    long long my_start_num = (long long)rank * numbers_per_proc + 1;
    long long my_end_num = (long long)(rank + 1) * numbers_per_proc;

    // Étape 1: Chaque processus calcule sa somme locale ET chronomètre ce calcul
    time_mpi_local_calc_start = MPI_Wtime();
    for (long long i = my_start_num; i <= my_end_num; ++i) {
        local_sum_mpi += i;
    }
    time_mpi_local_calc_end = MPI_Wtime();

    // Affichage des informations par chaque processus
    // Pour éviter un affichage chaotique, on peut faire un MPI_Barrier et laisser les processus afficher séquentiellement
    // ou accepter un affichage un peu mélangé (ce qui est courant en MPI).
    // Ici, on laisse l'affichage potentiellement mélangé, ce qui est plus réaliste.
    printf("Processus %d: a additionné les nombres de %lld à %lld. Somme locale = %lld. Temps de calcul local: %.6f sec.\n",
           rank, my_start_num, my_end_num, local_sum_mpi, time_mpi_local_calc_end - time_mpi_local_calc_start);


    // Étape 2: Collecter toutes les sommes locales sur le processus ROOT_RANK
    // MPI_Reduce est une opération collective efficace pour cela.
    // Elle prend la 'local_sum_mpi' de chaque processus, effectue MPI_SUM,
    // et stocke le résultat dans 'global_sum_mpi' sur le processus ROOT_RANK.
    MPI_Reduce(
        &local_sum_mpi,    // Adresse de la donnée à envoyer (la somme locale)
        &global_sum_mpi,   // Adresse où stocker le résultat (significatif seulement sur root)
        1,                 // Nombre d'éléments à réduire (une seule somme par processus)
        MPI_LONG_LONG,     // Type de la donnée
        MPI_SUM,           // Opération de réduction (somme)
        ROOT_RANK,         // Rang du processus qui reçoit le résultat final
        MPI_COMM_WORLD     // Communicateur
    );

    // Synchronisation avant d'arrêter le chronomètre global et d'afficher le résultat final
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == ROOT_RANK) {
        time_mpi_global_end = MPI_Wtime();
        printf("\nSomme globale MPI calculée par le root : %lld\n", global_sum_mpi);
        printf("Temps total d'exécution MPI (incluant communication et synchronisation) : %.6f secondes\n", time_mpi_global_end - time_mpi_global_start);

        // Comparaison optionnelle des résultats
        long long sum_seq_check = sum_sequential_range(1, TOTAL_NUMBERS);
        if (global_sum_mpi == sum_seq_check) {
            printf("VÉRIFICATION: Les sommes séquentielle et MPI sont IDENTIQUES.\n");
        } else {
            printf("ERREUR DE VÉRIFICATION: Sommes différentes! Seq: %lld, MPI: %lld\n", sum_seq_check, global_sum_mpi);
        }
    }

    // Finalisation de MPI
    MPI_Finalize();
    return 0;
}