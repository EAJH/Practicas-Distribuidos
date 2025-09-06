#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

void get_walltime(double* wcTime) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (tp.tv_sec + tp.tv_usec / 1000000.0);
}

int main(int argc, char* argv[]) {
    int i, j, k, n = 1000;
    int **matrizA, **matrizB, **matrizC;
    double S1, E1;

    // Inicializando matrices
    matrizA = (int **)malloc(n * sizeof(int *));
    matrizB = (int **)malloc(n * sizeof(int *));
    matrizC = (int **)malloc(n * sizeof(int *));
    for (i = 0; i < n; i++) {
        *(matrizA + i) = (int *)malloc(n * sizeof(int));
        *(matrizB + i) = (int *)malloc(n * sizeof(int));
        *(matrizC + i) = (int *)malloc(n * sizeof(int));
    }

    // Llenando matrices
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrizA[i][j] = rand() % 6;
            matrizB[i][j] = rand() % 6;
            matrizC[i][j] = 0;
        }
    }

    get_walltime(&S1);

    // Versión 1. ijk
    /*for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            for (k = 0; k < n; ++k) {
                matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
            }
        }
    }*/

    // Versión 2. ikj
    for (i = 0; i < n; ++i) {
        for (k = 0; k < n; ++k) {
            for (j = 0; j < n; ++j) {
                matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
            }
        }
    }

    // Versión 3. jik
    /*for (j = 0; j < n; ++j) {
        for (i = 0; i < n; ++i) {
            for (k = 0; k < n; ++k) {
                matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
            }
        }
    }*/

    // Versión 4. jki
    /*for (j = 0; j < n; ++j) {
        for (k = 0; k < n; ++k) {
            for (i = 0; i < n; ++i) {
                matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
            }
        }
    }*/

    // Versión 5. kij
    /*for (k = 0; k < n; ++k) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
            }
        }
    }*/

    // Versión 6. kji
    /*for (k = 0; k < n; ++k) {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < n; ++i) {
                matrizC[i][j] += matrizA[i][k] * matrizB[k][j];
            }
        }
    }*/

    get_walltime(&E1);

    //printf("Tiempo metodo ijk: %f s\n", (E1 - S1));
    printf("Tiempo metodo ikj: %f s\n", (E1 - S1));
    //printf("Tiempo metodo jik: %f s\n", (E1 - S1));
    //printf("Tiempo metodo jki: %f s\n", (E1 - S1));
    //printf("Tiempo metodo kij: %f s\n", (E1 - S1));
    //printf("Tiempo metodo kji: %f s\n", (E1 - S1));


    // Liberar memoria (es buena práctica)
    for (i = 0; i < n; i++) {
        free(matrizA[i]);
        free(matrizB[i]);
        free(matrizC[i]);
    }
    free(matrizA);
    free(matrizB);
    free(matrizC);

    return 0;
}