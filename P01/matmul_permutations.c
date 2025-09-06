// matmul_permutations.c
// Compilar (double por defecto):  gcc -O3 -march=native -pthread matmul_permutations.c -o matmul
// Compilar (float para ahorrar memoria):  gcc -O3 -march=native -pthread -DUSE_FLOAT matmul_permutations.c -o matmul
// Ejecutar:  ./matmul
//            ./matmul --progress
//            ./matmul 256 768 1024 5000 10000   (tamaños custom)
// Nota: --progress puede afectar ligeramente los tiempos medidos.
// [NUEVO] Ahora también se imprime el tiempo de llenado (A, B y A+B).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>

/* ===========================
   BLOQUE: Tipo numérico y helpers
   =========================== */
#ifdef USE_FLOAT
typedef float real_t;
#else
typedef double real_t;
#endif

static inline double wall_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

/* ===========================
   BLOQUE: Utilidades de memoria/índices
   =========================== */
static inline size_t idx(size_t i, size_t j, size_t n) { return i*n + j; }

static void fill_rand(real_t *M, size_t n) {
    for (size_t i = 0; i < n*n; ++i) M[i] = (real_t)(rand() % 6);
}
static void zero_mat(real_t *M, size_t n) {
    memset(M, 0, n*n*sizeof(real_t));
}

/* ===========================
   BLOQUE: Kernels (6 permutaciones)
   =========================== */

// ijk
static void kernel_ijk(const real_t *A, const real_t *B, real_t *C, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            real_t sum = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += A[idx(i,k,n)] * B[idx(k,j,n)];
            }
            C[idx(i,j,n)] = sum;
        }
    }
}

// ikj
static void kernel_ikj(const real_t *A, const real_t *B, real_t *C, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            real_t r = A[idx(i,k,n)];
            for (size_t j = 0; j < n; ++j) {
                C[idx(i,j,n)] += r * B[idx(k,j,n)];
            }
        }
    }
}

// jik
static void kernel_jik(const real_t *A, const real_t *B, real_t *C, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            real_t sum = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += A[idx(i,k,n)] * B[idx(k,j,n)];
            }
            C[idx(i,j,n)] = sum;
        }
    }
}

// jki
static void kernel_jki(const real_t *A, const real_t *B, real_t *C, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) {
            real_t r = B[idx(k,j,n)];
            for (size_t i = 0; i < n; ++i) {
                C[idx(i,j,n)] += A[idx(i,k,n)] * r;
            }
        }
    }
}

// kij
static void kernel_kij(const real_t *A, const real_t *B, real_t *C, size_t n) {
    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
            real_t r = A[idx(i,k,n)];
            for (size_t j = 0; j < n; ++j) {
                C[idx(i,j,n)] += r * B[idx(k,j,n)];
            }
        }
    }
}

// kji
static void kernel_kji(const real_t *A, const real_t *B, real_t *C, size_t n) {
    for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < n; ++j) {
            real_t r = B[idx(k,j,n)];
            for (size_t i = 0; i < n; ++i) {
                C[idx(i,j,n)] += A[idx(i,k,n)] * r;
            }
        }
    }
}

/* ===========================
   BLOQUE: Timer de progreso (opcional)
   =========================== */
typedef struct {
    volatile int running;
    double t0;
    char label[128];
} progress_t;

static void* progress_thread(void *arg) {
    progress_t *p = (progress_t*)arg;
    while (p->running) {
        double t = wall_time_sec() - p->t0;
        fprintf(stderr, "\r%s  tiempo transcurrido: %.2f s", p->label, t);
        fflush(stderr);
        sleep(1);
    }
    return NULL;
}

/* ===========================
   BLOQUE: Verificación (opcional)
   =========================== */
#ifndef VERIFY
#define VERIFY 0
#endif
static int compare_mats(const real_t *X, const real_t *Y, size_t n, double eps) {
    for (size_t i = 0; i < n*n; ++i) {
        double d = (double)X[i] - (double)Y[i];
        if (d < 0) d = -d;
        if (d > eps) return 0;
    }
    return 1;
}

/* ===========================
   BLOQUE: Runner / main
   =========================== */
typedef void (*kernel_fn)(const real_t*, const real_t*, real_t*, size_t);

typedef struct { const char *name; kernel_fn fn; int needs_zero; } kernel_desc;

static void print_size_info(size_t n) {
    const double bytes_mat = (double)n * (double)n * (double)sizeof(real_t);
    double total = bytes_mat * 3.0; // A,B,C
#if VERIFY
    total += bytes_mat; // Cref
#endif
    double gib = total / (1024.0*1024.0*1024.0);
    double flops = 2.0 * (double)n * (double)n * (double)n; // ~2n^3
    printf("  ~Memoria (A,B,C)%s: %.2f GiB | Elem=%zu bytes | FLOPs≈ %.2e\n",
#if VERIFY
           "+Cref"
#else
           ""
#endif
           , gib, (size_t)sizeof(real_t), flops);
}

int main(int argc, char **argv) {
    int show_progress = 0;
    int sizes_from_args = 0;
    int max_sizes = 16;
    int *sizes = (int*)malloc(max_sizes * sizeof(int));
    int nsizes = 0;

    for (int a = 1; a < argc; ++a) {
        if (strcmp(argv[a], "--progress") == 0) {
            show_progress = 1;
        } else {
            int n = atoi(argv[a]);
            if (n > 0) {
                if (nsizes >= max_sizes) { fprintf(stderr, "Demasiados tamaños.\n"); return 1; }
                sizes[nsizes++] = n;
                sizes_from_args = 1;
            }
        }
    }
    if (!sizes_from_args) {
        // Por defecto: 100, 500, 1000, 5000 y 10000
        sizes[0] = 100; sizes[1] = 500; sizes[2] = 1000; sizes[3] = 5000; sizes[4] = 10000; nsizes = 5;
    }

    kernel_desc kernels[6] = {
        {"ijk", kernel_ijk, 0},
        {"ikj", kernel_ikj, 1},
        {"jik", kernel_jik, 0},
        {"jki", kernel_jki, 1},
        {"kij", kernel_kij, 1},
        {"kji", kernel_kji, 1}
    };

    srand(12345);

    for (int si = 0; si < nsizes; ++si) {
        size_t n = (size_t)sizes[si];
        size_t bytes = n*n*sizeof(real_t);

        printf("=== n = %zu ===\n", n);
        print_size_info(n);

        real_t *A = (real_t*)malloc(bytes);
        real_t *B = (real_t*)malloc(bytes);
        real_t *C = (real_t*)malloc(bytes);
#if VERIFY
        real_t *Cref = (real_t*)malloc(bytes);
#endif
        if (!A || !B || !C
#if VERIFY
            || !Cref
#endif
        ) {
            fprintf(stderr, "Fallo de memoria para n=%zu (prueba con -DUSE_FLOAT o reduce n)\n\n", n);
            free(A); free(B); free(C);
#if VERIFY
            free(Cref);
#endif
            continue; // pasa al siguiente tamaño
        }

        // --------- [NUEVO] Medición del llenado de A y B ----------
        double tA0 = wall_time_sec();
        fill_rand(A, n);
        double tA = wall_time_sec() - tA0;

        double tB0 = wall_time_sec();
        fill_rand(B, n);
        double tB = wall_time_sec() - tB0;

        printf("Tiempo llenado A   : %.6f s\n", tA);
        printf("Tiempo llenado B   : %.6f s\n", tB);
        printf("Tiempo llenado A+B : %.6f s\n", tA + tB);

#if VERIFY
        zero_mat(Cref, n);
        kernels[0].fn(A, B, Cref, n);
#endif

        for (int ki = 0; ki < 6; ++ki) {
            zero_mat(C, n);

            char label[128];
            snprintf(label, sizeof(label), "[%s n=%zu]", kernels[ki].name, n);

            progress_t prog = {0};
            pthread_t tid;
            double t0 = wall_time_sec();

            if (show_progress) {
                prog.running = 1;
                prog.t0 = t0;
                snprintf(prog.label, sizeof(prog.label), "%s", label);
                if (pthread_create(&tid, NULL, progress_thread, &prog) != 0) {
                    perror("pthread_create");
                    prog.running = 0;
                }
            }

            kernels[ki].fn(A, B, C, n);

            double t1 = wall_time_sec();
            if (show_progress && prog.running) {
                prog.running = 0;
                pthread_join(tid, NULL);
                fprintf(stderr, "\r%*s\r", (int)strlen(prog.label) + 30, "");
            }

            printf("Tiempo %-3s : %.6f s\n", kernels[ki].name, t1 - t0);

#if VERIFY
            if (!compare_mats(C, Cref, n, 1e-6)) {
                fprintf(stderr, "Verificación FALLÓ para %s (n=%zu)\n", kernels[ki].name, n);
            }
#endif
        }
        printf("\n");

        free(A); free(B); free(C);
#if VERIFY
        free(Cref);
#endif
    }

    free(sizes);
    return 0;
}

/* =========================================================================================
   RESPUESTA SOLICITADA: Comparación de resultados y explicación (con tus mediciones)
   -----------------------------------------------------------------------------------------
   Resultados del alumno (n=100, 500, 1000):

   n = 100
   - ijk : 0.006119 s
   - ikj : 0.007745 s
   - jik : 0.006022 s
   - jki : 0.006014 s
   - kij : 0.006544 s
   - kji : 0.005921 s
   Ranking (rápido → lento): kji < jki ≈ jik < ijk < kij << ikj   (diferencias muy pequeñas)

   n = 500
   - ijk : 0.824959 s
   - ikj : 0.751153 s
   - jik : 0.729013 s
   - jki : 0.879030 s
   - kij : 0.708742 s
   - kji : 0.875437 s
   Ranking: kij < jik < ikj < ijk < kji ≈ jki

   n = 1000
   - ijk : 6.465558 s
   - ikj : 5.849394 s
   - jik : 6.207130 s
   - jki : 8.837978 s
   - kij : 5.899218 s
   - kji : 9.105847 s
   Ranking: ikj ≈ kij < jik < ijk << jki ≈ kji

   ¿Cuál es más rápida y cuál más lenta?
   - Para tamaños grandes (500 y 1000), las más rápidas en tu máquina fueron IKJ y KIJ.
     La más lenta fue KJI (muy cerca JKI). Para n=100 las diferencias son mínimas.

   ¿Por qué ocurre esto?
   - C es row-major. Recorrer filas (j interno) es contiguo y aprovecha la caché.
     Recorrer columnas (i interno) tiene stride n y provoca más fallos de caché.
   - IKJ/KIJ barren B[k,*] y C[i,*] por filas (contiguo) y reutilizan A(i,k) en registro.
   - JKI/KJI recorren columnas de A y C (stride n) → peor localidad → más lentas.
   - IJK/JIK quedan a mitad: escriben C una vez, pero leen B por columna.

   Notas:
   - n=5000 y 10000: consumo de memoria y tiempo crecen mucho (~O(n^3)).
     Para 10k considera compilar con -DUSE_FLOAT o medir en equipo con >3 GiB libres.
   ========================================================================================= */
