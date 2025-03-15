#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "graphics.h"
#include <omp.h>
#include <pthread.h>


#define EPSILON 1e-3
#define GRAPHICS 1 
#define pthread 0
#define CHUNK_SIZE 64

#if pthread
/*Parameters for ptheads*/
typedef struct{
    int thread_id;
    int N;
    double G;
    const double * restrict x;
    const double * restrict y;
    const double * restrict mass;
    double * ax_private;
    double * ay_private;
} thread_data;
// Functions for pthread
void* compute_forces_thread(void* args);
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int next_i = 0;
#endif

const int windowWidth = 800;
const float particleRadius = 0.005;
const float L = 1, W = 1;
const double EPS = EPSILON;
/* Function prototype, all pointers involving arrays are added with restrict qualifier */
void read_input_file(const char *filename, int N, 
                     double * restrict x, double * restrict y, 
                     double * restrict mass, double * restrict vx, 
                     double * restrict vy, double * restrict brightness);
void compute_forces(int N, double G, 
                    const double * restrict x, const double * restrict y, 
                    const double * restrict mass, 
                    double * restrict ax, double * restrict ay, int n_threads);
void update_positions(int N, double delta_t,
                      double * restrict x, double * restrict y, 
                      double * restrict vx, double * restrict vy, 
                      const double * restrict ax, const double * restrict ay, int n_threads);
void write_output_file(const char *filename, int N, 
                       const double * restrict x, const double * restrict y, 
                       const double * restrict mass, const double * restrict vx, 
                       const double * restrict vy, const double * restrict brightness);
void visualize(int N, const double * restrict x, const double * restrict y);
void simulate(int N, char *filename, int nsteps, double delta_t, int graphics, int n_threads);

int main(int argc, char *argv[]) {
    if (argc != 7) {
        printf("Usage: ./galsim N filename nsteps delta_t graphics n_threads\n");
        return 1;
    }
    int N = atoi(argv[1]);
    char *filename = argv[2];
    int nsteps = atoi(argv[3]);
    double delta_t = atof(argv[4]);
    int graphics = atoi(argv[5]);
    int n_threads = atoi(argv[6]);
    omp_set_num_threads(n_threads);
// #if pthread
//     n_threads > 1 ? n_threads-- : 1;
// #endif
    simulate(N, filename, nsteps, delta_t, graphics, n_threads);
    return 0;
}

/* Read data from a file that stores 6 doubles in sequence */
void read_input_file(const char *filename, int N, 
                     double * restrict x, double * restrict y, 
                     double * restrict mass, double * restrict vx, 
                     double * restrict vy, double * restrict brightness) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }
    /* Apply for a temporary buffer of size 6*N double*/
    double *temp = malloc(N * 6 * sizeof(double));
    if (!temp) {
        perror("Memory allocation failed");
        exit(1);
    }
    size_t count = fread(temp, sizeof(double), 6 * N, file);
    if (count != (size_t)(6 * N)) {
        perror("Error reading file");
        free(temp);
        fclose(file);
        exit(1);
    }
    fclose(file);
    /* Apply for a temporary buffer of size 6*N double */
    for (int i = 0; i < N; i++) {
        x[i]          = temp[6 * i];
        y[i]          = temp[6 * i + 1];
        mass[i]       = temp[6 * i + 2];
        vx[i]         = temp[6 * i + 3];
        vy[i]         = temp[6 * i + 4];
        brightness[i] = temp[6 * i + 5];
    }
    free(temp);
}


#if pthread
void *compute_forces_thread(void* args){
    thread_data *data = (thread_data *)args;
    int N = data->N;
    double G = data->G;
    const double * restrict x = data->x;
    const double * restrict y = data->y;
    const double * restrict mass = data->mass;
    double * restrict ax_private = data->ax_private;
    double * restrict ay_private = data->ay_private;

    // Use dynamic scheduling to solve the problem of uneven
    // workload of inner loops corresponding to different i

    while(1){
        // lock the mutex
        int current_i;
        pthread_mutex_lock(&mutex);
        current_i = next_i;
        next_i = next_i + CHUNK_SIZE;
        pthread_mutex_unlock(&mutex);

        if(current_i >= N){
            break;
        }
        int end_i = current_i + CHUNK_SIZE;
        if (end_i > N) {
            end_i = N;
        }
        for (int i = current_i; i < end_i; i++) {
            const double xi = x[i];
            const double yi = y[i];
            const double mi = mass[i];
            const double G_mi = G * mi;
            double axi = 0.0;
            double ayi = 0.0;
            for (int j = i + 1; j < N; j++) {
                const double xj = x[j];
                const double yj = y[j];
                const double dx = xj - xi;
                const double dy = yj - yi;
                const double r_sq = dx * dx + dy * dy;
                const double r = sqrt(r_sq);
                const double r_eps = r + EPSILON;
                const double denom = r_eps * r_eps * r_eps;
                const double a_i = G * mass[j] / denom;
                const double a_j = G_mi / denom;
                axi += a_i * dx;
                ayi += a_i * dy;
                ax_private[j] -= a_j * dx;
                ay_private[j] -= a_j * dy;
            }
            ax_private[i] += axi;
            ay_private[i] += ayi;
        }   
    }
    return NULL;
}
#endif


void compute_forces(int N, double G, 
                    const double * restrict x, const double * restrict y, 
                    const double * restrict mass, 
                    double * restrict ax, double * restrict ay, int n_threads) {

    memset(ax, 0, N * sizeof(double));
    memset(ay, 0, N * sizeof(double));

#if pthread
    // Initialize thread data and create threads
    pthread_t threads[n_threads];
    thread_data args[n_threads];
    for (int i = 0; i < n_threads; i++) {
        args[i].thread_id = i;
        args[i].N = N;
        args[i].G = G;
        args[i].x = x;
        args[i].y = y;
        args[i].mass = mass;
        args[i].ax_private = (double *)calloc(N, sizeof(double));
        args[i].ay_private = (double *)calloc(N, sizeof(double));
        if (args[i].ax_private == NULL || args[i].ay_private == NULL) {
            perror("Memory allocation failed in thread private arrays");
            exit(1);
        }
    }
    next_i = 0;
    pthread_mutex_init(&mutex, NULL);
    for (int i = 0; i < n_threads; i++) {
        pthread_create(&threads[i], NULL, compute_forces_thread, &args[i]);
    }
    // Wait for all threads to finish
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);

    // Add each thread's private array to the global array
    for(int i = 0; i < n_threads; i++) {
        for (int j = 0; j < N; j++) {
            ax[j] += args[i].ax_private[j];
            ay[j] += args[i].ay_private[j];
        }
        free(args[i].ax_private);
        free(args[i].ay_private);
    }
    pthread_mutex_destroy(&mutex);
#else    
    #pragma omp parallel for schedule(dynamic, 8) reduction(+:ax[:N],ay[:N])
    for (int i = 0; i < N; i++) {
        double axi = 0.0, ayi = 0.0;
        const double xi = x[i];
        const double yi = y[i];
        const double mi = mass[i];
        const double G_mi = G * mi;
        #pragma omp simd reduction(+:axi,ayi) 
        for (int j = i + 1; j < N; j++) {
            const double xj = x[j];
            const double yj = y[j];
            const double dx = xj - xi;
            const double dy = yj - yi;
            const double r_sq = dx * dx + dy * dy;
            const double r = sqrt(r_sq);
            const double r_eps = r + EPSILON;
            const double denom = r_eps * r_eps * r_eps;
            const double a_i = G * mass[j] / denom;
            const double a_j = G_mi / denom;
            axi += a_i * dx;
            ayi += a_i * dy;
            ax[j] -= a_j * dx;
            ay[j] -= a_j * dy;
        }
        ax[i] += axi;
        ay[i] += ayi;
    }
#endif
}


void update_positions(int N, double delta_t, 
                      double * restrict x, double * restrict y, 
                      double * restrict vx, double * restrict vy, 
                      const double * restrict ax, const double * restrict ay, int n_threads) {
    for (int i = 0; i < N; i++) {
        vx[i] += delta_t * ax[i];
        vy[i] += delta_t * ay[i];
        x[i]  += delta_t * vx[i];
        y[i]  += delta_t * vy[i];
    }
}


/* Convert each array data into AoS format and write it to the file */
void write_output_file(const char *filename, int N, 
                       const double * restrict x, const double * restrict y, 
                       const double * restrict mass, const double * restrict vx, 
                       const double * restrict vy, const double * restrict brightness) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening output file");
        exit(1);
    }
    /* Temporary buffer stores 6*N double */
    double *temp = malloc(N * 6 * sizeof(double));
    if (!temp) {
        perror("Memory allocation failed");
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        temp[6 * i]     = x[i];
        temp[6 * i + 1] = y[i];
        temp[6 * i + 2] = mass[i];
        temp[6 * i + 3] = vx[i];
        temp[6 * i + 4] = vy[i];
        temp[6 * i + 5] = brightness[i];
    }
    fwrite(temp, sizeof(double), 6 * N, file);
    fclose(file);
    free(temp);
}



/* Visualization function: draw all particles */
void visualize(int N, const double * restrict x, const double * restrict y) {
    ClearScreen();
    for (int i = 0; i < N; i++) {
        DrawCircle(x[i], y[i], L, W, particleRadius, 0);
    }
    Refresh();
}

/* Simulation function: allocate arrays, call modules and release memory */
void simulate(int N, char *filename, int nsteps, double delta_t, int graphics, int n_threads) {
    double *x          = malloc(N * sizeof(double));
    double *y          = malloc(N * sizeof(double));
    double *mass       = malloc(N * sizeof(double));
    double *vx         = malloc(N * sizeof(double));
    double *vy         = malloc(N * sizeof(double));
    double *brightness = malloc(N * sizeof(double));
    double *ax         = malloc(N * sizeof(double));
    double *ay         = malloc(N * sizeof(double));
    if (!x || !y || !mass || !vx || !vy || !brightness || !ax || !ay) {
        perror("Memory allocation failed");
        exit(1);
    }

    read_input_file(filename, N, x, y, mass, vx, vy, brightness);
    const double G = 100.0 / N;
    
#if GRAPHICS
    if (graphics) {
        InitializeGraphics("N-Body Simulation", windowWidth, windowWidth);
        SetCAxes(0, 1);
    }
#endif

    for (int step = 0; step < nsteps; step++) {
        compute_forces(N, G, x, y, mass, ax, ay, n_threads);
        update_positions(N, delta_t, x, y, vx, vy, ax, ay, n_threads);
#if GRAPHICS
        if (graphics) {
            visualize(N, x, y);
            usleep(3000);
            if (CheckForQuit()) break;
        }
#endif
    }
    
    write_output_file("result.gal", N, x, y, mass, vx, vy, brightness);
    
#if GRAPHICS
    if (graphics) {
        FlushDisplay();
        CloseDisplay();
    }
#endif

    free(x);
    free(y);
    free(mass);
    free(vx);
    free(vy);
    free(brightness);
    free(ax);
    free(ay);
}
