#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "graphics.h"

#define EPSILON 1e-3
#define GRAPHICS 1 

const int windowWidth = 800;
const float particleRadius = 0.005;
const float L = 1, W = 1;

/* Function prototype, all pointers involving arrays are added with restrict qualifier */
void read_input_file(const char *filename, int N, 
                     double * restrict x, double * restrict y, 
                     double * restrict mass, double * restrict vx, 
                     double * restrict vy, double * restrict brightness);
void compute_forces(int N, double G, 
                    const double * restrict x, const double * restrict y, 
                    const double * restrict mass, 
                    double * restrict ax, double * restrict ay);
void update_positions(int N, double delta_t, 
                      double * restrict x, double * restrict y, 
                      double * restrict vx, double * restrict vy, 
                      const double * restrict ax, const double * restrict ay);
void write_output_file(const char *filename, int N, 
                       const double * restrict x, const double * restrict y, 
                       const double * restrict mass, const double * restrict vx, 
                       const double * restrict vy, const double * restrict brightness);
void visualize(int N, const double * restrict x, const double * restrict y);
void simulate(int N, char *filename, int nsteps, double delta_t, int graphics);

int main(int argc, char *argv[]) {
    if (argc != 6) {
        printf("Usage: ./galsim N filename nsteps delta_t graphics\n");
        return 1;
    }
    int N = atoi(argv[1]);
    char *filename = argv[2];
    int nsteps = atoi(argv[3]);
    double delta_t = atof(argv[4]);
    int graphics = atoi(argv[5]);
    simulate(N, filename, nsteps, delta_t, graphics);
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

void compute_forces(int N, double G, 
                    const double * restrict x, const double * restrict y, 
                    const double * restrict mass, 
                    double * restrict ax, double * restrict ay) {
    const double EPS = EPSILON;
    memset(ax, 0, N * sizeof(double));
    memset(ay, 0, N * sizeof(double));
    for (int i = 0; i < N; i++) {
        // Precompute all constants and mark them with const so that the compiler can optimize
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
            const double r_sq = dx*dx + dy*dy;
            const double r = sqrt(r_sq);
            const double r_eps = r + EPS;
            const double denom = r_eps * r_eps * r_eps;
            const double mass_j = mass[j];
            const double a_i = G * mass_j / denom;
            const double a_j = G_mi / denom;
            axi += a_i * dx;
            ayi += a_i * dy;
            ax[j] -= a_j * dx;
            ay[j] -= a_j * dy;
        }
        ax[i] += axi;
        ay[i] += ayi;
    }
}

/* Update the positions and velocities of all particles */
void update_positions(int N, double delta_t, 
                      double * restrict x, double * restrict y, 
                      double * restrict vx, double * restrict vy, 
                      const double * restrict ax, const double * restrict ay) {
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
void simulate(int N, char *filename, int nsteps, double delta_t, int graphics) {
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
        compute_forces(N, G, x, y, mass, ax, ay);
        update_positions(N, delta_t, x, y, vx, vy, ax, ay);
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
