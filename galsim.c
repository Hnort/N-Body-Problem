#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "graphics.h"

#define EPSILON 1e-3
#define GRAPHICS 0

const int windowWidth = 800;
const float particleRadius = 0.005;
const float L = 1, W = 1;

typedef struct {
    double x, y;
    double mass;
    double vx, vy;
    double brightness;
} Particle;

void read_input_file(const char *filename, Particle **particles, int *N);
void compute_forces(Particle *particles, const int N, const double G, double *ax, double *ay);
void update_positions(Particle *particles, const int N, double delta_t, double *ax, double *ay);
void write_output_file(const char *filename, Particle *particles, int N);
void simulate(int N, char *filename, int nsteps, double delta_t, int graphics);
void visualize(Particle *particles, int N);

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

void read_input_file(const char *filename, Particle **particles, int *N) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    *particles = (Particle *)malloc((*N) * sizeof(Particle));
    if (!(*particles)) {
        perror("Memory allocation failed");
        exit(1);
    }

    size_t read_count = fread(*particles, sizeof(Particle), *N, file);
    if (read_count != (size_t)(*N)) {
        perror("Error reading file");
        free(*particles);
        fclose(file);
        exit(1);
    }

    fclose(file);
}

void compute_forces(Particle *particles, const int N, const double G, double *ax, double *ay) {
    for (int i = 0; i < N; i++) {
        ax[i] = 0.0;
        ay[i] = 0.0;
    }

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (i == j) continue;

    //         double dx = particles[j].x - particles[i].x;
    //         double dy = particles[j].y - particles[i].y;
    //         double r2 = dx * dx + dy * dy + EPSILON * EPSILON;
    //         double r = sqrt(r2);
    //         double F = G * particles[i].mass * particles[j].mass / (r2 * r);

    //         ax[i] += F * dx / particles[i].mass;
    //         ay[i] += F * dy / particles[i].mass;
    //     }
    // }

    for (int i = 0; i < N; i++) {
        // Symmetric Update
        for (int j = i + 1; j < N; j++) {
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double r2 = dx * dx + dy * dy + EPSILON * EPSILON;
            double inv_r3 = 1.0 / (r2 * sqrt(r2));
            double F = G * particles[i].mass * particles[j].mass * inv_r3;

            double Fx = F * dx;
            double Fy = F * dy;

            ax[i] += Fx / particles[i].mass;
            ay[i] += Fy / particles[i].mass;

            ax[j] -= Fx / particles[j].mass;
            ay[j] -= Fy / particles[j].mass;
        }
    }
}


void update_positions(Particle *particles, const int N, double delta_t, double *ax, double *ay) {
    for (int i = 0; i < N; i++) {
        particles[i].vx += delta_t * ax[i];
        particles[i].vy += delta_t * ay[i];
        particles[i].x += delta_t * particles[i].vx;
        particles[i].y += delta_t * particles[i].vy;
    }
}

void write_output_file(const char *filename, Particle *particles, int N) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening output file");
        exit(1);
    }

    fwrite(particles, sizeof(Particle), N, file);
    fclose(file);
}


void simulate(int N, char *filename, int nsteps, double delta_t, int graphics) {
    Particle *particles;
    read_input_file(filename, &particles, &N);

    const double G = 100.0 / N;

    double *ax = (double *)malloc(N * sizeof(double));
    double *ay = (double *)malloc(N * sizeof(double));
    if (!ax || !ay) {
        perror("Memory allocation failed");
        exit(1);
    }

#if GRAPHICS
    if (graphics) {
        InitializeGraphics("N-Body Simulation", 800, 800);
        SetCAxes(0, 1);
    }
#endif

    for (int step = 0; step < nsteps; step++) {
        compute_forces(particles, N, G, ax, ay);
        update_positions(particles, N, delta_t, ax, ay);

#if GRAPHICS
        if (graphics) {
            visualize(particles, N);
            usleep(3000);  // Refresh rate
            if (CheckForQuit()) break;
        }
#endif

    }

    write_output_file("result.gal", particles, N);

#if GRAPHICS
    if (graphics) {
        FlushDisplay();
        CloseDisplay();
    }
#endif

    free(particles);
    free(ax);
    free(ay);
}

void visualize(Particle *particles, int N) {
    ClearScreen();

    for (int i = 0; i < N; i++) {
        DrawCircle(particles[i].x, particles[i].y, 1, 1, 0.005, 0);
    }

    Refresh();
}