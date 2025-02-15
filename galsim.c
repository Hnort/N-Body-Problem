#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "graphics.h"

#define EPSILON 1e-3
#define GRAPHICS 0
#define UNROLLING 1

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
    // for (int i = 0; i < N; i++) {
    //     ax[i] = 0.0;
    //     ay[i] = 0.0;
    // }

    memset(ax, 0, N * sizeof(double));
    memset(ay, 0, N * sizeof(double));

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
#if UNROLLING

    const double EPS2 = EPSILON * EPSILON;

    for (int i = 0; i < N; i++) {
    double xi = particles[i].x;
    double yi = particles[i].y;
    double mi = particles[i].mass;

    for (int j = i + 1; j + 3 < N; j += 4) {
        double dx1 = particles[j].x - xi;
        double dy1 = particles[j].y - yi;
        double r2_1 = dx1 * dx1 + dy1 * dy1 + EPS2;
        double inv_r3_1 = 1.0 / (r2_1 * sqrt(r2_1));
        double F1 = G * mi * particles[j].mass * inv_r3_1;
        double Fx1 = F1 * dx1;
        double Fy1 = F1 * dy1;

        double dx2 = particles[j + 1].x - xi;
        double dy2 = particles[j + 1].y - yi;
        double r2_2 = dx2 * dx2 + dy2 * dy2 + EPS2;
        double inv_r3_2 = 1.0 / (r2_2 * sqrt(r2_2));
        double F2 = G * mi * particles[j + 1].mass * inv_r3_2;
        double Fx2 = F2 * dx2;
        double Fy2 = F2 * dy2;

        double dx3 = particles[j + 2].x - xi;
        double dy3 = particles[j + 2].y - yi;
        double r2_3 = dx3 * dx3 + dy3 * dy3 + EPS2;
        double inv_r3_3 = 1.0 / (r2_3 * sqrt(r2_3));
        double F3 = G * mi * particles[j + 2].mass * inv_r3_3;
        double Fx3 = F3 * dx3;
        double Fy3 = F3 * dy3;

        double dx4 = particles[j + 3].x - xi;
        double dy4 = particles[j + 3].y - yi;
        double r2_4 = dx4 * dx4 + dy4 * dy4 + EPS2;
        double inv_r3_4 = 1.0 / (r2_4 * sqrt(r2_4));
        double F4 = G * mi * particles[j + 3].mass * inv_r3_4;
        double Fx4 = F4 * dx4;
        double Fy4 = F4 * dy4;

        ax[i] += (Fx1 + Fx2 + Fx3 + Fx4) / mi;
        ay[i] += (Fy1 + Fy2 + Fy3 + Fy4) / mi;

        ax[j] -= Fx1 / particles[j].mass;
        ay[j] -= Fy1 / particles[j].mass;

        ax[j + 1] -= Fx2 / particles[j + 1].mass;
        ay[j + 1] -= Fy2 / particles[j + 1].mass;

        ax[j + 2] -= Fx3 / particles[j + 2].mass;
        ay[j + 2] -= Fy3 / particles[j + 2].mass;

        ax[j + 3] -= Fx4 / particles[j + 3].mass;
        ay[j + 3] -= Fy4 / particles[j + 3].mass;
    }

    for (int j = N - (N % 4); j < N; j++) {
        double dx = particles[j].x - xi;
        double dy = particles[j].y - yi;
        double r2 = dx * dx + dy * dy + EPS2;
        double inv_r3 = 1.0 / (r2 * sqrt(r2));
        double F = G * mi * particles[j].mass * inv_r3;
        double Fx = F * dx;
        double Fy = F * dy;

        ax[i] += Fx / mi;
        ay[i] += Fy / mi;

        ax[j] -= Fx / particles[j].mass;
        ay[j] -= Fy / particles[j].mass;
    }
}

#else
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
#endif
}


void update_positions(Particle *particles, const int N, double delta_t, double *ax, double *ay) {
#if UNROLLING
    int i;
    for (i = 0; i + 3 < N; i += 4) {
        particles[i].vx += delta_t * ax[i];
        particles[i].vy += delta_t * ay[i];
        particles[i].x += delta_t * particles[i].vx;
        particles[i].y += delta_t * particles[i].vy;

        particles[i + 1].vx += delta_t * ax[i + 1];
        particles[i + 1].vy += delta_t * ay[i + 1];
        particles[i + 1].x += delta_t * particles[i + 1].vx;
        particles[i + 1].y += delta_t * particles[i + 1].vy;

        particles[i + 2].vx += delta_t * ax[i + 2];
        particles[i + 2].vy += delta_t * ay[i + 2];
        particles[i + 2].x += delta_t * particles[i + 2].vx;
        particles[i + 2].y += delta_t * particles[i + 2].vy;

        particles[i + 3].vx += delta_t * ax[i + 3];
        particles[i + 3].vy += delta_t * ay[i + 3];
        particles[i + 3].x += delta_t * particles[i + 3].vx;
        particles[i + 3].y += delta_t * particles[i + 3].vy;
    }
    for (; i < N; i++) {
        particles[i].vx += delta_t * ax[i];
        particles[i].vy += delta_t * ay[i];
        particles[i].x += delta_t * particles[i].vx;
        particles[i].y += delta_t * particles[i].vy;
    }
#else
    for (int i = 0; i < N; i++) {
        particles[i].vx += delta_t * ax[i];
        particles[i].vy += delta_t * ay[i];
        particles[i].x += delta_t * particles[i].vx;
        particles[i].y += delta_t * particles[i].vy;
    }
#endif
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
        InitializeGraphics("N-Body Simulation", windowWidth, windowWidth);
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
        DrawCircle(particles[i].x, particles[i].y, L, W, particleRadius, 0);
    }

    Refresh();
}