/*
 * Example of how to use graphics routines.
 *
 * Based on code by Oliver Fringer, Stanford University.
 * 
 * Modified by Elias Rudberg, Uppsala University.
 *
 */

#include "graphics.h"
#include <math.h>

const float circleRadius=0.025, circleColor=0;
const int windowWidth=800;

void keep_within_box(float* xA, float* yA) {
  if(*xA > 1)
    *xA = 0;
  if(*yA > 1)
    *yA = 0;
}

int main(int argc, char *argv[]) {
  float L=1, W=1;

  float xA = 0.45;
  float yA = 0.41;
  float xB = 0.32;
  float yB = 0.56;

  InitializeGraphics(argv[0],windowWidth,windowWidth);
  SetCAxes(0,1);

  printf("Hit q to quit.\n");
  while(!CheckForQuit()) {
    /* Move A. */
    xA += 0.0012;
    yA += 0.0020;
    keep_within_box(&xA, &yA);
    /* Move B. */
    xB += 0.0007;
    yB += 0.0018;
    keep_within_box(&xB, &yB);
    /* Call graphics routines. */
    ClearScreen();
    DrawCircle(xA, yA, L, W, circleRadius, circleColor);
    DrawCircle(xB, yB, L, W, circleRadius, circleColor);
    Refresh();
    /* Sleep a short while to avoid screen flickering. */
    usleep(3000);
  }
  FlushDisplay();
  CloseDisplay();
  return 0;
}
