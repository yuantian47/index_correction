#pragma once
#ifndef RAYCAST_FILTER_CUH
#define RAYCAST_FILTER_CUH

#include "raycast.hpp"

struct uint3 {
    unsigned int x, y, z;
};

void median_filter(size_t order, char *out, const char *in, extent_t shape);

//Filters using a gaussian kernel
#define ROWS_Y 8
#define COLS_X 32
#define ROWS_X 8
#define COLS_Y 32
#define PIX_PER_THREAD 8
#define KERNEL_RADIUS 2
//gauss filter defines
static const float gaussK[2 * KERNEL_RADIUS + 1] = { 0.1117f, 0.2365f, 0.3036f, 0.2365f, 0.1117f };
void convolutionRows(char* output, const char* input, int imageW, int imageH, size_t pitch);

//void convolutionCols(cudaSurfaceObject_t surf, char* input, int imageW, int imageH, size_t pitch, size_t zLoc);

#endif
