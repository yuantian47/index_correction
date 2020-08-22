#pragma once
#ifndef RAYCAST_FILTER_CUH
#define RAYCAST_FILTER_CUH

#include <cuda_runtime_api.h>

//Median filter defines
__device__ inline void s(char* a, char* b) {
    char tmp;
    if (*a > *b) {
        tmp = *b;
        *b = *a;
        *a = tmp;
    }
}

#define min3(a, b, c)                   s(a,b); s(a,c);
#define max3(a, b, c)                   s(b,c); s(a,c);
#define min4(a, b, c, d)                s(a,b); s(c,d); s(a,c);
#define max4(a, b, c, d)                s(c,d); s(a,b); s(b,d);
#define min5(a, b, c, d, e)             s(a,b); s(c,d); min3(a, c, e);
#define max5(a, b, c, d, e)             s(a,b); s(c,d); max3(b, d, e);
#define min6(a, b, c, d, e, f)          s(a,b); s(c,d); s(e,f); min3(a, c, e);
#define max6(a, b, c, d, e, f)          s(a,b); s(c,d); s(e,f); max3(b, d, f);
#define min7(a, b, c, d, e, f, g)       s(a,b); s(c,d); s(e,f); min4(a, c, e, g);
#define max7(a, b, c, d, e, f, g)       s(a,b); s(c,d); s(e,f); max4(b, d, f, g);
#define min8(a, b, c, d, e, f, g, h)    s(a,b); s(c,d); s(e,f); s(g,h); min4(a, c, e, g);
#define max8(a, b, c, d, e, f, g, h)    s(a,b); s(c,d); s(e,f); s(g,h); max4(b, d, f, h);

#define minmax3(a, b, c)                                         max3(a,b,c); s(a,b);
#define minmax4(a, b, c, d)                                      s(a,b); s(c,d); s(a,c); s(b,d);
#define minmax5(a, b, c, d, e)                                   s(a,b); s(c,d); min3(a, c, e) max3(b, d, e);
#define minmax6(a, b, c, d, e, f)                                s(a,b); s(c,d); s(e,f); min3(a, c, e); max3(b, d, f);
#define minmax7(a, b, c, d, e, f, g)                             s(a,b); s(c,d); s(e,f); min4(a, c, e, g); max4(b, d, f, g);
#define minmax8(a, b, c, d, e, f, g, h)                          s(a,b); s(c,d); s(e,f); s(g,h); min4(a, c, e, g); max4(b, d, f, h);
#define minmax9(a, b, c, d, e, f, g, h, i)                       s(a,b); s(c,d); s(e,f); s(g,h); min5(a, c, e, g, i); max5(b, d, f, h, i);
#define minmax10(a, b, c, d, e, f, g, h, i, j)                   s(a,b); s(c,d); s(e,f); s(g,h); s(i,j); min5(a, c, e, g, i); max5(b, d, f, h, j);
#define minmax11(a, b, c, d, e, f, g, h, i, j, k)                s(a,b); s(c,d); s(e,f); s(g,h); s(i,j); min6(a, c, e, g, i, k); max6(b, d, f, h, j, k);
#define minmax12(a, b, c, d, e, f, g, h, i, j, k, l)             s(a,b); s(c,d); s(e,f); s(g,h); s(i,j); s(k,l); min6(a, c, e, g, i, k); max6(b, d, f, h, j, l);
#define minmax13(a, b, c, d, e, f, g, h, i, j, k, l, m)          s(a,b); s(c,d); s(e,f); s(g,h); s(i,j); s(k,l); min7(a, c, e, g, i, k, m); max7(b, d, f, h, j, l, m);
#define minmax14(a, b, c, d, e, f, g, h, i, j, k, l, m, n)       s(a,b); s(c,d); s(e,f); s(g,h); s(i,j); s(k,l); s(m,n); min7(a, c, e, g, i, k, m); max7(b, d, f, h, j, l, n);
#define minmax15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)    s(a,b); s(c,d); s(e,f); s(g,h); s(i,j); s(k,l); s(m,n); min8(a, c, e, g, i, k, m, o); max8(b, d, f, h, j, l, n, o);


////////////////////Filtering methods////////////////////////////
//Performs 3d median filter on 16 bscans
__global__ void medKernel(char *out, const char *in, size_t dimy, size_t dimx, size_t dimz, size_t pitch);
const int blockSizeRow = 32;
const int blockSizeCol = 16;


//Filters using a gaussian kernel
#define ROWS_Y 8
#define COLS_X 32
#define ROWS_X 8
#define COLS_Y 32
#define PIX_PER_THREAD 8
#define KERNEL_RADIUS 2
//gauss filter defines
static __device__ const float gaussK[2 * KERNEL_RADIUS + 1] = { 0.1117f, 0.2365f, 0.3036f, 0.2365f, 0.1117f };
__global__ void convolutionRows(char* output, const char* input, size_t imageW, size_t imageH, size_t pitch);
__global__ void convolutionCols(cudaSurfaceObject_t surf, char* input, size_t imageW, size_t imageH, size_t pitch, size_t zLoc);

void launch_medKernel(const dim3& blocks, const dim3& threads, char *out, const char *in, size_t dimy, size_t dimx, size_t dimz, size_t pitch);
void launch_convolutionRows(const dim3& blocks, const dim3& threads, char* output, const char* input, size_t imageW, size_t imageH, size_t pitch);
void launch_convolutionCols(const dim3& blocks, const dim3& threads, cudaSurfaceObject_t surf, char* input, size_t imageW, size_t imageH, size_t pitch, size_t zLoc);

#endif
