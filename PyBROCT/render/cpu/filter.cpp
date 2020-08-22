#include "filter.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>

/**********************************************Filter Methods***************************************************/

void median_filter(size_t order, char *out, const char *in, extent_t extent) {

#pragma omp parallel for
    for(ptrdiff_t x = 0; x < extent.width; x++) {
        std::vector<char> buf(order * order * order);

        for(size_t y = 0; y < extent.height; y++) {
            for(size_t z = 0; z < extent.depth; z++) {

                size_t n = 0;
                
                for(size_t k = z > 0 ? z - 1 : z; k <= (z + 1 < extent.depth ? z + 1 : z); k++) {
                    for(size_t j = y > 0 ? y - 1 : y; j <= (y + 1 < extent.height ? y + 1 : y); j++) {
                        for(size_t i = x > 0 ? x - 1 : x; i <= (x + 1 < extent.width ? x + 1 : x); i++) {
                            buf[n++] = in[k * extent.height * extent.width + j * extent.width + i];
                        }
                    }
                }

                std::nth_element(buf.begin(), buf.begin() + n / 2, buf.end());
                out[z * extent.height * extent.width + y * extent.width + x] = buf[n / 2];
            }
        }
    }
}
////
////
////
//////////////////////////////////////////////////////////////////////
//////Row convolution
//////////////////////////////////////////////////////////////////////
////
////void launch_convolutionRows(const dim3& blocks, const dim3& threads, char* output, const char* input, int imageW, int imageH, size_t pitch) {
////    convolutionRows<<<blocks, threads>>>(output, input, imageW, imageH, pitch);
////}
////
////__global__ void convolutionRows(char* output, const char* input, int imageW, int imageH, size_t pitch) {
////
////    __shared__ float sData[ROWS_Y][(COLS_X*PIX_PER_THREAD) + (2 * KERNEL_RADIUS)];
////
////
////    const int x = blockIdx.x * (COLS_X*PIX_PER_THREAD) + threadIdx.x;
////    const int y = blockIdx.height * blockDim.y + threadIdx.y;
////
////    //check to see if we are guaranteed to be within the image bounds
////    if ((y < imageH) && (x + (COLS_X * PIX_PER_THREAD) < imageW)) {
////
////
////        //load left overlap
////        if (threadIdx.x == 0) {
////            for (int n = KERNEL_RADIUS; n >= 1; n--) {
////                sData[threadIdx.y][threadIdx.x + (-n + KERNEL_RADIUS)] = ((x - n) > 0) ? input[y * pitch + x - n] : 0;
////            }
////        }
////
////
////
////        //load PIX_PER_THREAD values
////#pragma unroll
////        for (int i = 0; i < PIX_PER_THREAD; i++) {
////            sData[threadIdx.y][threadIdx.x + (COLS_X * i) + KERNEL_RADIUS] = input[y * pitch + x + (COLS_X * i)];
////        }
////
////
////        //load in right overlap
////        for (int n = 1; n <= KERNEL_RADIUS; n++) {
////            sData[threadIdx.y][(COLS_X * PIX_PER_THREAD) + KERNEL_RADIUS + n - 1] = (blockIdx.x * (COLS_X*PIX_PER_THREAD) + (COLS_X * PIX_PER_THREAD) + n < imageW) ? input[y * pitch + blockIdx.x * (COLS_X*PIX_PER_THREAD) + (COLS_X * PIX_PER_THREAD) + (n - 1)] : 0;
////        }
////
////
////        __syncthreads();
////
////        //do the convolution
////        //#pragma unroll
////        for (int i = 0; i < PIX_PER_THREAD; i++) {
////            float sum = 0;
////            //#pragma unroll
////            for (int n = -KERNEL_RADIUS; n <= KERNEL_RADIUS; n++) {
////                sum += gaussK[KERNEL_RADIUS - n] * (float)sData[threadIdx.y][threadIdx.x + i * COLS_X + n + KERNEL_RADIUS];
////            }
////
////            output[y * pitch + x + (COLS_X * i)] = sum;
////        }
////
////        //we are not guaranteed to be within the x image bounds
////    }
////    else if (y < imageH) {
////
////        //load left overlap (should evaluate to input[y*pitch+x-1] unless image is less than COLS_X*PIX_PER_THREAD wide)
////        if (threadIdx.x == 0) {
////            for (int n = KERNEL_RADIUS; n >= 1; n--) {
////                sData[threadIdx.y][threadIdx.x + (-n + KERNEL_RADIUS)] = ((x - n) > 0) ? input[y * pitch + x - n] : 0;
////            }
////        }
////
////        //load as many values as allowed fill the rest with zeros
////        //#pragma unroll
////        for (int i = 0; i < PIX_PER_THREAD; i++) {
////            if (x + (COLS_X * i) < imageW) {
////                sData[threadIdx.y][threadIdx.x + (COLS_X * i) + KERNEL_RADIUS] = input[y * pitch + x + (COLS_X * i)];
////            }
////            else {
////                //load in zeros for the rest of the shared memory
////                sData[threadIdx.y][threadIdx.x + (COLS_X * i) + KERNEL_RADIUS] = 0;
////            }
////        }
////        //load in right overlap
////        for (int n = 1; n <= KERNEL_RADIUS; n++) {
////            sData[threadIdx.y][(COLS_X * PIX_PER_THREAD) + KERNEL_RADIUS + n - 1] = (blockIdx.x * (COLS_X*PIX_PER_THREAD) + (COLS_X * PIX_PER_THREAD) + n < imageW) ? input[y * pitch + blockIdx.x * (COLS_X*PIX_PER_THREAD) + (COLS_X * PIX_PER_THREAD) + (n - 1)] : 0;
////        }
////
////        __syncthreads();
////
////        //do the convolution
////        //#pragma unroll
////        for (int i = 0; i < PIX_PER_THREAD; i++) {
////            float sum = 0;
////            if (x + (COLS_X * i) < imageW) {
////
////                //#pragma unroll
////                for (int n = -KERNEL_RADIUS; n <= KERNEL_RADIUS; n++) {
////                    sum += gaussK[KERNEL_RADIUS - n] * (float)sData[threadIdx.y][threadIdx.x + i * COLS_X + n + KERNEL_RADIUS];
////                }
////
////                output[y * pitch + x + (COLS_X * i)] = sum;
////            }
////            else {
////                break;
////            }
////        }
////    }
////}
////
////
//////////////////////////////////////////////////////////////////////
//////Column convolution
//////////////////////////////////////////////////////////////////////
////
////void launch_convolutionCols(const dim3& blocks, const dim3& threads, cudaSurfaceObject_t surf, char* input, int imageW, int imageH, size_t pitch, size_t zLoc) {
////    convolutionCols<<<blocks, threads>>>(surf, input, imageW, imageH, pitch, zLoc);
////}
////
////__global__ void convolutionCols(cudaSurfaceObject_t surf, char* input, int imageW, int imageH, size_t pitch, size_t zLoc) {
////
////    __shared__ float sData[COLS_Y*PIX_PER_THREAD + 2 * KERNEL_RADIUS][ROWS_X];
////
////    //const uint32_t zLoc = z;// bscanLocations[z];
////
////    const int x = blockIdx.x * blockDim.x + threadIdx.x;
////    const int y = blockIdx.y * (COLS_Y*PIX_PER_THREAD) + threadIdx.y;
////
////    //check to see if we are guaranteed to be within the image bounds
////    if ((x < imageW) && (y + (COLS_Y * PIX_PER_THREAD) < imageH)) {
////
////
////        //load top overlap
////        if (threadIdx.y == 0) {
////            for (int n = KERNEL_RADIUS; n >= 1; n--) {
////                sData[threadIdx.y + (-n + KERNEL_RADIUS)][threadIdx.x] = ((y - n) > 0) ? input[(y - n) * pitch + x] : 0;
////            }
////        }
////        //load PIX_PER_THREAD values
////#pragma unroll
////        for (int i = 0; i < PIX_PER_THREAD; i++) {
////            sData[threadIdx.y + (COLS_Y * i) + KERNEL_RADIUS][threadIdx.x] = input[(y + (COLS_Y * i)) * pitch + x];
////        }
////
////
////        //load in bottom overlap
////        for (int n = 1; n <= KERNEL_RADIUS; n++) {
////            sData[(COLS_Y * PIX_PER_THREAD) + KERNEL_RADIUS + n - 1][threadIdx.x] = (blockIdx.y * (COLS_Y*PIX_PER_THREAD) + (COLS_Y * PIX_PER_THREAD) + n < imageH) ? input[(blockIdx.y * (COLS_Y*PIX_PER_THREAD) + (COLS_Y * PIX_PER_THREAD) + (n - 1)) * pitch + x] : 0;
////        }
////
////
////        __syncthreads();
////
////        //do the convolution
////#pragma unroll
////        for (int i = 0; i < PIX_PER_THREAD; i++) {
////            float sum = 0;
////#pragma unroll
////            for (int n = -KERNEL_RADIUS; n <= KERNEL_RADIUS; n++) {
////                sum += gaussK[KERNEL_RADIUS - n] * (float)sData[threadIdx.y + i * COLS_Y + n + KERNEL_RADIUS][threadIdx.x];
////            }
////
////            //output[(y + (COLS_Y * i)) * pitch + x] = sum;
////            //write out the the texture
////            //float tempVol = 255 * (sum - 40) / (110 - 40); //works for logged values
////
////            //if (tempVol < 0) {
////            //    tempVol = 0.0;
////            //}
////            //if (tempVol > 255) {
////            //    tempVol = 255.;
////            //}
////
////            ////why does writing out tempvol not work?
////            //output[(y + (COLS_Y * i)) * pitch + x] = tempVol;
////            surf3Dwrite((unsigned char)sum, surf, x, (y + (COLS_Y * i)), zLoc);
////        }
////
////        //we are not guaranteed to be within the x image bounds
////    }
////    else if (x < imageW) {
////
////        //load left overlap (should evaluate to input[(y-1)*pitch+x] unless image is less than COLS_Y*PIX_PER_THREAD high)
////        if (threadIdx.y == 0) {
////            for (int n = KERNEL_RADIUS; n >= 1; n--) {
////                sData[threadIdx.y + (-n + KERNEL_RADIUS)][threadIdx.x] = ((y - n) > 0) ? input[(y - n) * pitch + x] : 0;
////            }
////        }
////
////        //load as many values as allowed fill the rest with zeros
////#pragma unroll
////        for (int i = 0; i < PIX_PER_THREAD; i++) {
////            if (y + (COLS_Y * i)  < imageH) {
////                sData[threadIdx.y + (COLS_Y * i) + KERNEL_RADIUS][threadIdx.x] = input[(y + (COLS_Y * i)) * pitch + x];
////            }
////            else {
////                //load in zeros for the rest of the shared memory
////                sData[threadIdx.y + (COLS_Y * i) + KERNEL_RADIUS][threadIdx.x] = 0;
////            }
////        }
////
////        //load in bottom overlap
////
////        for (int n = 1; n <= KERNEL_RADIUS; n++) {
////            sData[(COLS_Y * PIX_PER_THREAD) + KERNEL_RADIUS + n - 1][threadIdx.x] = (blockIdx.y * (COLS_Y*PIX_PER_THREAD) + (COLS_Y * PIX_PER_THREAD) + n < imageH) ? input[(blockIdx.y * (COLS_Y*PIX_PER_THREAD) + (COLS_Y * PIX_PER_THREAD) + (n - 1)) * pitch + x] : 0;
////        }
////
////        __syncthreads();
////
////        //do the convolution
////#pragma unroll
////        for (int i = 0; i < PIX_PER_THREAD; i++) {
////            float sum = 0;
////            if (y + (COLS_Y * i) < imageH) {
////
////#pragma unroll
////                for (int n = -KERNEL_RADIUS; n <= KERNEL_RADIUS; n++) {
////                    sum += gaussK[KERNEL_RADIUS - n] * (float)sData[threadIdx.y + i * COLS_Y + n + KERNEL_RADIUS][threadIdx.x];
////                }
////
////                //write out to the texture
////               /* float tempVol = 255 * (sum - 40) / (110 - 40);
////
////                if (tempVol < 0) {
////                    tempVol = 0.0;
////                }
////                if (tempVol > 255) {
////                    tempVol = 255.;
////                }*/
////                //output[(y + (COLS_Y * i)) * pitch + x] = sum;
////                surf3Dwrite((unsigned char)sum, surf, x, (y + (COLS_Y * i)), zLoc);
////
////            }
////            else {
////                break;
////            }
////        }
////    }
////}
