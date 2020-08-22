#include "filter.cuh"

#include <cstdint>

/**********************************************Filter Methods***************************************************/

void launch_medKernel(const dim3& blocks, const dim3& threads, char *out, const char *in, size_t dimy, size_t dimx, size_t dimz, size_t pitch) {
    medKernel<<<blocks, threads>>>(out, in, dimy, dimx, dimz, pitch);
}

__global__ void medKernel(char *out, const char *in, size_t dimy, size_t dimx, size_t dimz, size_t pitch) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    __shared__ char im[blockSizeRow + 2][blockSizeCol + 2][3];

    if ((threadIdx.y + blockIdx.y * blockDim.y < dimy - 2) && (threadIdx.x + blockIdx.x * blockDim.x < dimx - 2)) {

        im[row + 1][col + 1][0] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch);
        im[row + 1][col + 1][1] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy);
        im[row + 1][col + 1][2] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);

        if (threadIdx.x == 0) {
            im[row + 1][col][0] = *(in + (blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch);
            im[row + 1][col][1] = *(in + (blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy);
            im[row + 1][col][2] = *(in + (blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);
        }

        if ((threadIdx.x == blockDim.x - 1) || (threadIdx.x + blockIdx.x * blockDim.x == dimx - 3)) {
            im[row + 1][col + 2][0] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch);
            im[row + 1][col + 2][1] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy);
            im[row + 1][col + 2][2] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);
        }

        if (threadIdx.y == 0) {
            im[row][col + 1][0] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch);
            im[row][col + 1][1] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy);
            im[row][col + 1][2] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);
        }

        if ((threadIdx.y == blockDim.y - 1) || (threadIdx.y + blockIdx.y * blockDim.y == dimy - 3)) {
            im[row + 2][col + 1][0] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch);
            im[row + 2][col + 1][1] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy);
            im[row + 2][col + 1][2] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);
        }

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            im[row][col][0] = *(in + (blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch);
            im[row][col][1] = *(in + (blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy);
            im[row][col][2] = *(in + (blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);
        }

        if ((threadIdx.x == blockDim.x - 1 || (threadIdx.x + blockIdx.x * blockDim.x == dimx - 3)) && threadIdx.y == 0) {
            im[row][col + 2][0] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch);
            im[row][col + 2][1] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy);
            im[row][col + 2][2] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);
        }

        if (threadIdx.x == 0 && (threadIdx.y == blockDim.y - 1 || (threadIdx.y + blockIdx.y * blockDim.y == dimy - 3))) {
            im[row + 2][col][0] = *(in + (blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch);
            im[row + 2][col][1] = *(in + (blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy);
            im[row + 2][col][2] = *(in + (blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);
        }

        if ((threadIdx.x == blockDim.x - 1 || (threadIdx.x + blockIdx.x * blockDim.x == dimx - 3)) && (threadIdx.y == blockDim.y - 1 || (threadIdx.y + blockIdx.y * blockDim.y == dimy - 3))) {
            im[row + 2][col + 2][0] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch);
            im[row + 2][col + 2][1] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy);
            im[row + 2][col + 2][2] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * 2);
        }


    }
    //
    __syncthreads();

    if ((threadIdx.y + blockIdx.y * blockDim.y < dimy - 2) && (threadIdx.x + blockIdx.x * blockDim.x < dimx - 2)) {
        char a0 = im[row][col][0];
        char a1 = im[row + 1][col][0];
        char a2 = im[row + 2][col][0];
        char a3 = im[row][col + 1][0];
        char a4 = im[row + 1][col + 1][0];
        char a5 = im[row + 2][col + 1][0];
        char a6 = im[row][col + 2][0];
        char a7 = im[row + 1][col + 2][0];
        char a8 = im[row + 2][col + 2][0];
        char a9 = im[row][col][1];
        char a10 = im[row + 1][col][1];
        char a11 = im[row + 2][col][1];
        char a12 = im[row][col + 1][1];
        char a13 = im[row + 1][col + 1][1];
        char a14 = im[row + 2][col + 1][1];

        minmax15(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row][col + 2][1];
        minmax14(&a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row + 1][col + 2][1];
        minmax13(&a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row + 2][col + 2][1];
        minmax12(&a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row][col][2];
        minmax11(&a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row + 1][col][2];
        minmax10(&a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row + 2][col][2];
        minmax9(&a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row][col + 1][2];
        minmax8(&a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row + 1][col + 1][2];
        minmax7(&a8, &a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row + 2][col + 1][2];
        minmax6(&a9, &a10, &a11, &a12, &a13, &a14);
        a14 = im[row][col + 2][2];
        minmax5(&a10, &a11, &a12, &a13, &a14);
        a14 = im[row + 1][col + 2][2];
        minmax4(&a11, &a12, &a13, &a14);
        a14 = im[row + 2][col + 2][2];
        minmax3(&a12, &a13, &a14);

        *(out + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch) = a13;

    }

    for (int i = 1; i < dimz - 1; i++) {
        int frame = (i - 1) % 3;

        if ((threadIdx.y + blockIdx.y * blockDim.y < dimy - 2) && (threadIdx.x + blockIdx.x * blockDim.x < dimx - 2)) {

            im[row + 1][col + 1][frame] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));

            if (threadIdx.x == 0) {

                im[row + 1][col][frame] = *(in + (blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));
            }

            if ((threadIdx.x == blockDim.x - 1) || (threadIdx.x + blockIdx.x * blockDim.x == dimx - 3)) {

                im[row + 1][col + 2][frame] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));

            }

            if (threadIdx.y == 0) {

                im[row][col + 1][frame] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));

            }

            if ((threadIdx.y == blockDim.y - 1) || (threadIdx.y + blockIdx.y * blockDim.y == dimy - 3)) {

                im[row + 2][col + 1][frame] = *(in + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));

            }

            if (threadIdx.x == 0 && threadIdx.y == 0) {

                im[row][col][frame] = *(in + (blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));

            }

            if ((threadIdx.x == blockDim.x - 1 || (threadIdx.x + blockIdx.x * blockDim.x == dimx - 3)) && threadIdx.y == 0) {

                im[row][col + 2][frame] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));

            }

            if (threadIdx.x == 0 && (threadIdx.y == blockDim.y - 1 || (threadIdx.y + blockIdx.y * blockDim.y == dimy - 3))) {

                im[row + 2][col][frame] = *(in + (blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));

            }

            if ((threadIdx.x == blockDim.x - 1 || (threadIdx.x + blockIdx.x * blockDim.x == dimx - 3)) && (threadIdx.y == blockDim.y - 1 || (threadIdx.y + blockIdx.y * blockDim.y == dimy - 3))) {

                im[row + 2][col + 2][frame] = *(in + (2 + threadIdx.x + blockIdx.x * blockDim.x) + (2 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch * dimy * (1 + i));

            }


        }

        __syncthreads();

        if ((threadIdx.y + blockIdx.y * blockDim.y < dimy - 2) && (threadIdx.x + blockIdx.x * blockDim.x < dimx - 2)) {
            char a0 = im[row][col][0];
            char a1 = im[row + 1][col][0];
            char a2 = im[row + 2][col][0];
            char a3 = im[row][col + 1][0];
            char a4 = im[row + 1][col + 1][0];
            char a5 = im[row + 2][col + 1][0];
            char a6 = im[row][col + 2][0];
            char a7 = im[row + 1][col + 2][0];
            char a8 = im[row + 2][col + 2][0];
            char a9 = im[row][col][1];
            char a10 = im[row + 1][col][1];
            char a11 = im[row + 2][col][1];
            char a12 = im[row][col + 1][1];
            char a13 = im[row + 1][col + 1][1];
            char a14 = im[row + 2][col + 1][1];

            minmax15(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row][col + 2][1];
            minmax14(&a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row + 1][col + 2][1];
            minmax13(&a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row + 2][col + 2][1];
            minmax12(&a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row][col][2];
            minmax11(&a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row + 1][col][2];
            minmax10(&a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row + 2][col][2];
            minmax9(&a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row][col + 1][2];
            minmax8(&a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row + 1][col + 1][2];
            minmax7(&a8, &a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row + 2][col + 1][2];
            minmax6(&a9, &a10, &a11, &a12, &a13, &a14);
            a14 = im[row][col + 2][2];
            minmax5(&a10, &a11, &a12, &a13, &a14);
            a14 = im[row + 1][col + 2][2];
            minmax4(&a11, &a12, &a13, &a14);
            a14 = im[row + 2][col + 2][2];
            minmax3(&a12, &a13, &a14);

            *(out + (1 + threadIdx.x + blockIdx.x * blockDim.x) + (1 + threadIdx.y + blockIdx.y * blockDim.y) * pitch + pitch*dimy*i) = a13;

            __syncthreads();

        }
    }
}



//////////////////////////////////////////////////////////////////
//Row convolution
//////////////////////////////////////////////////////////////////

void launch_convolutionRows(const dim3& blocks, const dim3& threads, char* output, const char* input, size_t imageW, size_t imageH, size_t pitch) {
    convolutionRows<<<blocks, threads>>>(output, input, imageW, imageH, pitch);
}

__global__ void convolutionRows(char* output, const char* input, size_t imageW, size_t imageH, size_t pitch) {

    __shared__ float sData[ROWS_Y][(COLS_X*PIX_PER_THREAD) + (2 * KERNEL_RADIUS)];


    const int x = blockIdx.x * (COLS_X*PIX_PER_THREAD) + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    //check to see if we are guaranteed to be within the image bounds
    if ((y < imageH) && (x + (COLS_X * PIX_PER_THREAD) < imageW)) {


        //load left overlap
        if (threadIdx.x == 0) {
            for (int n = KERNEL_RADIUS; n >= 1; n--) {
                sData[threadIdx.y][threadIdx.x + (-n + KERNEL_RADIUS)] = ((x - n) > 0) ? input[y * pitch + x - n] : 0;
            }
        }



        //load PIX_PER_THREAD values
#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD; i++) {
            sData[threadIdx.y][threadIdx.x + (COLS_X * i) + KERNEL_RADIUS] = input[y * pitch + x + (COLS_X * i)];
        }


        //load in right overlap
        for (int n = 1; n <= KERNEL_RADIUS; n++) {
            sData[threadIdx.y][(COLS_X * PIX_PER_THREAD) + KERNEL_RADIUS + n - 1] = (blockIdx.x * (COLS_X*PIX_PER_THREAD) + (COLS_X * PIX_PER_THREAD) + n < imageW) ? input[y * pitch + blockIdx.x * (COLS_X*PIX_PER_THREAD) + (COLS_X * PIX_PER_THREAD) + (n - 1)] : 0;
        }


        __syncthreads();

        //do the convolution
        //#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD; i++) {
            float sum = 0;
            //#pragma unroll
            for (int n = -KERNEL_RADIUS; n <= KERNEL_RADIUS; n++) {
                sum += gaussK[KERNEL_RADIUS - n] * (float)sData[threadIdx.y][threadIdx.x + i * COLS_X + n + KERNEL_RADIUS];
            }

            output[y * pitch + x + (COLS_X * i)] = sum;
        }

        //we are not guaranteed to be within the x image bounds
    }
    else if (y < imageH) {

        //load left overlap (should evaluate to input[y*pitch+x-1] unless image is less than COLS_X*PIX_PER_THREAD wide)
        if (threadIdx.x == 0) {
            for (int n = KERNEL_RADIUS; n >= 1; n--) {
                sData[threadIdx.y][threadIdx.x + (-n + KERNEL_RADIUS)] = ((x - n) > 0) ? input[y * pitch + x - n] : 0;
            }
        }

        //load as many values as allowed fill the rest with zeros
        //#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD; i++) {
            if (x + (COLS_X * i) < imageW) {
                sData[threadIdx.y][threadIdx.x + (COLS_X * i) + KERNEL_RADIUS] = input[y * pitch + x + (COLS_X * i)];
            }
            else {
                //load in zeros for the rest of the shared memory
                sData[threadIdx.y][threadIdx.x + (COLS_X * i) + KERNEL_RADIUS] = 0;
            }
        }
        //load in right overlap
        for (int n = 1; n <= KERNEL_RADIUS; n++) {
            sData[threadIdx.y][(COLS_X * PIX_PER_THREAD) + KERNEL_RADIUS + n - 1] = (blockIdx.x * (COLS_X*PIX_PER_THREAD) + (COLS_X * PIX_PER_THREAD) + n < imageW) ? input[y * pitch + blockIdx.x * (COLS_X*PIX_PER_THREAD) + (COLS_X * PIX_PER_THREAD) + (n - 1)] : 0;
        }

        __syncthreads();

        //do the convolution
        //#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD; i++) {
            float sum = 0;
            if (x + (COLS_X * i) < imageW) {

                //#pragma unroll
                for (int n = -KERNEL_RADIUS; n <= KERNEL_RADIUS; n++) {
                    sum += gaussK[KERNEL_RADIUS - n] * (float)sData[threadIdx.y][threadIdx.x + i * COLS_X + n + KERNEL_RADIUS];
                }

                output[y * pitch + x + (COLS_X * i)] = sum;
            }
            else {
                break;
            }
        }
    }
}


//////////////////////////////////////////////////////////////////
//Column convolution
//////////////////////////////////////////////////////////////////

void launch_convolutionCols(const dim3& blocks, const dim3& threads, cudaSurfaceObject_t surf, char* input, size_t imageW, size_t imageH, size_t pitch, size_t zLoc) {
    convolutionCols<<<blocks, threads>>>(surf, input, imageW, imageH, pitch, zLoc);
}

__global__ void convolutionCols(cudaSurfaceObject_t surf, char* input, size_t imageW, size_t imageH, size_t pitch, size_t zLoc) {

    __shared__ float sData[COLS_Y*PIX_PER_THREAD + 2 * KERNEL_RADIUS][ROWS_X];

    //const uint32_t zLoc = z;// bscanLocations[z];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * (COLS_Y*PIX_PER_THREAD) + threadIdx.y;

    //check to see if we are guaranteed to be within the image bounds
    if ((x < imageW) && (y + (COLS_Y * PIX_PER_THREAD) < imageH)) {


        //load top overlap
        if (threadIdx.y == 0) {
            for (int n = KERNEL_RADIUS; n >= 1; n--) {
                sData[threadIdx.y + (-n + KERNEL_RADIUS)][threadIdx.x] = ((y - n) > 0) ? input[(y - n) * pitch + x] : 0;
            }
        }
        //load PIX_PER_THREAD values
#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD; i++) {
            sData[threadIdx.y + (COLS_Y * i) + KERNEL_RADIUS][threadIdx.x] = input[(y + (COLS_Y * i)) * pitch + x];
        }


        //load in bottom overlap
        for (int n = 1; n <= KERNEL_RADIUS; n++) {
            sData[(COLS_Y * PIX_PER_THREAD) + KERNEL_RADIUS + n - 1][threadIdx.x] = (blockIdx.y * (COLS_Y*PIX_PER_THREAD) + (COLS_Y * PIX_PER_THREAD) + n < imageH) ? input[(blockIdx.y * (COLS_Y*PIX_PER_THREAD) + (COLS_Y * PIX_PER_THREAD) + (n - 1)) * pitch + x] : 0;
        }


        __syncthreads();

        //do the convolution
#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD; i++) {
            float sum = 0;
#pragma unroll
            for (int n = -KERNEL_RADIUS; n <= KERNEL_RADIUS; n++) {
                sum += gaussK[KERNEL_RADIUS - n] * (float)sData[threadIdx.y + i * COLS_Y + n + KERNEL_RADIUS][threadIdx.x];
            }

            //output[(y + (COLS_Y * i)) * pitch + x] = sum;
            //write out the the texture
            //float tempVol = 255 * (sum - 40) / (110 - 40); //works for logged values

            //if (tempVol < 0) {
            //    tempVol = 0.0;
            //}
            //if (tempVol > 255) {
            //    tempVol = 255.;
            //}

            ////why does writing out tempvol not work?
            //output[(y + (COLS_Y * i)) * pitch + x] = tempVol;
            surf3Dwrite((unsigned char)sum, surf, x, (y + (COLS_Y * i)), zLoc);
        }

        //we are not guaranteed to be within the x image bounds
    }
    else if (x < imageW) {

        //load left overlap (should evaluate to input[(y-1)*pitch+x] unless image is less than COLS_Y*PIX_PER_THREAD high)
        if (threadIdx.y == 0) {
            for (int n = KERNEL_RADIUS; n >= 1; n--) {
                sData[threadIdx.y + (-n + KERNEL_RADIUS)][threadIdx.x] = ((y - n) > 0) ? input[(y - n) * pitch + x] : 0;
            }
        }

        //load as many values as allowed fill the rest with zeros
#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD; i++) {
            if (y + (COLS_Y * i)  < imageH) {
                sData[threadIdx.y + (COLS_Y * i) + KERNEL_RADIUS][threadIdx.x] = input[(y + (COLS_Y * i)) * pitch + x];
            }
            else {
                //load in zeros for the rest of the shared memory
                sData[threadIdx.y + (COLS_Y * i) + KERNEL_RADIUS][threadIdx.x] = 0;
            }
        }

        //load in bottom overlap

        for (int n = 1; n <= KERNEL_RADIUS; n++) {
            sData[(COLS_Y * PIX_PER_THREAD) + KERNEL_RADIUS + n - 1][threadIdx.x] = (blockIdx.y * (COLS_Y*PIX_PER_THREAD) + (COLS_Y * PIX_PER_THREAD) + n < imageH) ? input[(blockIdx.y * (COLS_Y*PIX_PER_THREAD) + (COLS_Y * PIX_PER_THREAD) + (n - 1)) * pitch + x] : 0;
        }

        __syncthreads();

        //do the convolution
#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD; i++) {
            float sum = 0;
            if (y + (COLS_Y * i) < imageH) {

#pragma unroll
                for (int n = -KERNEL_RADIUS; n <= KERNEL_RADIUS; n++) {
                    sum += gaussK[KERNEL_RADIUS - n] * (float)sData[threadIdx.y + i * COLS_Y + n + KERNEL_RADIUS][threadIdx.x];
                }

                //write out to the texture
               /* float tempVol = 255 * (sum - 40) / (110 - 40);

                if (tempVol < 0) {
                    tempVol = 0.0;
                }
                if (tempVol > 255) {
                    tempVol = 255.;
                }*/
                //output[(y + (COLS_Y * i)) * pitch + x] = sum;
                surf3Dwrite((unsigned char)sum, surf, x, (y + (COLS_Y * i)), zLoc);

            }
            else {
                break;
            }
        }
    }
}
