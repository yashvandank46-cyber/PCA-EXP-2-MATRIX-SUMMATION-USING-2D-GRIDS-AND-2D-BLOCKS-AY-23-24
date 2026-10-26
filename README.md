# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>AIM:</h3>
<h3>ENTER YOUR NAME: YASHVANDAN K</h3>
<h3>ENTER YOUR REGISTER NO: 212225100060</h3>
<h3>EX. NO: 2</h3>
<h3>DATE: 26/10/2026</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```
%%writefile sumMatrixOnGPU-2D-grid-2D-block.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// CUDA error checking macro
#define CHECK(call)                                                          \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);               \
        fprintf(stderr, "code: %d, reason: %s\n",                            \
                error, cudaGetErrorString(error));                           \
        exit(1);                                                             \
    }                                                                        \
}

// Timer function
double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6);
}

// Initialize matrix with random integer values
void initialData(int *ip, const int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        ip[i] = (int)(rand() & 0xFF);
    }
}

// Host (CPU) 2D Matrix Summation
void sumMatrixOnHost(int *A, int *B, int *C, const int nx, const int ny)
{
    int *ia = A;
    int *ib = B;
    int *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

// Compare CPU and GPU results
void checkResult(int *hostRef, int *gpuRef, const int N)
{
    bool match = true;
    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            match = false;
            printf("Host %d GPU %d at index %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays DO NOT match!\n\n");
}

// Kernel for 2D Matrix Addition using 2D Grid and 2D Blocks
__global__ void sumMatrixOnGPU2D(int *A, int *B, int *C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    printf("Matrix Summation on GPU with 2D Grid and 2D Blocks\n");

    // Set up GPU device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Matrix dimensions (16384 x 16384 elements)
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Allocate host memory
    int *h_A = (int *)malloc(nBytes);
    int *h_B = (int *)malloc(nBytes);
    int *hostRef = (int *)malloc(nBytes);
    int *gpuRef = (int *)malloc(nBytes);

    // Initialize host data
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // Host (CPU) computation
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    double cpuTime = seconds() - iStart;
    printf("sumMatrixOnHost elapsed time: %f sec\n\n", cpuTime);

    // Allocate device memory
    int *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // Transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // Test different 2D block configurations to find the best execution setup
    int blockSizes[][2] = {
        {32, 32},
        {32, 16},
        {16, 32},
        {16, 16},
        {8, 8}
    };
    int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);

    for (int i = 0; i < numConfigs; i++)
    {
        int dimx = blockSizes[i][0];
        int dimy = blockSizes[i][1];

        dim3 block(dimx, dimy);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

        iStart = seconds();
        sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
        CHECK(cudaDeviceSynchronize());
        double gpuTime = seconds() - iStart;

        printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed time: %f sec\n",
               grid.x, grid.y, block.x, block.y, gpuTime);

        // Verify result from GPU
        CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
        checkResult(hostRef, gpuRef, nxy);
    }

    // Free device memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // Reset device
    CHECK(cudaDeviceReset());

    return 0;
}
```

## OUTPUT:
<img width="1078" height="497" alt="Screenshot 2026-10-26 100807" src="https://github.com/user-attachments/assets/f564736f-5637-44f5-963d-9483f7aeca93" />


## RESULT:
The host took 0.584308 seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in 0.012965 seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
