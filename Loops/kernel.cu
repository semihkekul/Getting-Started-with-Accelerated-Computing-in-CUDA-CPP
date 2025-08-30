
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

__global__ void singleBlock()
{
    
    printf("Single iteration number %d\n", threadIdx.x);
    
}

__global__ void multiBlock()
{
    printf("Multi iteration number %d\n", blockIdx.x * blockDim.x  + threadIdx.x);
}

int main()
{
    /*
     * When refactoring `loop` to launch as a kernel, be sure
     * to use the execution configuration to control how many
     * "iterations" to perform.
     *
     * For this exercise, only use 1 block of threads.
     */

    int N = 10;
    singleBlock <<<1, N>>>();

    multiBlock << <2, 5 >> > ();

}
