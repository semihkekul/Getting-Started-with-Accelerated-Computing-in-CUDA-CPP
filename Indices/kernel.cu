#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

    if (threadIdx.x == 1023 && blockIdx.x == 255)
    {
        printf("Success!\n");
    }

}

int main()
{
    /*
     * Update the execution configuration so that the kernel
     * will print `"Success!"`.
     */

    printSuccessForCorrectExecutionConfiguration <<<256, 1024 >> > ();
    cudaDeviceSynchronize();
    return 0;
}
