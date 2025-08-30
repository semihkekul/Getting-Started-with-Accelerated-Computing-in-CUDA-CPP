
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void CPUFunction()
{
	printf("This function is defined to run on the CPU.\n");
}


__global__ void PrintConfiguration()
{
	printf("%d , %d : %d %d \n", gridDim.x, blockDim.x, blockIdx.x , threadIdx.x);
}

/*
- The __global__ keyword indicates that the following function will run on the GPU, and can be invoked globally, which in this context means either by the CPU, or, by the GPU.
- Often, code executed on the CPU is referred to as host code, and code running on the GPU is referred to as device code.
- Notice the return type void. It is required that functions defined with the __global__ keyword return type void.
*/
__global__ void GPUFunction()
{
	printf("This function is defined to run on the GPU.\n");
}

int main()
{
	/*
	- When launching a kernel, we must provide an execution configuration, which is done by using the <<< ... >>> syntax just prior to passing the kernel any expected arguments.
	- At a high level, execution configuration allows programmers to specify the thread hierarchy for a kernel launch, which defines the number of thread groupings (called blocks), as well as how many threads to execute in each block. 
	*/
	GPUFunction <<<2, 2 >>> ();
	/*
	Launching kernels is asynchronous: the CPU code will continue to execute without waiting for the kernel launch to complete. 
	A call to cudaDeviceSynchronize, a function provided by the CUDA runtime, will cause the host (CPU) code to wait until the device (GPU) code completes, and only then resume execution on the CPU.
	*/
	cudaDeviceSynchronize();
	CPUFunction();

	PrintConfiguration << <2, 4 >> > ();

}