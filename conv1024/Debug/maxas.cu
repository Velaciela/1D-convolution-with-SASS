// Note this file isn't configured to automatically compile.
// Here's how:

// If you want to look at the ptx first:
// nvcc -arch sm_50 -m 32 -ptx sgemm.cu

// Manually compile your kernel to a cubin.
// You should only have to do this once, unless you change params or shared size or globals:
// nvcc -arch sm_50 -m 32 -cubin sgemm.cu

// If tweaking a kernel or writing a new one based on this shell code you would then do this:
// maxas.pl -e kernel.cubin kernel.sass

// I've already included a modified kernel (sgemm.sass) so the next step is..

// Splice the manually assembled code back into the cubin:
// maxas.pl -i sgemm.sass sgemm.cubin

#include <device_functions.h>
#include <device_launch_parameters.h>

// Use extern C so C++ doesn't mangle our kernel name
extern "C"
// This kernel requires 32x1x1 threads per block
__global__ void __launch_bounds__(128) conv_kernel_128(
float *H, float *X, float *Y)
{
	// Declare any shared memory your kernel requires
	// Or you could just pass the amount in as a param to cuLaunchKernel

	__shared__ float share[4096];
	
	int tid = threadIdx.x;
	int bx  = blockIdx.x;

	share[tid] = 1.0f;


	// output something so your setup isn't optimized away.
	Y[tid] = share[tid + bx] ;

}


