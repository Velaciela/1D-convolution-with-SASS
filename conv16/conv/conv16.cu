
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <device_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

CUcontext      hContext = 0;

#define CUDA_CHECK( fn ) do { \
		CUresult status = (fn); \
		if ( CUDA_SUCCESS != status ) { \
			const char* errstr; \
			cuGetErrorString(status, &errstr); \
			printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
			exit(EXIT_FAILURE); \
						} \
		} while (0)


void gflops(const char* ident, int N, float ms, int repeat)
{

	float msecPerMatrixMul = ms / repeat;
	double flopsPerMatrixMul = (N+16-1) * 16.0 ;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf("ms = %f \n", msecPerMatrixMul);
	printf("%s GFLOPS: %.2f (size: %d, iterations: %d)\n", ident, gigaFlops, N, repeat);

}



int main()
{
	//-----------------sample_data_config---------------------
	int N = 1024032;//1024032;//1023985;
	int M = 16;//16;
	int P = 1024000;
	size_t sizeSampleFloat = N * 4;
	size_t sizeFilterFloat = M * 4;//16 * 4;
	size_t sizeResultFloat = P * 4;

	dim3 threads(32, 1, 1);
	dim3 grid(2000, 1, 1);

	cudaError_t error;

	char deviceName[32];
	int count, ordinal, major, minor;
	CUdevice  hDevice;
	CUevent hStart, hStop;
	CUdeviceptr devH, devX, devY;


	// ------Initialize the Driver API and find a device-----
	CUDA_CHECK(cuInit(0));
	CUDA_CHECK(cuDeviceGetCount(&count));
	for (ordinal = 0; ordinal < count; ordinal++)
	{
		CUDA_CHECK(cuDeviceGet(&hDevice, ordinal));
		CUDA_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice));
		CUDA_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hDevice));
		CUDA_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), hDevice));
		if (major >= 5 && minor >= 2)
		{
			//printf("Using: Id:%d %s (%d.%d)\n\n", ordinal, deviceName, major, minor);
			break;
		}
	}
	if (ordinal == count)
	{
		printf("No compute 5.0 device found, exiting.\n");
		exit(EXIT_FAILURE);
	}


	//-----------------device_test------------------------

	int device = 0;
	error = cudaSetDevice(0);

	if (error != cudaSuccess)
	{
		printf("device error");
		exit(EXIT_FAILURE);
	}

	else printf("device:  %d  \n", device);

	cudaDeviceProp deviceProp;
	error = cudaGetDeviceProperties(&deviceProp, 0);

	if (error != cudaSuccess)
	{
		printf("DeviceProperties error");
		exit(EXIT_FAILURE);
	}

	//-----------------------host----------------------------

	float* H = (float*)malloc(sizeFilterFloat);
	float* X = (float*)malloc(sizeSampleFloat);
	float* Y = (float*)malloc(sizeResultFloat);
	float* T = (float*)malloc(sizeResultFloat);

	for (int i = 0; i < N ; i++) 
	{
		X[i] = (float)rand()/1000;
	}

	for (int i = 0; i < M; i++) 
	{
		H[i] = (float)rand()/1000;
	}

	for (int i = 0; i < P; i++) //
	{
		Y[i] = (float)0.0;
		T[i] = (float)0.0;
	}


	for (int i = 0; i < P; i++) 
	{
		int k = i;
		for (int j = 16; j > 0; j--)
		{
			T[i] += H[j - 1] * X[k];
			k++;
		}
	}


	//-----------------------Dev----------------------------

	CUDA_CHECK(cuCtxCreate(&hContext, 0, hDevice));

	CUDA_CHECK(cuEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC)); // CU_EVENT_DEFAULT 
	CUDA_CHECK(cuEventCreate(&hStop, CU_EVENT_BLOCKING_SYNC));

	CUDA_CHECK(cuMemAlloc(&devH, sizeFilterFloat));
	CUDA_CHECK(cuMemAlloc(&devX, sizeSampleFloat));
	CUDA_CHECK(cuMemAlloc(&devY, sizeResultFloat));

	CUDA_CHECK(cuMemcpyHtoD(devH, H, sizeFilterFloat));
	CUDA_CHECK(cuMemcpyHtoD(devX, X, sizeSampleFloat));


	//---------------------Kernel----------------------------

	printf("Computing result using CUDA Kernel...\n");

	// Load the cubin
	CUmodule hModule;
	CUDA_CHECK(cuModuleLoad(&hModule, "conv.cubin"));

	// Load the kernel function
	CUfunction hKernel;
	CUDA_CHECK(cuModuleGetFunction(&hKernel, hModule, "conv_kernel_32"));

	void * params[] = {&devH, &devX, &devY};

	int repeat = 20;
	float totalTime = 0;
	// Launch the kernel repeat times.. but break it up into pieces so as not to lock things up.


	CUDA_CHECK(cuEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC)); // CU_EVENT_DEFAULT 
	CUDA_CHECK(cuEventCreate(&hStop, CU_EVENT_BLOCKING_SYNC));


	while (repeat > 0)
	{
		float ms;
		int r = repeat;
		CUDA_CHECK(cuEventRecord(hStart, NULL));

		for (int i = 0; i < repeat; i++)
			CUDA_CHECK(cuLaunchKernel(hKernel, grid.x, 1, 1, threads.x, 1, 1, 0, 0, params, 0));

		CUDA_CHECK(cuEventRecord(hStop, NULL));
		CUDA_CHECK(cuEventSynchronize(hStop));
		CUDA_CHECK(cuEventElapsedTime(&ms, hStart, hStop));

		totalTime += ms;

		//gflops("conv_kernel_32", N, ms, repeat);

		repeat -= r;


	}

	//CUDA_CHECK(cuLaunchKernel(hKernel, grid.x, grid.y, 1, threads.x, 1, 1, 0, 0, params, 0));
	//CUDA_CHECK(cuLaunchKernel(hKernel, grid.x, grid.y, 1, threads.x, 1, 1, 0, 0, params, 0));

	CUDA_CHECK(cuModuleUnload(hModule));

	printf("first time done\n");



	// Copy result from device to host
	CUDA_CHECK(cuMemcpyDtoH(Y, devY, sizeResultFloat));
	CUDA_CHECK(cuMemcpyDtoH(H, devH, sizeFilterFloat));
	CUDA_CHECK(cuMemcpyDtoH(X, devX, sizeSampleFloat));



	for (int i = 1024*0; i<1024*1; i++)
		printf("Y[%d] = %f --- and --- T[%d] = %f    delta = %f\n", i, Y[i], i, T[i], T[i] - Y[i]);


	for (int i = 1024*0; i<P; i++) 
	{
		if (Y[i] - T[i] > 1e-2)
			printf("Y[%d] = %f --- but --- T[%d] = %f    delta = %f\n", i, Y[i], i, T[i], T[i] - Y[i]);
	}

	//-----------------------free----------------------------



	// Cleanup and shutdown of cuda
	CUDA_CHECK(cuMemFree(devH));
	CUDA_CHECK(cuMemFree(devX));
	CUDA_CHECK(cuMemFree(devY));

	free(H);
	free(X);
	free(Y);

	CUDA_CHECK(cuEventDestroy(hStart));
	CUDA_CHECK(cuEventDestroy(hStop));

	//CUBLAS_CHECK( cublasDestroy(hCublas) );
	//hCublas  = 0;
	CUDA_CHECK(cuCtxDestroy(hContext));
	hContext = 0;


	cudaDeviceReset();

	printf("done\n");


	return EXIT_SUCCESS;


}






