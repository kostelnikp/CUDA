
// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Manipulation with prepared image.
//
// ***********************************************************************
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_img.h"

__global__ void RGB(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_bw_cuda_img.m_size.y) 
    {
        return;
    }
    if (l_x >= t_bw_cuda_img.m_size.x)
    { 
        return;
    }
    uchar3 l_bgr;
    l_bgr = t_color_cuda_img.atuchar3(l_y, l_x);
	uchar3 l_bgr_temp;

	//BGR
	//RGB
	l_bgr_temp.x = l_bgr.x;
	l_bgr_temp.y = l_bgr.y;
	l_bgr_temp.z = l_bgr.z;
    l_bgr.x = l_bgr_temp.y;
    l_bgr.y = l_bgr_temp.z;
    l_bgr.z = l_bgr_temp.x;
    
    t_bw_cuda_img.atuchar3(l_y, l_x) = l_bgr;
   
}
void cu_run_RGB(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    cudaError_t l_cerr;
    int l_block_size = 16;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    RGB << < l_blocks, l_threads >> > (t_color_cuda_img, t_bw_cuda_img);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    cudaDeviceSynchronize();
}

__global__ void kernel_rotate(CudaImg input, CudaImg output)
{
	// X,Y coordinates 
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if (l_y >= input.m_size.y) 
    {
        return;
    }
	if (l_x >= input.m_size.x)  
    {
        return;
    }
	int pom_x = input.m_size.x - l_x;
	int pom_y = input.m_size.y - l_y;
	output.atuchar4(pom_y, pom_x) =input.atuchar4(l_y, l_x);
}
void cu_rotate(CudaImg input, CudaImg output)
{
	cudaError_t l_cerr;
	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks((input.m_size.x + l_block_size - 1) / l_block_size,
		(input.m_size.y + l_block_size - 1) / l_block_size);
	dim3 l_threads(l_block_size, l_block_size);
	kernel_rotate << < l_blocks, l_threads >> > (input, output);
	if ((l_cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
	cudaDeviceSynchronize();
}
__global__ void kernel_rotate1(CudaImg input, CudaImg output, float angle_degrees)
{
    // X,Y coordinates 
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= input.m_size.y || l_x >= input.m_size.x)
    {
        return;
    }

    float angle_radians = angle_degrees * M_PI / 180.0;

    int in_x = l_x - input.m_size.x / 2;
    int in_y = l_y - input.m_size.y / 2;
    int out_x = in_x * cos(angle_radians) - in_y * sin(angle_radians);
    int out_y = in_x * sin(angle_radians) + in_y * cos(angle_radians);
    out_x += output.m_size.x / 2;
    out_y += output.m_size.y / 2;

    if (out_x < 0 || out_x >= output.m_size.x || out_y < 0 || out_y >= output.m_size.y)
    {
        return;
    }

	output.atuchar4(out_y, out_x) = input.atuchar4(l_y, l_x);
}
void cu_rotate1(CudaImg input, CudaImg output, float angle_degrees)
{
    cudaError_t l_cerr;
    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 32;
    dim3 l_blocks((input.m_size.x + l_block_size - 1) / l_block_size,
        (input.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_rotate1 << < l_blocks, l_threads >> > (input, output, angle_degrees);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    cudaDeviceSynchronize();
}

__global__ void kernel_insertimage(CudaImg puvodni, CudaImg vysledny, int2 pozice)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if (l_y >= vysledny.m_size.y) return;
	if (l_x >= vysledny.m_size.x) return;
	int pom_y = l_y + pozice.y;
	int pom_x = l_x + pozice.x;
	if (pom_y>= puvodni.m_size.y || pom_y<0) return;
	if (pom_x>= puvodni.m_size.x || pom_x<0) return;
	uchar4 vyslednyObr = vysledny.atuchar4(l_y,l_x);
	puvodni.atuchar4(pom_y, pom_x) = vyslednyObr;
}
void cu_insertimage(CudaImg puvodni, CudaImg vysledny, int2 pozice)
{
	cudaError_t l_cerr;
	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks((vysledny.m_size.x + l_block_size - 1) / l_block_size,
		(vysledny.m_size.y + l_block_size - 1) / l_block_size);
	dim3 l_threads(l_block_size, l_block_size);
	kernel_insertimage << < l_blocks, l_threads >> > (puvodni, vysledny, pozice);
	if ((l_cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
	cudaDeviceSynchronize();
}
__global__ void kernel_resize_zmenseni(CudaImg bigpic, CudaImg smallpic, int zmenseni)
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if (l_y >= smallpic.m_size.y) return;
	if (l_x >= smallpic.m_size.x)  return;
	uchar4 vysledek = { 0,0,0,0 };
	if (l_y > 0 && l_x > 0 ) {
		vysledek.x = (bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni - 1).x + bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni).x +
			bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni + 1).x + bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni - 1).x +
			bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni).x + bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni + 1).x +
			bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni - 1).x + bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni).x +
			bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni + 1).x) / 9;
		vysledek.y = (bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni - 1).y + bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni).y +
			bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni + 1).y + bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni - 1).y +
			bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni).y + bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni + 1).y +
			bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni - 1).y + bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni).y +
			bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni + 1).y) / 9;
		vysledek.z = (bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni - 1).z + bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni).z +
			bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni + 1).z + bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni - 1).z +
			bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni).z + bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni + 1).z +
			bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni - 1).z + bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni).z +
			bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni + 1).z) / 9;
		vysledek.w = (bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni - 1).w + bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni).w +
			bigpic.atuchar4(l_y * zmenseni + 1, l_x * zmenseni + 1).w + bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni - 1).w +
			bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni).w + bigpic.atuchar4(l_y * zmenseni, l_x * zmenseni + 1).w +
			bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni - 1).w + bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni).w +
			bigpic.atuchar4(l_y * zmenseni - 1, l_x * zmenseni + 1).w) / 9;
	}
	smallpic.atuchar4(l_y, l_x) = vysledek;
}
void cu_resize_zmenseni(CudaImg bigpic, CudaImg smallpic, int zmenseni)
{
	cudaError_t l_cerr;
	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks((smallpic.m_size.x + l_block_size - 1) / l_block_size,
		(smallpic.m_size.y + l_block_size - 1) / l_block_size);
	dim3 l_threads(l_block_size, l_block_size);
	kernel_resize_zmenseni << < l_blocks, l_threads >> > (bigpic, smallpic, zmenseni);
	if ((l_cerr = cudaGetLastError()) != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
	cudaDeviceSynchronize();
}
