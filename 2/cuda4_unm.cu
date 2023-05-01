// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_color_cuda_img.m_size.y)
        return;
    if (l_x >= t_color_cuda_img.m_size.x)
        return;

    // Get point from color picture
    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x];

    // Store BW point to new image
    t_bw_cuda_img.m_p_uchar1[l_y * t_bw_cuda_img.m_size.x + l_x].x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
}

void cu_run_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_grayscale<<<l_blocks, l_threads>>>(t_color_cuda_img, t_bw_cuda_img);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_maska(CudaImg t_color_cuda_img, CudaImg t_maska_cuda_img, uchar3 maska)
{
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (index_x >= t_color_cuda_img.m_size.x)
        return;
    if (index_y >= t_color_cuda_img.m_size.y)
        return;

    *(t_maska_cuda_img.at3(index_y, index_x)) = *(t_color_cuda_img.at3(index_y, index_x));
    t_maska_cuda_img.at3(index_y, index_x)->x *= (float)maska.x / 255.0;
    t_maska_cuda_img.at3(index_y, index_x)->y *= (float)maska.y / 255.0;
    t_maska_cuda_img.at3(index_y, index_x)->z *= (float)maska.z / 255.0;
}

void cu_run_maska(CudaImg t_color_cuda_img, CudaImg t_maska_cuda_img, uchar3 maska)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_maska<<<l_blocks, l_threads>>>(t_color_cuda_img, t_maska_cuda_img, maska);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_split(CudaImg t_color_cuda_img, CudaImg t_img1_cuda, CudaImg t_img2_cuda, bool flipX)
{
    int l_y = threadIdx.y + blockIdx.y * blockDim.y;
    int l_x = threadIdx.x + blockIdx.x * blockDim.x;
    if (l_x >= t_color_cuda_img.m_size.x)
        return;
    if (l_y >= t_color_cuda_img.m_size.y)
        return;

    int half_point = t_color_cuda_img.m_size.x / 2.0;

    if (flipX)
    {
        if (l_x >= half_point)
        {
            *(t_img2_cuda.at3((t_img2_cuda.m_size.x - 1) - (l_x - half_point), l_y)) = *(t_color_cuda_img.at3(l_x, l_y));
        }
        else
        {
            *(t_img1_cuda.at3((t_img1_cuda.m_size.x - 1) - l_x, l_y)) = *(t_color_cuda_img.at3(l_x, l_y));
        }
    }
    else
    {
        if (l_x >= half_point)
        {
            *(t_img2_cuda.at3((l_x - half_point), (t_img2_cuda.m_size.y - 1) - l_y)) = *(t_color_cuda_img.at3(l_x, l_y));
        }
        else
        {
            *(t_img1_cuda.at3(l_x, (t_img1_cuda.m_size.y - 1) - l_y)) = *(t_color_cuda_img.at3(l_x, l_y));
        }
    }
}

void cu_run_split(CudaImg t_color_cuda_img, CudaImg t_img1_cuda, CudaImg t_img2_cuda, bool flipX)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_split<<<l_blocks, l_threads>>>(t_color_cuda_img, t_img1_cuda, t_img2_cuda, flipX);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}