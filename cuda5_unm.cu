
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_img.h"

__global__ void kernel_RGB(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
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
    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x];
    //uchar3 l_bgr = t_color_cuda_img.atuchar3(l_y, l_x);
    l_bgr.x = l_bgr.x /2;
    l_bgr.y = l_bgr.y /2;
    l_bgr.z = l_bgr.z /2;
    t_bw_cuda_img.m_p_uchar3[l_y * t_bw_cuda_img.m_size.x + l_x] = l_bgr;
    //t_bw_cuda_img.atuchar3(l_y, l_x) = l_bgr;
   
}
void cu_run_RGB(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    cudaError_t l_cerr;
    int l_block_size = 16;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_RGB << < l_blocks, l_threads >> > (t_color_cuda_img, t_bw_cuda_img);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    cudaDeviceSynchronize();
}

__global__ void kernel_chessboard(CudaImg t_color_cuda_img, int n)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_color_cuda_img.m_size.y)
    {
        return;
    }
    if (l_x >= t_color_cuda_img.m_size.x)
    {
        return;
    }

    int red = 0;
    int green = 0;
    int blue = 0;

    // Set color based on block position
    int block_index = blockIdx.y * gridDim.x + blockIdx.x;
    int color_index = block_index % 3;
    if (n == 1) //ZLY GBR
    {
    if (color_index == 0)
    {
         green = 255;
    }
    else if (color_index == 1)
    {
         red = 255;
    }
    else
    {
        blue = 255;
    }
    }


    else if (n == 2) //BRG
    {
    if (color_index == 0)
    {
        red = 255;
    }
    else if (color_index == 1)
    {
        
        blue = 255;
    }
    else
    {
        green = 255;
    }
    }

    else if (n == 3) //RGB
    {
        if (color_index == 0)
    {
        blue = 255;
    }
    else if (color_index == 1)
    {
        green = 255;
    }
    else
    {
        red = 255;
        
    }
    }
    
    t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x].x = red;
    t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x].y = green;
    t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x].z = blue;
}


void cu_create_chessboard(CudaImg t_color_cuda_img, int t_square_size, int n)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    dim3 l_blocks((t_color_cuda_img.m_size.x + t_square_size - 1) / t_square_size,
                  (t_color_cuda_img.m_size.y + t_square_size - 1) / t_square_size);
    dim3 l_threads(t_square_size, t_square_size);
    kernel_chessboard<<<l_blocks, l_threads>>>(t_color_cuda_img, n);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
