 
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
global void change_theme(CudaPicture picture)
{
    int l_x = threadIdx.x + blockIdx.x * blockDim.x;
    int l_y = threadIdx.y + blockIdx.y * blockDim.y;
    picture.cdata[l_y * picture.size.x + l_x].x /= 2;
    picture.cdata[l_y * picture.size.x + l_x].y /= 2;
    picture.cdata[l_y * picture.size.x + l_x].z /= 2;
}
void zmenaBarvu(CudaPicture picture, dim3 t_grid_size, dim3 t_block_size )
{
   // cudaError_t l_cerr;
    CudaPicture cuda_picture;
    cuda_picture.size = picture.size;
    cudaMalloc(&cuda_picture.vdata, cuda_picture.size.x * cuda_picture.size.y * sizeof(uchar3));
    cudaMemcpy( cuda_picture.vdata, picture.vdata, cuda_picture.size.x * cuda_picture.size.y * sizeof( uchar3 ), cudaMemcpyHostToDevice );
    change_theme<<< t_grid_size, t_block_size>>>(cuda_picture);
    cudaMemcpy( picture.vdata, cuda_picture.vdata, cuda_picture.size.x * cuda_picture.size.y * sizeof( uchar3 ), cudaMemcpyDeviceToHost );
    cudaFree(cuda_picture.vdata);
    cudaDeviceSynchronize();
}
global void change_rgb_theme(CudaPicture picture)
{
    int l_x = threadIdx.x + blockIdx.x * blockDim.x;
    int l_y = threadIdx.y + blockIdx.y * blockDim.y;
        picture.cdata[l_y * picture.size.x + l_x].x = 255;
        picture.cdata[l_y * picture.size.x + l_x].y = 0;
        picture.cdata[l_y * picture.size.x + l_x].z = 0;
        if((blockIdx.x % 2) == 0){
            picture.cdata[l_y * picture.size.x + l_x].x = 0;
            picture.cdata[l_y * picture.size.x + l_x].y = 255;
            picture.cdata[l_y * picture.size.x + l_x].z = 0;
        }
        if(blockIdx.x % 3 == 0){
            picture.cdata[l_y * picture.size.x + l_x].x = 0;
            picture.cdata[l_y * picture.size.x + l_x].y = 0;
            picture.cdata[l_y * picture.size.x + l_x].z = 255;
        }
}
void zmena_rgb(CudaPicture picture, dim3 t_grid_size, dim3 t_block_size )
{
    //cudaError_t l_cerr;
    CudaPicture cuda_picture;
    cuda_picture.size = picture.size;
    cudaMalloc(&cuda_picture.vdata, cuda_picture.size.x * cuda_picture.size.y * sizeof(uchar3));
    cudaMemcpy( cuda_picture.vdata, picture.vdata, cuda_picture.size.x * cuda_picture.size.y * sizeof( uchar3 ), cudaMemcpyHostToDevice );
    change_rgb_theme<<< t_grid_size, t_block_size>>>(cuda_picture);
    cudaMemcpy( picture.vdata, cuda_picture.vdata, cuda_picture.size.x * cuda_picture.size.y * sizeof( uchar3 ), cudaMemcpyDeviceToHost );
    cudaFree(cuda_picture.vdata);
    cudaDeviceSynchronize();
}