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
// Image manipulation is performed by OpenCV library.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv
{
}

// Function prototype from .cu file
void cu_run_grayscale(CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img);
void cu_run_maska(CudaImg t_color_cuda_img, CudaImg t_maska_cuda_img, uchar3 maska);
void cu_run_split(CudaImg t_color_cuda_img, CudaImg t_img1_cuda, CudaImg t_img2_cuda, bool flipX);

int main(int t_numarg, char **t_arg)
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (t_numarg < 2)
    {
        printf("Enter picture filename!\n");
        return 1;
    }

    // Load image
    cv::Mat l_bgr_cv_img = cv::imread(t_arg[1], cv::IMREAD_COLOR); // CV_LOAD_IMAGE_COLOR );

    if (!l_bgr_cv_img.data)
    {
        printf("Unable to read file '%s'\n", t_arg[1]);
        return 1;
    }

    // create empty BW image
    cv::Mat l_bw_cv_img(l_bgr_cv_img.size(), CV_8UC1);

    // data for CUDA
    CudaImg l_bgr_cuda_img, l_bw_cuda_img;
    l_bgr_cuda_img.m_size.x = l_bw_cuda_img.m_size.x = l_bgr_cv_img.size().width;
    l_bgr_cuda_img.m_size.y = l_bw_cuda_img.m_size.y = l_bgr_cv_img.size().height;
    l_bgr_cuda_img.m_p_uchar3 = (uchar3 *)l_bgr_cv_img.data;
    l_bw_cuda_img.m_p_uchar1 = (uchar1 *)l_bw_cv_img.data;

    // Function calling from .cu file
    cu_run_grayscale(l_bgr_cuda_img, l_bw_cuda_img);

    cv::Mat l_mask_cv_img(l_bgr_cv_img.size(), CV_8UC3);

    CudaImg l_mask_cuda_img;
    l_mask_cuda_img.m_size.x = l_bgr_cv_img.size().width;
    l_mask_cuda_img.m_size.y = l_bgr_cv_img.size().height;
    l_mask_cuda_img.m_p_uchar3 = (uchar3 *)l_mask_cv_img.data;

    cu_run_maska(l_bgr_cuda_img, l_mask_cuda_img, {255,0,154});

    cv::Size half_size = l_bgr_cv_img.size();
    half_size.width /= 2.0;

    cv::Mat l_img1_cv_img(half_size, CV_8UC3);
    cv::Mat l_img2_cv_img(half_size, CV_8UC3);

    CudaImg l_img1_cuda_img, l_img2_cuda_img;
    l_img1_cuda_img.m_size.x = l_img2_cuda_img.m_size.x = l_img1_cv_img.size().width;
    l_img1_cuda_img.m_size.y = l_img2_cuda_img.m_size.y = l_img1_cv_img.size().height;
    l_img1_cuda_img.m_p_uchar3 = (uchar3 *)l_img1_cv_img.data;
    l_img2_cuda_img.m_p_uchar3 = (uchar3 *)l_img2_cv_img.data;

    cu_run_split(l_bgr_cuda_img, l_img1_cuda_img, l_img2_cuda_img, true);

    // Show the Color and BW image
    cv::imshow("Color", l_bgr_cv_img);
    cv::imshow("GrayScale", l_bw_cv_img);
    cv::imshow("Part 1", l_img1_cv_img);
    cv::imshow("Part 2", l_img2_cv_img);
    cv::imshow("Mask", l_mask_cv_img);
    cv::waitKey(0);
}
