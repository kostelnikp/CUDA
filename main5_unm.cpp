// *********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Image creation and its modification using CUDA.
// Image manipulation is performed by OpenCV library. 
//
// *********************************************************************
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "uni_mem_allocator.h"
#include "cuda_img.h"

void cu_run_RGB(CudaImg puvodni, CudaImg oriznuty);
void cu_create_chessboard( CudaImg t_color_cuda_img, int t_square_size , int n);
int main()
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);
    cv::Mat nacteny_obrazek = cv::imread("/home/fei/kuc0396/Downloads/biela.jpg", cv::IMREAD_UNCHANGED);
    CudaImg pomocny_obrazek; 
    pomocny_obrazek.m_size.x = nacteny_obrazek.cols;
    pomocny_obrazek.m_size.y = nacteny_obrazek.rows;
    pomocny_obrazek.m_p_uchar4 = (uchar4*)nacteny_obrazek.data;
    cv::imwrite("povodny.jpg", nacteny_obrazek);

    cv::Mat pulka(pomocny_obrazek.m_size.y, pomocny_obrazek.m_size.x, CV_8UC3);
    CudaImg pulka2;
    pulka2.m_size.x = pulka.cols;
    pulka2.m_size.y = pulka.rows;
    pulka2.m_p_uchar3 = (uchar3*)pulka.data;
    cu_run_RGB(pomocny_obrazek, pulka2);
    cv::imwrite("polkaRGB.jpg", pulka);


    cv::Mat l_chessboard_cv_img( 511, 515, CV_8UC3 );
    CudaImg l_chessboard_cuda_img;
    l_chessboard_cuda_img.m_size.x = l_chessboard_cv_img.cols;
    l_chessboard_cuda_img.m_size.y = l_chessboard_cv_img.rows;
    l_chessboard_cuda_img.m_p_uchar3 = ( uchar3 * ) l_chessboard_cv_img.data;

    cu_create_chessboard( l_chessboard_cuda_img, 21 , 1);
    cv::imwrite( "GBR.jpg", l_chessboard_cv_img );

    cu_create_chessboard( l_chessboard_cuda_img, 21, 2);
    cv::imwrite( "BRG.jpg", l_chessboard_cv_img );

    cu_create_chessboard( l_chessboard_cuda_img, 21, 3);
    cv::imwrite( "RGB.jpg", l_chessboard_cv_img );
}
