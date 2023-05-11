// ***********************************************************************
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
// ***********************************************************************
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "uni_mem_allocator.h"
#include "cuda_img.h"
// Prototype of function in .cu file
void cu_rotate(CudaImg input, CudaImg output);
void cu_insertimage(CudaImg puvodni, CudaImg vysledny, int2 pozice);
void cu_resize_zmenseni(CudaImg bigpic, CudaImg smallpic, int zmenseni);

void cu_run_RGB(CudaImg puvodni, CudaImg oriznuty);

void cu_rotate1(CudaImg input, CudaImg output, float deg);


int main()
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    cv::Mat nacteny = cv::imread("/home/fei/kuc0396/Downloads/image.jpg", cv::IMREAD_UNCHANGED);
    CudaImg pomoc; 
    pomoc.m_size.x = nacteny.cols;
    pomoc.m_size.y = nacteny.rows;
    pomoc.m_p_uchar4 = (uchar4*)nacteny.data;
    cv::imwrite("povodny.jpg", nacteny);


    cv::Mat RGBpolovica(pomoc.m_size.y, pomoc.m_size.x, CV_8UC3);
    CudaImg pulka2;
    pulka2.m_size.x = RGBpolovica.cols;
    pulka2.m_size.y = RGBpolovica.rows;
    pulka2.m_p_uchar3 = (uchar3*)RGBpolovica.data;
    cu_run_RGB(pomoc, pulka2);
    cv::imwrite("polovicaRGB.jpg", RGBpolovica);


    cv::Mat nacteny_obrazek = cv::imread("/home/fei/kuc0396/Downloads/apps-cuda-demo-main/cuda5_unm/ball.png", cv::IMREAD_UNCHANGED); //cv:IMREAD_COLOR
    CudaImg pomocny_obrazek; //do pomocneho obrazku nactu stejne rozmery jako puvodniho obrazku
    pomocny_obrazek.m_size.x = nacteny_obrazek.cols;
    pomocny_obrazek.m_size.y = nacteny_obrazek.rows;
    pomocny_obrazek.m_p_uchar4 = (uchar4*)nacteny_obrazek.data;


    cv::Mat nacteny_obrazek2 = cv::imread("/home/fei/kuc0396/Downloads/apps-cuda-demo-main/cuda5_unm/ball.png", cv::IMREAD_UNCHANGED); //cv:IMREAD_COLOR
    CudaImg pomocny_obrazek2; //do pomocneho obrazku nactu stejne rozmery jako puvodniho obrazku
    pomocny_obrazek2.m_size.x = nacteny_obrazek2.cols;
    pomocny_obrazek2.m_size.y = nacteny_obrazek2.rows;
    pomocny_obrazek2.m_p_uchar4 = (uchar4*)nacteny_obrazek2.data;
    //std::cout << nacteny_obrazek.channels() << std::endl;
    //std::cout << nacteny_obrazek2.channels() << std::endl;

    cv::Mat trojnasobne_mensi_obrazek(pomocny_obrazek.m_size.y/3 , pomocny_obrazek.m_size.x/3 , CV_8UC4);
    CudaImg trojnasobne_mensi_obrazek2;
    trojnasobne_mensi_obrazek2.m_size.x = trojnasobne_mensi_obrazek.cols;
    trojnasobne_mensi_obrazek2.m_size.y = trojnasobne_mensi_obrazek.rows;
    trojnasobne_mensi_obrazek2.m_p_uchar4 = (uchar4*)trojnasobne_mensi_obrazek.data;
   cu_resize_zmenseni(pomocny_obrazek, trojnasobne_mensi_obrazek2, 3);
   cv::imwrite("3x zmenseny.jpg", trojnasobne_mensi_obrazek);
   //cv::imshow("pred resize 3xmensi", nacteny_obrazek);
   //cv::imshow("resized 3xmensi", trojnasobne_mensi_obrazek);


   cv::Mat devitinasobne_mensi_obrazek(trojnasobne_mensi_obrazek2.m_size.y / 3, trojnasobne_mensi_obrazek2.m_size.x / 3, CV_8UC4);
   CudaImg devitinasobne_mensi_obrazek2;
   devitinasobne_mensi_obrazek2.m_size.x = devitinasobne_mensi_obrazek.cols;
   devitinasobne_mensi_obrazek2.m_size.y = devitinasobne_mensi_obrazek.rows;
   devitinasobne_mensi_obrazek2.m_p_uchar4 = (uchar4*)devitinasobne_mensi_obrazek.data;
   cu_resize_zmenseni(trojnasobne_mensi_obrazek2, devitinasobne_mensi_obrazek2, 3);
   cv::imwrite("9x zmenseny.jpg", devitinasobne_mensi_obrazek);
   //cv::imshow("pred resize 9xmensi", trojnasobne_mensi_obrazek);
   //cv::imshow("resized 9xmensi", devitinasobne_mensi_obrazek);


   cv::Mat obrazek_rotate(pomocny_obrazek.m_size.y, pomocny_obrazek.m_size.x, CV_8UC4);
   CudaImg obrazek_rotate2;
   obrazek_rotate2.m_size.x = obrazek_rotate.cols;
   obrazek_rotate2.m_size.y = obrazek_rotate.rows;
   obrazek_rotate2.m_p_uchar4 = (uchar4*)obrazek_rotate.data;
    
   //cv::imshow("pred rotate", devitinasobne_mensi_obrazek_rotate);
   cu_rotate1(pomocny_obrazek, obrazek_rotate2, 245);
   cv::imwrite("rotovany.jpg", obrazek_rotate);

   
   cv::Mat devitinasobne_mensi_obrazek_rotate(pomocny_obrazek.m_size.y / 9, pomocny_obrazek.m_size.x / 9, CV_8UC4);
   CudaImg devitinasobne_mensi_obrazek_rotate2;
   devitinasobne_mensi_obrazek_rotate2.m_size.x = devitinasobne_mensi_obrazek_rotate.cols;
   devitinasobne_mensi_obrazek_rotate2.m_size.y = devitinasobne_mensi_obrazek_rotate.rows;
   devitinasobne_mensi_obrazek_rotate2.m_p_uchar4 = (uchar4*)devitinasobne_mensi_obrazek_rotate.data;
    
   //cv::imshow("pred rotate", devitinasobne_mensi_obrazek_rotate);
   cu_rotate(devitinasobne_mensi_obrazek2, devitinasobne_mensi_obrazek_rotate2);
   cv::imwrite("9x zmenseny a zrotovany.jpg", devitinasobne_mensi_obrazek_rotate);
  // cv::imshow("po rotate", devitinasobne_mensi_obrazek_rotate);
 
    
    int2 pozice;
    pozice.x = pomocny_obrazek.m_size.x / 2 - devitinasobne_mensi_obrazek_rotate2.m_size.x/2;
    pozice.y = pomocny_obrazek.m_size.y / 2 - devitinasobne_mensi_obrazek_rotate2.m_size.y / 2;
    cu_insertimage(pomocny_obrazek2, devitinasobne_mensi_obrazek_rotate2, pozice);
    cv::imwrite("insert 9x zmenseny a rotovany.jpg", nacteny_obrazek2);
    //cv::imshow("puvodni", nacteny_obrazek);
    //cv::imshow("po insertu", nacteny_obrazek2);
    
    
    //cv::waitKey(0);
}