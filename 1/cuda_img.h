// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************
#pragma once
#include <opencv2/core/mat.hpp>
struct CudaPicture {
    uint3 size;
    union {
        void *vdata;
        uchar3 *cdata;
    };
    uchar3 getPoint(int x, int y)
    {
      return this-> cdata[x + size.x * y];
    }
};
