#pragma once
#ifndef __KERNELFUNCTIONS_H__
#define __KERNELFUNCTIONS_H__

#include <CL/cl.hpp>

unsigned char* simpleConvolution(cl::Kernel kernel,
	cl::CommandQueue queue,
	cl::Image2D inputImage,
	cl::Image2D outputImage,
	cl::Buffer filter,
	int fx,	// filter width
	int fy);	// filter height

#endif // __KERNELFUNCTIONS_H__
