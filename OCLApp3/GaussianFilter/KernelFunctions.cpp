#include "KernelFunctions.h"
#include <iostream>

unsigned char* simpleConvolution(cl::Kernel kernel,
	cl::CommandQueue queue,
	cl::Image2D inputImage,
	cl::Image2D outputImage,
	cl::Buffer filter,
	int fx,	// filter width
	int fy)	// filter height
{
	auto iw = inputImage.getImageInfo<CL_IMAGE_WIDTH>();
	auto ih = inputImage.getImageInfo<CL_IMAGE_HEIGHT>();
	unsigned char* output = new unsigned char[iw * ih * 4];
	cl_int err;

	kernel.setArg(0, inputImage);
	kernel.setArg(1, outputImage);
	kernel.setArg(2, filter);
	kernel.setArg(3, fx);
	kernel.setArg(4, fy);

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(iw, ih));
	if (err != CL_SUCCESS)
	{
		std::cerr << "Unable to enqueue kernel: " << kernel.getInfo<CL_KERNEL_FUNCTION_NAME>() << std::endl;
		throw std::runtime_error("Unable to enqueue kernel: " + kernel.getInfo<CL_KERNEL_FUNCTION_NAME>());
	}

	cl::size_t<3> region; region[0] = iw; region[1] = ih; region[2] = 1;
	err = queue.enqueueReadImage(outputImage, CL_TRUE, cl::size_t<3>(), region, 0, 0, output);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Unable to read image buffer: " << kernel.getInfo<CL_KERNEL_FUNCTION_NAME>() << std::endl;
		throw std::runtime_error("Unable to read image buffer: " + kernel.getInfo<CL_KERNEL_FUNCTION_NAME>());
	}

	return output;
}
