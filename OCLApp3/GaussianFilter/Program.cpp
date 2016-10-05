#include <iostream>
#include <fstream>
#include <unordered_map>

#include <CL/cl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "OCLUtils.h"
#include "Filters.h"

#define CL_FILENAME "Convolution.cl"

#define INPUT_IMAGE_FILENAME "bunnycity1.bmp"

#define SIMPLE_CONVOLUTION_KERNEL "simpleConvolution"
#define ONE_PASS_CONVOLUTION_KERNEL "onePassConvolution"

#define VENDOR_INTEL "Intel"
#define VENDOR_AMD "Advanced Micro Devices"
#define VENDOR_NVIDIA "NVIDIA"
#define SELECTED_VENDOR VENDOR_NVIDIA

int main()
{
	cl_int err;

	// ==============================================================
	//
	// Setup OCL Context
	//
	// ==============================================================
	cl::Device device = GetDevice(SELECTED_VENDOR);
	cl::Context context = MakeContext(device);
	cl::CommandQueue queue = MakeCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

	std::vector<const char*> sourceFileNames;
	sourceFileNames.push_back(CL_FILENAME);
	cl::Program program = MakeAndBuildProgram(sourceFileNames, context, device);

	std::unordered_map<std::string, cl::Kernel> kernels = MakeKernels(program);

	// ==============================================================
	//
	// Handle user input
	//
	// ==============================================================
	int filterSize = 7;
	std::cout << "Gaussian filter window size? (3/5/7)" << std::endl;
	std::cin >> filterSize;
	while (filterSize != 3 && filterSize != 5 && filterSize != 7)
	{
		std::cout << "Invalid input. Try again." << std::endl;
		std::cout << "Gaussian filter window size? (3/5/7)" << std::endl;
		std::cin >> filterSize;
	}

	// ==============================================================
	//
	// Create buffers for image data
	//
	// ==============================================================
	int w, h, n;
	unsigned char* inputImage = stbi_load(INPUT_IMAGE_FILENAME, &w, &h, &n, 4);
	unsigned char* outputImage = new unsigned char[w * h * 4];
	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = w;
	region[1] = h;
	region[2] = 1;
	cl::ImageFormat imageFormat(CL_RGBA, CL_UNORM_INT8);
	cl::Sampler sampler = MakeSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST);

	cl::Image2D imageBufferA = MakeImage2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
	                                       imageFormat, w, h, 0, inputImage);
	cl::Image2D imageBufferB = MakeImage2D(context, CL_MEM_READ_WRITE,
	                                       imageFormat, w, h, 0);

	// ==============================================================
	//
	// Create buffer for filter data
	//
	// ==============================================================
	std::unordered_map<int, const float*> filters;
	filters.insert(std::make_pair(3, GaussianFilter3));
	filters.insert(std::make_pair(5, GaussianFilter5));
	filters.insert(std::make_pair(7, GaussianFilter7));
	filters.insert(std::make_pair(9, GaussianFilter3x3));
	filters.insert(std::make_pair(25, GaussianFilter5x5));
	filters.insert(std::make_pair(49, GaussianFilter7x7));

	// ==============================================================
	//
	// Simple gaussian blur
	//
	// ==============================================================
	float* filter = const_cast<float*>(filters[filterSize * filterSize]);
	cl::Buffer filterBuffer = MakeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                                     sizeof(float) * filterSize * filterSize, filter);

	err = kernels[SIMPLE_CONVOLUTION_KERNEL].setArg(0, imageBufferA);
	err |= kernels[SIMPLE_CONVOLUTION_KERNEL].setArg(1, imageBufferB);
	err |= kernels[SIMPLE_CONVOLUTION_KERNEL].setArg(2, sampler);
	err |= kernels[SIMPLE_CONVOLUTION_KERNEL].setArg(3, filterBuffer);
	err |= kernels[SIMPLE_CONVOLUTION_KERNEL].setArg(4, filterSize);
	CheckErrorCode(err, "Unable to set simple convolution kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[SIMPLE_CONVOLUTION_KERNEL], cl::NullRange, cl::NDRange(w, h));
	CheckErrorCode(err, "Unable to enqueue simple convolution kernel");

	err = queue.enqueueReadImage(imageBufferB, CL_TRUE, origin, region, 0, 0, outputImage);
	CheckErrorCode(err, "Unable to read output image buffer");

	stbi_write_bmp("Output/simpleBlurImage.bmp", w, h, 4, outputImage);

	// ==============================================================
	//
	// Two pass gaussian blur
	//
	// ==============================================================
	filter = const_cast<float*>(filters[filterSize]);
	filterBuffer = MakeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                          sizeof(float) * filterSize, filter);

	err = kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(0, imageBufferA);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(1, imageBufferB);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(2, sampler);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(3, filterBuffer);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(4, filterSize);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(5, 1);
	CheckErrorCode(err, "Unable to set one pass convolution kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[ONE_PASS_CONVOLUTION_KERNEL], cl::NullRange, cl::NDRange(w, h));
	CheckErrorCode(err, "Unable to enqueue one pass convolution kernel");

	err = queue.enqueueReadImage(imageBufferB, CL_TRUE, origin, region, 0, 0, outputImage);
	CheckErrorCode(err, "Unable to read discarded pixels output image");

	stbi_write_bmp("Output/onePassBlurredImage.bmp", w, h, 4, outputImage);

	err = kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(0, imageBufferB);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(1, imageBufferA);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(5, 0);
	CheckErrorCode(err, "Unable to set one pass convolution kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[ONE_PASS_CONVOLUTION_KERNEL], cl::NullRange, cl::NDRange(w, h));
	CheckErrorCode(err, "Unable to enqueue one pass convolution kernel");

	err = queue.enqueueReadImage(imageBufferA, CL_TRUE, origin, region, 0, 0, outputImage);
	CheckErrorCode(err, "Unable to read output image buffer");

	stbi_write_bmp("Output/twoPassBlurredImage.bmp", w, h, 4, outputImage);

	err = queue.enqueueWriteImage(imageBufferA, CL_TRUE, origin, region, 0, 0, inputImage);
	CheckErrorCode(err, "Unable to write image buffer A");

	// ==============================================================
	//
	// Perform profiling
	//
	// ==============================================================
	cl::Event startEvent, finishEvent;
	int filterSizes[3] = {3, 5, 7};
	std::ofstream outfile;

	for (auto i = 0; i < 3; ++i)
	{
		std::string name = "Simple" + std::to_string(filterSizes[i]) + "x" + std::to_string(filterSizes[i]);
		outfile.open("Profiling/" + name + ".txt");
		outfile << name << std::endl;
		cl_ulong totalTime = 0;

		for (auto u = 0; u < 1000; ++u)
		{
			filter = const_cast<float*>(filters[filterSizes[i] * filterSizes[i]]);
			filterBuffer = MakeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			                          sizeof(float) * filterSizes[i] * filterSizes[i], filter);

			err = kernels[SIMPLE_CONVOLUTION_KERNEL].setArg(3, filterBuffer);
			err |= kernels[SIMPLE_CONVOLUTION_KERNEL].setArg(4, filterSizes[i]);
			CheckErrorCode(err, "Unable to set simple convolution kernel arguments");

			err = queue.enqueueNDRangeKernel(kernels[SIMPLE_CONVOLUTION_KERNEL], cl::NullRange, cl::NDRange(w, h),
			                                 cl::NullRange, nullptr, &finishEvent);
			CheckErrorCode(err, "Unable to enqueue simple convolution kernel");

			err = queue.finish();
			CheckErrorCode(err, "Unable to finish queue");
			auto start = finishEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			auto end = finishEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
			totalTime += end - start;
			outfile << (end - start) / 1000000.0f << std::endl;
		}

		std::cout << "Average time for " << name << ": " << totalTime / 1000000.0f / 1000.0f << std::endl;
		outfile.close();
	}

	for (auto i = 0; i < 3; ++i)
	{
		std::string name = "TwoPass" + std::to_string(filterSizes[i]) + "x" + std::to_string(filterSizes[i]);
		outfile.open("Profiling/" + name + ".txt");
		outfile << name << std::endl;
		cl_ulong totalTime = 0;

		for (auto u = 0; u < 1000; ++u)
		{
			filter = const_cast<float*>(filters[filterSizes[i]]);
			filterBuffer = MakeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			                          sizeof(float) * filterSizes[i], filter);

			err = kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(0, imageBufferA);
			err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(1, imageBufferB);
			err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(3, filterBuffer);
			err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(4, filterSizes[i]);
			err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(5, 1);
			CheckErrorCode(err, "Unable to set one pass convolution kernel arguments");

			err = queue.enqueueNDRangeKernel(kernels[ONE_PASS_CONVOLUTION_KERNEL], cl::NullRange, cl::NDRange(w, h),
			                                 cl::NullRange, nullptr, &startEvent);
			CheckErrorCode(err, "Unable to enqueue one pass convolution kernel");

			err = kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(0, imageBufferB);
			err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(1, imageBufferA);
			err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(5, 0);
			CheckErrorCode(err, "Unable to set one pass convolution kernel arguments");

			err = queue.enqueueNDRangeKernel(kernels[ONE_PASS_CONVOLUTION_KERNEL], cl::NullRange, cl::NDRange(w, h),
			                                 cl::NullRange, nullptr, &finishEvent);
			CheckErrorCode(err, "Unable to enqueue one pass convolution kernel");

			err = queue.finish();
			CheckErrorCode(err, "Unable to finish queue");
			auto start = startEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			auto end = finishEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
			totalTime += end - start;
			outfile << (end - start) / 1000000.0f << std::endl;

			err = queue.enqueueWriteImage(imageBufferA, CL_TRUE, origin, region, 0, 0, inputImage);
			CheckErrorCode(err, "Unable to write image buffer A");
		}

		std::cout << "Average time for " << name << ": " << totalTime / 1000000.0f / 1000.0f << std::endl;
		outfile.close();
	}

	delete[] outputImage;
	stbi_image_free(inputImage);

	return 0;
}
