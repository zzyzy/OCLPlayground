#include <iostream>
#include <fstream>
#include <unordered_map>

#include <CL/cl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "OCLUtils.h"

#define CL_FILENAME "Reduction.cl"

#define INPUT_IMAGE_FILENAME "bunnycity1.bmp"

#define LUMINANCE_KERNEL "Luminance"
#define REDUCTION_STEP_KERNEL "ReductionStep"
#define REDUCTION_COMPLETE_KERNEL "ReductionComplete"

#define VENDOR_INTEL "Intel"
#define VENDOR_AMD "Advanced Micro Devices"
#define VENDOR_NVIDIA "NVIDIA"
#define SELECTED_VENDOR VENDOR_INTEL

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
	// Create buffers for image data
	//
	// ==============================================================
	int w, h, n;
	unsigned char* inputImage = stbi_load(INPUT_IMAGE_FILENAME, &w, &h, &n, 4);
	cl::ImageFormat imageFormat(CL_RGBA, CL_UNORM_INT8);
	cl::Sampler sampler = MakeSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST);
	cl::Image2D inputImageBuffer = MakeImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, w, h, 0, inputImage);

	// ==============================================================
	//
	// Create buffers for luminance data
	//
	// ==============================================================
	float sum = 0.0f;
	size_t localSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	size_t globalSize = (w * h) / 4;
	cl::Buffer luminanceBuffer = MakeBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * w * h);
	cl::Buffer sumBuffer = MakeBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float));

	// ==============================================================
	//
	// Get luminance values
	//
	// ==============================================================
	err = kernels[LUMINANCE_KERNEL].setArg(0, inputImageBuffer);
	err |= kernels[LUMINANCE_KERNEL].setArg(1, sampler);
	err |= kernels[LUMINANCE_KERNEL].setArg(2, luminanceBuffer);
	CheckErrorCode(err, "Unablet to set luminance kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[LUMINANCE_KERNEL], cl::NullRange, cl::NDRange(w, h));
	CheckErrorCode(err, "Unablet to enqueue luminance kernel");

	stbi_image_free(inputImage);

	// ==============================================================
	//
	// Start reduction
	//
	// ==============================================================
	err = kernels[REDUCTION_STEP_KERNEL].setArg(0, luminanceBuffer);
	err |= kernels[REDUCTION_STEP_KERNEL].setArg(1, sizeof(float) * 4 * localSize, nullptr);
	CheckErrorCode(err, "Unablet to set reduction step kernel arguments");

	err = kernels[REDUCTION_COMPLETE_KERNEL].setArg(0, luminanceBuffer);
	err |= kernels[REDUCTION_COMPLETE_KERNEL].setArg(1, sizeof(float) * 4 * localSize, nullptr);
	err |= kernels[REDUCTION_COMPLETE_KERNEL].setArg(2, sumBuffer);
	CheckErrorCode(err, "Unablet to set reduction complete kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[REDUCTION_STEP_KERNEL], cl::NullRange,
	                                 cl::NDRange(globalSize), cl::NDRange(localSize));
	CheckErrorCode(err, "Unablet to enqueue reduction step kernel");

	while (globalSize / localSize > localSize)
	{
		globalSize = globalSize / localSize;
		err = queue.enqueueNDRangeKernel(kernels[REDUCTION_STEP_KERNEL], cl::NullRange,
		                                 cl::NDRange(globalSize), cl::NDRange(localSize));
		CheckErrorCode(err, "Unablet to enqueue reduction step kernel");
	}

	globalSize = globalSize / localSize;
	err = queue.enqueueNDRangeKernel(kernels[REDUCTION_COMPLETE_KERNEL], cl::NullRange, cl::NDRange(globalSize));
	CheckErrorCode(err, "Unablet to enqueue reduction complete kernel");

	err = queue.enqueueReadBuffer(sumBuffer, CL_TRUE, 0, sizeof(float), &sum);
	CheckErrorCode(err, "Unablet to read sum buffer");

	std::cout << "Sum: " << sum << std::endl;
	std::cout << "Avg: " << sum / (w * h) << std::endl;

	return 0;
}
