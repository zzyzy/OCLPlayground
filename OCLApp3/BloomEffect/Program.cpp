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

#define REDUCTION_CL_FILENAME "Reduction.cl"
#define CONVOLUTION_CL_FILENAME "Convolution.cl"
#define BLOOM_CL_FILENAME "Bloom.cl"

#define LUMINANCE_KERNEL "luminance"
#define REDUCTION_STEP_KERNEL "reductionStep"
#define REDUCTION_COMPLETE_KERNEL "reductionComplete"
#define ONE_PASS_CONVOLUTION_KERNEL "onePassConvolution"
#define DISCARD_PIXELS_KERNEL "discardPixels"
#define MERGE_IMAGES_KERNEL "mergeImages"

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
	cl::CommandQueue queue = MakeCommandQueue(context, device);

	std::vector<const char*> sourceFileNames;
	sourceFileNames.push_back(REDUCTION_CL_FILENAME);
	sourceFileNames.push_back(CONVOLUTION_CL_FILENAME);
	sourceFileNames.push_back(BLOOM_CL_FILENAME);
	cl::Program program = MakeAndBuildProgram(sourceFileNames, context, device);

	std::unordered_map<std::string, cl::Kernel> kernels = MakeKernels(program);

	// ==============================================================
	//
	// Handle user input
	//
	// ==============================================================
	std::string filename;
	std::ifstream infile;
	char input;
	int filterSize = 7;
	float luminanceAverage = 0.0f;

	std::cout << "Image filename: ";
	std::cin >> filename;
	infile.open(filename);
	while (!(infile.is_open() && infile.good()))
	{
		std::cout << "Invalid image file. Try again." << std::endl;
		std::cout << "Image filename: ";
		std::cin >> filename;
		infile.open(filename);
	}
	infile.close();

	std::cout << "Use custom settings? (y/n)" << std::endl;
	std::cin >> input;
	while (input != 'y' && input != 'n')
	{
		std::cout << "Invalid input. Try again." << std::endl;
		std::cout << "Use custom settings? (y/n)" << std::endl;
		std::cin >> input;
	}

	if (input == 'y')
	{
		std::cout << "Gaussian filter window size? (3/5/7)" << std::endl;
		std::cin >> filterSize;
		while (filterSize != 3 && filterSize != 5 && filterSize != 7)
		{
			std::cout << "Invalid input. Try again." << std::endl;
			std::cout << "Gaussian filter window size? (3/5/7)" << std::endl;
			std::cin >> filterSize;
		}

		std::cout << "Bloom threshold value? (1-255, 0 for default)" << std::endl;
		std::cin >> luminanceAverage;
		while (luminanceAverage < 0 || luminanceAverage > 255)
		{
			std::cout << "Invalid input. Try again." << std::endl;
			std::cout << "Bloom threshold value? (1-255, 0 for default)" << std::endl;
			std::cin >> luminanceAverage;
		}
	}

	// ==============================================================
	//
	// Create buffers for image data
	//
	// ==============================================================
	int w, h, n;
	unsigned char* inputImage = stbi_load(filename.c_str(), &w, &h, &n, 4);
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

	//TODO: For some reason Intel doesn't allow not using host ptr, but NVIDIA does
	//		Maybe something wrong with C++ interface
	cl::Image2D imageBufferB = MakeImage2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
	                                       imageFormat, w, h, 0, inputImage);
	cl::Image2D imageBufferC = MakeImage2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
	                                       imageFormat, w, h, 0, inputImage);

	// ==============================================================
	//
	// Create buffer for filter data
	//
	// ==============================================================
	std::unordered_map<int, const float*> filters;
	filters.insert(std::make_pair(3, GaussianFilter3));
	filters.insert(std::make_pair(5, GaussianFilter5));
	filters.insert(std::make_pair(7, GaussianFilter7));
	float* filter = const_cast<float*>(filters[filterSize]);;
	cl::Buffer filterBuffer = MakeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	                                     sizeof(float) * filterSize, filter);

	// ==============================================================
	//
	// Find average luminance of input image
	//
	// ==============================================================
	if (luminanceAverage == 0.0f)
	{
		float luminanceSum;
		cl::Buffer luminanceBuffer = MakeBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * w * h);
		cl::Buffer sumBuffer = MakeBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float));
		size_t localSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		size_t globalSize = (w * h) / 4;

		err = kernels[LUMINANCE_KERNEL].setArg(0, imageBufferA);
		err |= kernels[LUMINANCE_KERNEL].setArg(1, sampler);
		err |= kernels[LUMINANCE_KERNEL].setArg(2, luminanceBuffer);
		CheckErrorCode(err, "Unable to set luminance kernel arguments");

		err = kernels[REDUCTION_STEP_KERNEL].setArg(0, luminanceBuffer);
		err |= kernels[REDUCTION_STEP_KERNEL].setArg(1, sizeof(float) * 4 * localSize, nullptr);
		CheckErrorCode(err, "Unable to set reduction step kernel arguments");

		err = kernels[REDUCTION_COMPLETE_KERNEL].setArg(0, luminanceBuffer);
		err |= kernels[REDUCTION_COMPLETE_KERNEL].setArg(1, sizeof(float) * 4 * localSize, nullptr);
		err |= kernels[REDUCTION_COMPLETE_KERNEL].setArg(2, sumBuffer);
		CheckErrorCode(err, "Unable to set reduction complete kernel arguments");

		err = queue.enqueueNDRangeKernel(kernels[LUMINANCE_KERNEL], cl::NullRange, cl::NDRange(w, h));
		CheckErrorCode(err, "Unable to enqueue luminance kernel");

		err = queue.enqueueNDRangeKernel(kernels[REDUCTION_STEP_KERNEL], cl::NullRange, cl::NDRange(globalSize),
		                                 cl::NDRange(localSize));
		CheckErrorCode(err, "Unable to enqueue reduction step kernel");

		while (globalSize / localSize > localSize)
		{
			globalSize = globalSize / localSize;
			err = queue.enqueueNDRangeKernel(kernels[REDUCTION_STEP_KERNEL], cl::NullRange, cl::NDRange(globalSize),
			                                 cl::NDRange(localSize));
			CheckErrorCode(err, "Unable to enqueue reduction step kernel");
		}

		globalSize = globalSize / localSize;
		err = queue.enqueueNDRangeKernel(kernels[REDUCTION_COMPLETE_KERNEL], cl::NullRange, cl::NDRange(globalSize));
		CheckErrorCode(err, "Unable to enqueue reduction complete kernel");

		err = queue.enqueueReadBuffer(sumBuffer, CL_TRUE, 0, sizeof(float), &luminanceSum);
		CheckErrorCode(err, "Unable to read sum");

		luminanceAverage = luminanceSum / (w * h);
	}

	std::cout << "Using threshold value: " << luminanceAverage << std::endl;

	// ==============================================================
	//
	// Discard pixels
	//
	// ==============================================================
	err = kernels[DISCARD_PIXELS_KERNEL].setArg(0, imageBufferA);
	err |= kernels[DISCARD_PIXELS_KERNEL].setArg(1, imageBufferB);
	err |= kernels[DISCARD_PIXELS_KERNEL].setArg(2, sampler);
	err |= kernels[DISCARD_PIXELS_KERNEL].setArg(3, luminanceAverage);
	CheckErrorCode(err, "Unable to set discard pixels kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[DISCARD_PIXELS_KERNEL], cl::NullRange, cl::NDRange(w, h));
	CheckErrorCode(err, "Unable to enqueue discard pixels kernel");

	err = queue.enqueueReadImage(imageBufferB, CL_TRUE, origin, region, 0, 0, outputImage);
	CheckErrorCode(err, "Unable to read discarded pixels output image");

	stbi_write_bmp("Output/discardedPixelsImage.bmp", w, h, 4, outputImage);

	// ==============================================================
	//
	// Two pass gaussian blur
	//
	// ==============================================================
	err = kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(0, imageBufferB);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(1, imageBufferA);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(2, sampler);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(3, filterBuffer);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(4, filterSize);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(5, 1);
	CheckErrorCode(err, "Unable to set one pass convolution kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[ONE_PASS_CONVOLUTION_KERNEL], cl::NullRange, cl::NDRange(w, h));
	CheckErrorCode(err, "Unable to enqueue one pass convolution kernel");

	err = queue.enqueueReadImage(imageBufferA, CL_TRUE, origin, region, 0, 0, outputImage);
	CheckErrorCode(err, "Unable to read discarded pixels output image");

	stbi_write_bmp("Output/onePassBlurredImage.bmp", w, h, 4, outputImage);

	err = kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(0, imageBufferA);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(1, imageBufferB);
	err |= kernels[ONE_PASS_CONVOLUTION_KERNEL].setArg(5, 0);
	CheckErrorCode(err, "Unable to set one pass convolution kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[ONE_PASS_CONVOLUTION_KERNEL], cl::NullRange, cl::NDRange(w, h));
	CheckErrorCode(err, "Unable to enqueue one pass convolution kernel");

	err = queue.enqueueReadImage(imageBufferB, CL_TRUE, origin, region, 0, 0, outputImage);
	CheckErrorCode(err, "Unable to read output image buffer");

	stbi_write_bmp("Output/twoPassBlurredImage.bmp", w, h, 4, outputImage);

	// ==============================================================
	//
	// Merge original input image with two pass blurred image
	//
	// ==============================================================
	err = queue.enqueueWriteImage(imageBufferA, CL_TRUE, origin, region, 0, 0, inputImage);
	CheckErrorCode(err, "Unable to write image buffer A");

	err = kernels[MERGE_IMAGES_KERNEL].setArg(0, imageBufferA);
	err |= kernels[MERGE_IMAGES_KERNEL].setArg(1, imageBufferB);
	err |= kernels[MERGE_IMAGES_KERNEL].setArg(2, imageBufferC);
	err |= kernels[MERGE_IMAGES_KERNEL].setArg(3, sampler);
	CheckErrorCode(err, "Unable to set merge images kernel arguments");

	err = queue.enqueueNDRangeKernel(kernels[MERGE_IMAGES_KERNEL], cl::NullRange, cl::NDRange(w, h));
	CheckErrorCode(err, "Unable to enqueue merge images kernel");

	err = queue.enqueueReadImage(imageBufferC, CL_TRUE, origin, region, 0, 0, outputImage);
	CheckErrorCode(err, "Unable to read output image buffer");

	stbi_write_bmp("Output/bloomImage.bmp", w, h, 4, outputImage);

	delete[] outputImage;
	stbi_image_free(inputImage);

	return 0;
}
