#include <iostream>
#include <fstream>
#include <sstream>

#include <CL/cl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const std::string CL_FILENAME = "Reduction.cl";
const std::string INPUT_IMAGE_FILENAME = "Input/bunnycity1.bmp";
const std::string OUTPUT_IMAGE_FILENAME = "Output/bunnycity1.bmp";
const std::string VENDOR_INTEL = "Intel";
const std::string VENDOR_AMD = "Advanced Micro Devices";
const std::string VENDOR_NVIDIA = "NVIDIA";
const std::string SELECTED_VENDOR = VENDOR_NVIDIA;

int main()
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Platform platform;
	cl::Device device;
	cl_int err;

	// Get platforms
	err = cl::Platform::get(&platforms);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to get OpenCL platforms" << std::endl;
		return err;
	}

	for (auto p : platforms)
	{
		if (p.getInfo<CL_PLATFORM_VENDOR>().find(SELECTED_VENDOR) != std::string::npos)
		{
			platform = p;
			break;
		}
	}
	std::cout << "Selected platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	// Get GPU devices
	err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to get devices" << std::endl;
		return err;
	}

	device = devices[0];
	std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

	// Create context
	cl::Context context(device, nullptr, nullptr, nullptr, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create context" << std::endl;
		return err;
	}
	std::cout << "Context created" << std::endl;

	// Create command queue
	cl::CommandQueue queue(context, device, 0, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create command queue" << std::endl;
		return err;
	}
	std::cout << "Command queue created" << std::endl;

	// Read .cl source file
	std::ifstream infile;
	std::stringstream stream;
	std::string buffer;
	cl::Program::Sources sources;

	infile.open(CL_FILENAME);
	if (infile.is_open() && infile.good())
	{
		stream << infile.rdbuf();
	}
	infile.close();

	buffer = stream.str();
	sources.push_back(std::make_pair(buffer.c_str(), buffer.length()));

	// Create program object
	cl::Program program(context, sources, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create program object" << std::endl;
		return err;
	}

	// Build program
	err = program.build();
	std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to build program" << std::endl;
		return err;
	}
	std::cout << "Build successful" << std::endl;

	// Create kernels
	std::vector<cl::Kernel> kernels;
	err = program.createKernels(&kernels);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create kernels" << std::endl;
		return err;
	}
	std::cout << "Kernels created" << std::endl;

	cl::Kernel luminanceKernel = kernels[0];
	cl::Kernel reductionStepKernel = kernels[1];
	cl::Kernel reductionCompleteKernel = kernels[2];

	// Prepare data
	int w, h, n;
	unsigned char* inputImage = stbi_load(INPUT_IMAGE_FILENAME.c_str(), &w, &h, &n, 4);
	float* luminance = new float[w * h];
	float sum = 0.0f;
	size_t localSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	size_t globalSize = (w * h) / 4;
	cl::ImageFormat imageFormat(CL_RGBA, CL_UNORM_INT8);

	// Create buffers
	cl::Image2D inputImageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, w, h, 0, inputImage, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create image2d object for input" << std::endl;
		return err;
	}

	cl::Buffer luminanceBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * w * h, luminance, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create luminance" << std::endl;
		return err;
	}

	cl::Buffer sumBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), nullptr, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create sum buffer" << std::endl;
		return err;
	}

	// Set kernel arguments
	luminanceKernel.setArg(0, inputImageBuffer);
	luminanceKernel.setArg(1, luminanceBuffer);

	reductionStepKernel.setArg(0, luminanceBuffer);
	reductionStepKernel.setArg(1, sizeof(float) * 4 * localSize, nullptr);

	reductionCompleteKernel.setArg(0, luminanceBuffer);
	reductionCompleteKernel.setArg(1, sizeof(float) * 4 * localSize, nullptr);
	reductionCompleteKernel.setArg(2, sumBuffer);

	// Execute kernels
	// Luminance kernel
	queue.enqueueNDRangeKernel(luminanceKernel, cl::NullRange, cl::NDRange(w, h));

	queue.enqueueReadBuffer(luminanceBuffer, CL_TRUE, 0, sizeof(float) * w * h, luminance);
	stbi_image_free(inputImage);

	// Reduction kernels
	queue.enqueueNDRangeKernel(reductionStepKernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));
	while (globalSize / localSize > localSize)
	{
		globalSize = globalSize / localSize;
		queue.enqueueNDRangeKernel(reductionStepKernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));
	}
	globalSize = globalSize / localSize;
	queue.enqueueNDRangeKernel(reductionCompleteKernel, cl::NullRange, cl::NDRange(globalSize));

	// Fetch result from kernels
	queue.enqueueReadBuffer(sumBuffer, CL_TRUE, 0, sizeof(float), &sum);

	std::cout << "Sum: " << sum << std::endl;
	std::cout << "Avg: " << sum / (w * h) << std::endl;

	delete[] luminance;

	return 0;
}