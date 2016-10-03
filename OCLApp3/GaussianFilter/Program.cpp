#include <iostream>
#include <fstream>
#include <sstream>

#include <CL/cl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Filters.h"
#include "KernelFunctions.h"

const std::string CL_FILENAME = "Convolution.cl";
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

	cl::Kernel convolutionKernel = kernels[0];

	// Prepare data
	int w, h, n;
	unsigned char* inputImage = stbi_load(INPUT_IMAGE_FILENAME.c_str(), &w, &h, &n, 4);
	unsigned char* outputImage;
	cl::ImageFormat imageFormat(CL_RGBA, CL_UNORM_INT8);
	int fx = 3, fy = 3;
	float* selectedFilter = const_cast<float*>(GaussianFilter3x3);

	// Create buffers
	cl::Image2D inputImageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, w, h, 0, inputImage, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create image2d object for input image" << std::endl;
		return err;
	}

	cl::Image2D outputImageBuffer(context, CL_MEM_WRITE_ONLY, imageFormat, w, h, 0, nullptr, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create image2d object for output image" << std::endl;
		return err;
	}

	cl::Buffer filterBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * fx * fy,
	                        selectedFilter, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": Unable to create filter buffer" << std::endl;
		return err;
	}

	// Set kernel arguments


	// Execute kernels
	outputImage = simpleConvolution(convolutionKernel, queue, inputImageBuffer, outputImageBuffer, filterBuffer, fx, fy);

	// Fetch result from kernels
	stbi_write_bmp(OUTPUT_IMAGE_FILENAME.c_str(), w, h, 4, outputImage);

	delete[] outputImage;
	stbi_image_free(inputImage);

	return 0;
}
