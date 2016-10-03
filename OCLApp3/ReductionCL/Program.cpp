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

	// Prepare data


	// Create buffers


	// Set kernel arguments


	// Execute kernels


	// Fetch result from kernels


	return 0;
}