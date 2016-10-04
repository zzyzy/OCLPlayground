#include "OCLUtils.h"
#include <iostream>
#include <fstream>
#include <sstream>

void CheckErrorCode(const cl_int& err, const std::string& errMsg)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error " << err << ": " << errMsg << std::endl;
		throw std::runtime_error("Error " + std::to_string(err) + ": " + errMsg);
	}
}

cl::Device GetDevice(const std::string& vendorName)
{
	cl_int err;
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Platform platform;
	cl::Device device;

	// Get all platforms
	err = cl::Platform::get(&platforms);
	CheckErrorCode(err, "Unable to get OpenCL platforms");
	std::cout << "Found " << platforms.size() << " platform(s)" << std::endl;

	// Get specified vendor platform
	if (vendorName.empty())
	{
		platform = platforms[0];
	}
	else
	{
		for (auto p : platforms)
		{
			if (p.getInfo<CL_PLATFORM_VENDOR>().find(vendorName) != std::string::npos)
			{
				platform = p;
				break;
			}
		}
	}
	std::cout << "Selected platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	// Get GPU devices
	err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	CheckErrorCode(err, "Unable to get devices");
	std::cout << "Found " << devices.size() << " device(s)" << std::endl;

	// Use first GPU device
	device = devices[0];
	std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

	return device;
}

cl::Context MakeContext(const cl::Device& device)
{
	cl_int err;

	cl::Context context(device, nullptr, nullptr, nullptr, &err);
	CheckErrorCode(err, "Unable to create context");

	return context;
}

cl::CommandQueue MakeCommandQueue(const cl::Context& context, const cl::Device& device)
{
	cl_int err;

	cl::CommandQueue queue(context, device, 0, &err);
	CheckErrorCode(err, "Unable to create command queue");

	return queue;
}

cl::Program MakeAndBuildProgram(const std::vector<const char*>& sourceFileNames, const cl::Context& context, const cl::Device& device)
{
	cl_int err;
	std::ifstream infile;
	std::stringstream stream;
	std::string buffer;
	cl::Program program;
	cl::Program::Sources sources;

	for (auto fileName : sourceFileNames)
	{
		infile.open(fileName);
		if (infile.is_open() && infile.good())
		{
			stream << infile.rdbuf();
		}
		infile.close();
	}

	buffer = stream.str();
	sources.push_back(std::make_pair(buffer.c_str(), buffer.length()));

	// Create program object
	program = cl::Program(context, sources, &err);
	CheckErrorCode(err, "Unable to create program object");

	// Build program
	err = program.build();
	std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
	CheckErrorCode(err, "Unable to build program");
	std::cout << "Build successful" << std::endl;

	return program;
}

std::unordered_map<std::string, cl::Kernel> MakeKernels(cl::Program& program)
{
	cl_int err;
	std::vector<cl::Kernel> tempKernels;
	std::unordered_map<std::string, cl::Kernel> kernels;

	err = program.createKernels(&tempKernels);
	CheckErrorCode(err, "Unable to create kernels");
	std::cout << tempKernels.size() << " Kernels created" << std::endl;

	for (auto k : tempKernels)
	{
		std::cout << k.getInfo<CL_KERNEL_FUNCTION_NAME>() << " kernel created" << std::endl;
		kernels.insert(std::make_pair(k.getInfo<CL_KERNEL_FUNCTION_NAME>(), k));
	}

	return kernels;
}

cl::Image2D MakeImage2D(const cl::Context& context, cl_mem_flags flags, cl::ImageFormat imageFormat, size_t w, size_t h, size_t rowPitch, void* hostPtr)
{
	cl_int err;

	cl::Image2D image2D(context, flags, imageFormat, w, h, rowPitch, hostPtr, &err);
	CheckErrorCode(err, "Unable to create image2D object");

	return image2D;
}

cl::Buffer MakeBuffer(const cl::Context& context, cl_mem_flags flags, size_t size, void* hostPtr)
{
	cl_int err;

	cl::Buffer buffer(context, flags, size, hostPtr, &err);
	CheckErrorCode(err, "Unable to create buffer object");

	return buffer;
}

cl::Sampler MakeSampler(const cl::Context& context, cl_bool normalizedCoords, cl_addressing_mode addressingMode, cl_filter_mode filterMode)
{
	cl_int err;

	cl::Sampler sampler(context, normalizedCoords, addressingMode, filterMode, &err);
	CheckErrorCode(err, "Unable to create sampler");

	return sampler;
}
