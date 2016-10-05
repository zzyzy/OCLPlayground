#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

#include <CL/cl.hpp>

bool BuildProgram(const std::string& filename,
                  const cl::Context& context,
                  const cl::Device& device,
                  cl::Program* program);

int main()
{
	/*
		Setup platform and device objects
	*/
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Platform defaultPlatform;
	cl_int err = -1;

	cl::Platform::get(&platforms);
	if (platforms.empty())
	{
		std::cerr << "Unable to find any platforms" << std::endl;
		return err;
	}

	std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl << std::endl;

	for (auto platform : platforms)
	{
		std::cout << "CL_PLATFORM_NAME: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
		std::cout << "CL_PLATFORM_VENDOR: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
		std::cout << "CL_PLATFORM_VERSION: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
		std::cout << std::endl;
	}

	defaultPlatform = platforms[0];
	std::cout << "Selected platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.empty())
	{
		std::cerr << "Unable to find any devices for this platform" << std::endl;
		return err;
	}

	std::cout << "Number of devices: " << devices.size() << std::endl;

	for (auto device : devices)
	{
		std::cout << "CL_DEVICE_TYPE: ";
		auto deviceType = device.getInfo<CL_DEVICE_TYPE>();
		switch (deviceType)
		{
		case CL_DEVICE_TYPE_CPU:
			std::cout << "CPU" << std::endl;
			break;
		case CL_DEVICE_TYPE_GPU:
			std::cout << "GPU" << std::endl;
			break;
		case CL_DEVICE_TYPE_ACCELERATOR:
			std::cout << "Accelerator" << std::endl;
			break;
		case CL_DEVICE_TYPE_DEFAULT:
			std::cout << "Default" << std::endl;
			break;
		default:
			break;
		}

		std::cout << "CL_DEVICE_NAME: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
		std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;

		auto workItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		for (size_t i = 0; i < workItemSizes.size(); ++i)
		{
			std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES[" << i << "]: " << workItemSizes[i] << std::endl;
		}

		std::cout << std::endl;
	}

	/*
		Setup context and command queue
	*/
	cl::Device defaultDevice;
	std::unique_ptr<cl::Context> context; // TODO Something wrong with cl::Context assignment operator
	cl::CommandQueue commandQueue;

	// Create context with the first GPU device
	for (auto device : devices)
	{
		if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU)
		{
			defaultDevice = device;
			break;
		}
	}

	// If there are no GPU devices, use the first CPU device
	if (defaultDevice.getInfo<CL_DEVICE_NAME>() == "")
	{
		for (auto device : devices)
		{
			if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU)
			{
				defaultDevice = device;
				break;
			}
		}
	}

	if (defaultDevice.getInfo<CL_DEVICE_NAME>() == "")
	{
		std::cerr << "No available CPU or GPU device" << std::endl;
		return err;
	}

	context = std::unique_ptr<cl::Context>(new cl::Context(defaultDevice, nullptr, nullptr, nullptr, &err));
	if (err != CL_SUCCESS)
	{
		std::cerr << "Unable to create context" << std::endl;
		return err;
	}

	std::cout << "Created context" << std::endl;

	commandQueue = cl::CommandQueue(*context, defaultDevice, 0, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Unable to create command queue" << std::endl;
		return err;
	}

	std::cout << "Created command queue" << std::endl;

	std::cout << "Number of devices in context: " << context->getInfo<CL_CONTEXT_NUM_DEVICES>() << std::endl;
	auto contextDevices = context->getInfo<CL_CONTEXT_DEVICES>();
	for (size_t i = 0; i < contextDevices.size(); ++i)
	{
		std::cout << "CL_CONTEXT_DEVICES[" << i << "]: " << contextDevices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
	}

	std::cout << std::endl;

	/*
		Build program from source
	*/
	cl::Program program;

	if (!BuildProgram("Program.cl", *context, contextDevices[0], &program))
	{
		std::cerr << "Unable to build program" << std::endl;
		return err;
	}

	std::cout << "Successfully built program" << std::endl;

	for (auto d : program.getInfo<CL_PROGRAM_DEVICES>())
	{
		std::cout << d.getInfo<CL_DEVICE_NAME>() << std::endl;
	}

	std::cout << program.getInfo<CL_PROGRAM_SOURCE>() << std::endl;

	/*
		Create kernel objects
	*/
	std::vector<cl::Kernel> kernels;

	if (program.createKernels(&kernels) != CL_SUCCESS)
	{
		std::cerr << "Unable to create kernels from program" << std::endl;
		return err;
	}

	std::cout << "Successfully created kernels from the program" << std::endl;
	std::cout << "Number of kernels: " << kernels.size() << std::endl;

	for (size_t i = 0; i < kernels.size(); ++i)
	{
		std::cout << "CL_KERNEL_FUNCTION_NAME[" << i << "]: " << kernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>() << std::endl;
	}

	return 0;
}

bool BuildProgram(const std::string& filename,
                  const cl::Context& context,
                  const cl::Device& device,
                  cl::Program* program)
{
	std::ifstream infile;
	std::stringstream stream;
	std::string buffer;
	cl::Program::Sources sources;
	cl_int err;

	infile.open(filename.c_str());
	if (infile.is_open() && infile.good())
	{
		stream << infile.rdbuf();
	}
	infile.close();

	buffer = stream.str();
	sources.push_back(std::make_pair(buffer.c_str(), buffer.length()));

	*program = cl::Program(context, sources, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << "Unable to create program object" << std::endl;
		return false;
	}

	if (program->build() != CL_SUCCESS)
	{
		std::cerr << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		return false;
	}

	std::cout << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
	return true;
}
