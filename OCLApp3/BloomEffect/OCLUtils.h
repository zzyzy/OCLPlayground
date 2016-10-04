#pragma once
#ifndef __OCL_UTILS_H__
#define __OCL_UTILS_H__

#include <unordered_map>
#include <CL/cl.hpp>

void
CheckErrorCode(const cl_int& err, const std::string& errMsg);

cl::Device
GetDevice(const std::string& vendorName = "");

cl::Context
MakeContext(const cl::Device& device);

cl::CommandQueue
MakeCommandQueue(const cl::Context& context, const cl::Device& device,
	cl_command_queue_properties properties = 0);

cl::Program
MakeAndBuildProgram(const std::vector<const char*>& sourceFileNames,
	const cl::Context& context, const cl::Device& device);

std::unordered_map<std::string, cl::Kernel>
MakeKernels(cl::Program& program);

cl::Image2D
MakeImage2D(const cl::Context& context,
	cl_mem_flags flags,
	cl::ImageFormat imageFormat,
	size_t w, size_t h,
	size_t rowPitch = 0,
	void* hostPtr = nullptr);

cl::Buffer
MakeBuffer(const cl::Context& context,
	cl_mem_flags flags,
	size_t size,
	void* hostPtr = nullptr);

cl::Sampler
MakeSampler(const cl::Context& context,
	cl_bool normalizedCoords,
	cl_addressing_mode addressingMode,
	cl_filter_mode filterMode);

#endif // __OCL_UTILS_H__
