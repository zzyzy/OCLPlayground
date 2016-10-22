#include <iostream>
#include <fstream>
#include <unordered_map>

#include <CL/cl.hpp>

#include "OCLUtils.h"

#define CL_FILENAME "PatternMatching.cl"
#define INPUT_FILENAME "DNA_sequence.txt"

#define PRIME_NUMBERS_KERNEL "PatternMatching"

#define VENDOR_INTEL "Intel"
#define VENDOR_AMD "Advanced Micro Devices"
#define VENDOR_NVIDIA "NVIDIA"
#define SELECTED_VENDOR VENDOR_INTEL

int main()
{
    cl_int err;

    /*
     * 
     * Setup OCL Context
     * 
     */
    cl::Device device = GetDevice(SELECTED_VENDOR);
    cl::Context context = MakeContext(device);
    cl::CommandQueue queue = MakeCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::vector<const char*> sourceFileNames;
    sourceFileNames.push_back(CL_FILENAME);
    cl::Program program = MakeAndBuildProgram(sourceFileNames, context, device);

    std::unordered_map<std::string, cl::Kernel> kernels = MakeKernels(program);

    /*
     *
     * Read text file
     *
     */
    char* data = nullptr;
    size_t size = 0;
    std::ifstream infile;
    infile.open(INPUT_FILENAME);
    if (infile.is_open() && infile.good())
    {
        infile.seekg(0, infile.end);
        size = infile.tellg();
        infile.seekg(0, infile.beg);
        data = new char[size];

        infile.read(data, size);
    }
    infile.close();

    /*
     *
     * Get device information
     *
     */
    auto numberOfWorkGroups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    auto maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    auto globalSize = numberOfWorkGroups * maxWorkGroupSize;
    auto charsPerItem = size / globalSize + 1;

    /*
     *
     * Create buffers 
     * 
     */
    auto textBuffer = MakeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, data);
    auto resultBuffer = MakeBuffer(context, CL_MEM_WRITE_ONLY, size, nullptr);

    return 0;
}
