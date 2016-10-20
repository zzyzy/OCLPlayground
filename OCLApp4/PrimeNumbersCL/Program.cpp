#include <iostream>
#include <fstream>
#include <unordered_map>

#include <CL/cl.hpp>

#include "OCLUtils.h"

#define CL_FILENAME "PrimeNumbers.cl"

#define PRIME_NUMBERS_KERNEL "PrimeNumbers"

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
     * Accept user input
     *
     */
    int startNumber, endNumber;
    std::cout << "Enter the start number: ";
    std::cin >> startNumber;
    while (startNumber < 1)
    {
        std::cout << "Start number cannot be less than 0" << std::endl;
        std::cout << "Enter the start number: ";
        std::cin >> startNumber;
    }

    std::cout << "Enter the end number: ";
    std::cin >> endNumber;
    while (endNumber < startNumber)
    {
        std::cout << "End number cannot be less than start number" << std::endl;
        std::cout << "Enter the end number: ";
        std::cin >> endNumber;
    }

    /*
     *
     * Get device information
     *
     */
    auto numberOfWorkGroups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    auto maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    /*
     *
     * Create buffers 
     * 
     */


    return 0;
}
