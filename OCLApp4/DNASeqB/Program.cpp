#include <iostream>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <limits>

#include <CL/cl.hpp>

#include "OCLUtils.h"

#define CL_FILENAME "PatternMatching.cl"
#define INPUT_FILENAME "DNA_sequence.txt"
#define EOF_SYMBOL "//"

#define PRIME_NUMBERS_KERNEL "PatternMatching"

#define VENDOR_INTEL "Intel"
#define VENDOR_AMD "Advanced Micro Devices"
#define VENDOR_NVIDIA "NVIDIA"
#define SELECTED_VENDOR VENDOR_INTEL

char* ReadDNASeq(char* filename, size_t& size, size_t& rowSize, size_t& numberOfRows);

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
    size_t size = 0;
    size_t rowSize = 0;
    size_t numberOfRows = 0;
    char* data = ReadDNASeq(INPUT_FILENAME, size, rowSize, numberOfRows);
    char* output = new char[rowSize * numberOfRows + 1];
#ifdef WIN32
    strcpy_s(output, rowSize * numberOfRows + 1, "");
#else
    strcpy(output, "");
#endif

    std::cout << "DNA Sequence Input:" << std::endl << data << std::endl;

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
    auto resultBuffer = MakeBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, rowSize * numberOfRows + 1, output);

    return 0;
}

char* ReadDNASeq(char* filename, size_t& size, size_t& rowSize, size_t& numberOfRows)
{
    char* data = nullptr;

    std::string buffer;
    std::ifstream infile;
    std::stringstream stream;

    size = 0;
    rowSize = 0;
    numberOfRows = 0;

    infile.open(filename);
    if (infile.is_open() && infile.good())
    {
        while (getline(infile, buffer))
        {
            if (buffer != EOF_SYMBOL)
            {
                stream << buffer;
                ++numberOfRows;
            }
        }

        buffer = stream.str();
        size = buffer.length() + 1;
        data = new char[size];
        std::copy(buffer.begin(), buffer.end(), data);
        data[size - 1] = '\0';

        while (!isdigit(stream.peek()))
        {
            stream.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
        }
        stream >> rowSize;
    }
    infile.close();

    return data;
}
