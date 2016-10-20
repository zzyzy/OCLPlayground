# OCLPlayground
Repository containing various OpenCL small applications for learning OpenCL and parallel computing.

## Dependencies
1. OpenCL 1.1+ (if using CL/cl.h) from either one of the following SDK
    * Intel OpenCL SDK
    * AMD APP SDK
    * NVIDIA CUDA SDK
2. OpenCL 1.2+ (if using CL/cl.hpp)
    * Same as above
3. MSVC++ Platform Toolset v110+ (VS2012+)

## Building
Open and build the respective project solutions with Visual Studio 2012 and above.

## Projects
1. OCLApp1 - Introduction to OpenCL
    * How to get platforms and devices that is available
    * How to retrieve information about platforms and devices
    * How to build OpenCL kernel from source
    * How to create kernels
2. OCLApp2 - Introduction to parallel programming
    * How to effectively enqueue kernels with the right work group size and number of work groups
    * How to 'convert' a serial code to parallel code
3. OCLApp3 - Image processing
    * How to perform image convolution (gaussian filter)
    * How to perform parallel reduction
    * How to apply bloom effect to an image
4. OCLApp4 - Pattern matching and prime numbers
    * KMP algorithm
    * Segmented sieve of eratosthenes
    * Pollard's rho algorithm
