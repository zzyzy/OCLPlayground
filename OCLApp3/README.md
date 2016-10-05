# OCLApp3
This project is about image processing, such as image convolutions, finding the luminance value of a pixel and creating a glowing image.

## Dependencies
1. OpenCL 1.2 from either one of the following SDK
    * Intel OpenCL SDK
    * AMD APP SDK
    * NVIDIA CUDA SDK
2. MSVC++ Platform Toolset v110 (VS2012)

## Building
Open and build the solution with Visual Studio 2012 and above.

## What's implemented
1. Transform color image to grayscale image
2. Parallel reduction to find average luminance of an image
3. Simple gaussian filter convolution
4. Two pass gaussian filter convolution
5. Transform color image to bloom image (make it glow)

## TODOs
1. Bloom image doesn't look like it is glowing at all
2. Avoid branching statements in kernel functions

## References
1. https://www.evl.uic.edu/kreda/gpu/image-convolution/
2. https://en.wikipedia.org/wiki/Relative_luminance
3. http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
4. http://www.gamasutra.com/view/feature/130520/realtime_glow.php
