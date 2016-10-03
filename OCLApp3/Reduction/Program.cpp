#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const std::string INPUT_IMAGE_FILENAME = "Input/bunnycity1.bmp";
const std::string OUTPUT_IMAGE_FILENAME = "Output/bunnycity1.bmp";

int main()
{
	int x, y, n;
	unsigned char* data = stbi_load(INPUT_IMAGE_FILENAME.c_str(), &x, &y, &n, 4);
	float luminanceSum = 0.0f;
	float luminanceAvg = 0.0f;

	if (data != nullptr)
	{
		std::cout << INPUT_IMAGE_FILENAME << std::endl;
		std::cout << x << " * " << y << " * " << n << " = " << x * y * n << std::endl;

		for (auto i = 0; i < x * y * 4; i += 4)
		{
			float luminance = 0.299f * data[i] + 0.587f * data[i + 1] + 0.114f * data[i + 2];
			data[i] = static_cast<int>(luminance);		// R
			data[i + 1] = static_cast<int>(luminance);	// G
			data[i + 2] = static_cast<int>(luminance);	// B
			data[i + 3] = 255;							// A
			luminanceSum += luminance;
		}

		luminanceAvg = luminanceSum / (x * y);

		std::cout << "Luminance sum: " << luminanceSum << std::endl;
		std::cout << "Luminance average: " << luminanceAvg << std::endl;
	}

	// Writes a grayscale image
	stbi_write_bmp(OUTPUT_IMAGE_FILENAME.c_str(), x, y, 4, data);

    stbi_image_free(data);
	return 0;
}
