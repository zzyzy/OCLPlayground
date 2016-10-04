__kernel
void simpleConvolution(__read_only image2d_t inputImage,
                       __write_only image2d_t outputImage,
					   __read_only sampler_t sampler,
                       __constant float* filter,
                       __private int fx,
                       __private int fy)
{
	// Get work-item’s row and column position
	int column = get_global_id(0);
	int row = get_global_id(1);

	// Accumulated pixel value
	float4 sum = (float4)(0.0);

	// Filter's current index
	int filterIndex = 0;

	int2 coord;
	float4 pixel;

	const int hfx = fx / 2;
	const int hfy = fy / 2;

	// Iterate over the rows
	for (int i = -(hfx); i <= hfx; i++)
	{
		coord.y = row + i;

		// Iterate over the columns
		for (int j = -(hfy); j <= hfy; j++)
		{
			coord.x = column + j;

			// Read value pixel from the image
			pixel = read_imagef(inputImage, sampler, coord);

			// Acculumate weighted sum
			sum.xyz += pixel.xyz * filter[filterIndex++];
			sum.w = 1.0f;
		}
	}

	// Write new pixel value to output
	coord = (int2)(column, row);
	write_imagef(outputImage, coord, sum);
}

__kernel
void onePassConvolution(__read_only image2d_t inputImage,
                        __write_only image2d_t outputImage,
						__read_only sampler_t sampler,
                        __constant float* filter,
                        __private int filterSize,
                        __private int horizontalPass)
{
	// Get work-item’s row and column position
	int column = get_global_id(0);
	int row = get_global_id(1);

	// Accumulated pixel value
	float4 sum = (float4)(0.0);

	// Filter's current index
	int filterIndex = 0;

	int2 coord = (int2)(column, row);
	float4 pixel;

	const int halfFilterSize = filterSize / 2;

	// Iterate over the filter
	for (int i = -(halfFilterSize); i <= halfFilterSize; i++)
	{
		if (horizontalPass)
		{
			coord.x = column + i;
		}
		else
		{
			coord.y = row + i;
		}

		// Read value pixel from the image
		pixel = read_imagef(inputImage, sampler, coord);

		// Acculumate weighted sum
		sum.xyz += pixel.xyz * filter[filterIndex++];
		sum.w = 1.0f;
	}

	// Write new pixel value to output
	coord = (int2)(column, row);
	write_imagef(outputImage, coord, sum);
}
