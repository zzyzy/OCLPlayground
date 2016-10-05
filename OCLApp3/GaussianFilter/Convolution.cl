__kernel
void SimpleConvolution(__read_only image2d_t inputImage,
					   __write_only image2d_t outputImage,
					   sampler_t sampler,
					   __constant float* filter,
					   __private int filterSize)
{
	// Get work-item’s row and column position
	int column = get_global_id(0);
	int row = get_global_id(1);

	// Accumulated pixel value
	float4 sum = (float4)(0.0f);

	// Filter's current index
	int filterIndex = 0;

	int2 coord;
	float4 pixel;

	const int halfFilterSize = filterSize / 2;

	// Iterate over the rows
	for (int i = -(halfFilterSize); i <= halfFilterSize; i++)
	{
		coord.y = row + i;

		// Iterate over the columns
		for (int j = -(halfFilterSize); j <= halfFilterSize; j++)
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
void OnePassConvolution(__read_only image2d_t inputImage,
						__write_only image2d_t outputImage,
						sampler_t sampler,
						__constant float* filter,
						__private int filterSize,
						__private int horizontalPass)
{
	// Get work-item’s row and column position
	int column = get_global_id(0);
	int row = get_global_id(1);

	// Accumulated pixel value
	float4 sum = (float4)(0.0f);

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
