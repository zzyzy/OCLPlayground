__kernel
void discardPixels(__read_only image2d_t inputImage,
                   __write_only image2d_t outputImage,
                   __read_only sampler_t sampler,
                   __private float luminanceAverage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 pixel = read_imagef(inputImage, sampler, coord);
	float luminance = 0.299f * (pixel.x * 255) + 0.587f * (pixel.y * 255) + 0.114f * (pixel.z * 255);

	if (luminance < luminanceAverage)
	{
		pixel.xyz = 0.0f;
	}

	write_imagef(outputImage, coord, pixel);
}

__kernel
void mergeImages(__read_only image2d_t inputImageA,
                 __read_only image2d_t inputImageB,
                 __write_only image2d_t outputImage,
                 __read_only sampler_t sampler)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 pixelA = read_imagef(inputImageA, sampler, coord);
	float4 pixelB = read_imagef(inputImageB, sampler, coord);
	write_imagef(outputImage, coord, pixelA + pixelB);
}
