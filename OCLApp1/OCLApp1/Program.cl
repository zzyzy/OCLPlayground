__kernel
void Add(__global float *a,
         __global float *b,
         __global float *c)
{

   *c = *a + *b;
}

__kernel
void Sub(__global float *a,
         __global float *b,
         __global float *c)
{

   *c = *a - *b;
}

__kernel 
void Mult(__global float *a,
          __global float *b,
          __global float *c)
{

   *c = *a * *b;
}

__kernel
void Div(__global float *a,
         __global float *b,
         __global float *c)
{

   *c = *a / *b;
}
