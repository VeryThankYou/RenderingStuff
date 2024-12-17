#ifndef PHASEFUNC_H
#define PHASEFUNC_H

#include <optix.h>

__host__ __device__ __inline__ float p_isotropic()
{
  return M_1_PIf*0.25f;
}

__host__ __device__ __inline__ float p_rayleigh(float cos_theta)
{
  return M_1_PIf*0.1875f*(1.0f + cos_theta*cos_theta);
}

__host__ __device__ __inline__ float p_hg(float cos_theta, float g)
{
  float g_sqr = g*g;
  float tmp = 1.0f + g_sqr - 2.0f*g*cos_theta;
  return M_1_PIf*0.25f*(1.0f - g_sqr)/powf(tmp, 1.5f);
}

__host__ __device__ __inline__ float3 p_hg(float cos_theta, float3 g)
{
  float3 g_sqr = g*g;
  float3 tmp = 1.0f + g_sqr - 2.0f*g*cos_theta;
  return M_1_PIf*0.25f*(1.0f - g_sqr)/(tmp*sqrtf(tmp));
}

#endif // PHASEFUNC_H