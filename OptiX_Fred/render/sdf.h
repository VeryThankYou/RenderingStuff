#pragma once

__device__ const float precis = 0.001f;

__inline__ __device__ float get_dist(cudaTextureObject_t sdf, const float3& p)
{
  return tex3D<float>(sdf, p.x, p.y, p.z)/128.0f;
}

__inline__ __device__ float raycast(const float3& ro, const float3& rd, float tmin, float tmax, cudaTextureObject_t sdf)
{
  // ray marching
  float t = tmin;
  float d = get_dist(sdf, ro + t*rd);
  const float sgn = copysignf(1.0f, d);
  for(unsigned int i = 0; i < 100u; ++i)
  {
    if(fabsf(d) < precis*t || t > tmax) break;
    t += sgn*d; // *1.2f;
    d = get_dist(sdf, ro + t*rd);
  }
  return t;
}

// https://iquilezles.org/articles/normalsSDF
__inline__ __device__ float3 calcNormal(const float3& pos, cudaTextureObject_t sdf)
{
  float2 e = make_float2(1.0f, -1.0f)*0.5773f*precis;
  return normalize(
    make_float3(e.x, e.y, e.y)*get_dist(sdf, pos + make_float3(e.x, e.y, e.y)) +
    make_float3(e.y, e.y, e.x)*get_dist(sdf, pos + make_float3(e.y, e.y, e.x)) +
    make_float3(e.y, e.x, e.y)*get_dist(sdf, pos + make_float3(e.y, e.x, e.y)) +
    make_float3(e.x)*get_dist(sdf, pos + make_float3(e.x)));
}