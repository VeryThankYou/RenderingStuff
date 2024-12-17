#ifndef SAMPLER_H
#define SAMPLER_H

#include <optix.h>
#include <cuda/random.h>
#include "rand31.h"
#include "drand48.h"
#ifndef rnd
#define rnd rnd31
#endif

// Given a direction vector v sampled around the z-axis of a
// local coordinate system, this function applies the same
// rotation to v as is needed to rotate the z-axis to the
// actual direction n that v should have been sampled around
// [Frisvad, Journal of Graphics Tools 16, 2012;
//  Duff et al., Journal of Computer Graphics Techniques 6, 2017].
__inline__ __host__ __device__ void rotate_to_normal(const float3& normal, float3& v)
{
  float sign = copysignf(1.0f, normal.z);
  const float a = -1.0f/(1.0f + fabsf(normal.z));
  const float b = normal.x*normal.y*a;
  v = make_float3(1.0f + normal.x*normal.x*a, b, -sign*normal.x)*v.x
    + make_float3(sign*b, sign*(1.0f + normal.y*normal.y*a), -normal.y)*v.y
    + normal*v.z;
}

// Given spherical coordinates, where theta is the 
// polar angle and phi is the azimuthal angle, this
// function returns the corresponding direction vector
__inline__ __host__ __device__ float3 spherical_direction(float sin_theta, float cos_theta, float phi)
{
  float sin_phi = sinf(phi), cos_phi = cosf(phi);
  return make_float3(sin_theta*cos_phi, sin_theta*sin_phi, cos_theta);
}

__inline__ __device__ float3 sample_hemisphere(const float3& normal, unsigned int& t)
{
  // Get random numbers
  float cos_theta = rnd(t);
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate new direction as if the z-axis were the normal
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(normal, v);
  return v;
}

__inline__ __device__ float3 sample_cosine_weighted(const float3& normal, unsigned int& t)
{
  // Get random numbers
  float cos_theta = sqrtf(rnd(t));
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate new direction as if the z-axis were the normal
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(normal, v);
  return v;
}

__inline__ __device__ float3 sample_isotropic(float3& forward, unsigned int& t)
{
  float xi = rnd(t);
  float cos_theta = 1.0f - 2.0f*xi;
  float phi = 2.0f*M_PIf*rnd(t);
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(forward, v);
  return v;
}

__inline__ __device__ float3 sample_HG(const float3& forward, float g, unsigned int& t)
{
  float xi = rnd(t);
  float cos_theta;
  if(fabs(g) < 1.0e-3f)
    cos_theta = 1.0f - 2.0f*xi;
  else
  {
    float two_g = 2.0f*g;
    float g_sqr = g*g;
    float tmp = (1.0f - g_sqr)/(1.0f - g + two_g*xi);
    cos_theta = 1.0f/two_g*(1.0f + g_sqr - tmp*tmp);
  }
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate new direction as if the z-axis were the forward direction
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual forward direction and return
  rotate_to_normal(forward, v);
  return v;
}

__inline__ __device__ float3 sample_HG(const float3& forward, float g, unsigned long long& t)
{
  float xi = rnd48(t);
  float cos_theta;
  if(fabs(g) < 1.0e-3f)
    cos_theta = 1.0f - 2.0f*xi;
  else
  {
    float two_g = 2.0f*g;
    float g_sqr = g*g;
    float tmp = (1.0f - g_sqr)/(1.0f - g + two_g*xi);
    cos_theta = 1.0f/two_g*(1.0f + g_sqr - tmp*tmp);
  }
  float phi = 2.0f*M_PIf*rnd48(t);

  // Calculate new direction as if the z-axis were the forward direction
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual forward direction and return
  rotate_to_normal(forward, v);
  return v;
}

__inline__ __device__ float3 sample_barycentric(unsigned int& t)
{
  // Get random numbers
  float sqrt_xi1 = sqrtf(rnd(t));
  float xi2 = rnd(t);

  // Calculate Barycentric coordinates
  float u = 1.0f - sqrt_xi1;
  float v = (1.0f - xi2)*sqrt_xi1;
  float w = xi2*sqrt_xi1;

  // Return barycentric coordinates
  return make_float3(u, v, w);
}

__inline__ __device__ float3 sample_Phong_distribution(const float3& normal, const float3& dir, float shininess, unsigned int& t)
{
  // Get random numbers
  float cos_theta = powf(rnd(t), 1.0f/(shininess + 2.0f));
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate sampled direction as if the z-axis were the reflected direction
  float sin_theta = sqrtf(fmaxf(1.0f - cos_theta*cos_theta, 0.0f));
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual reflected direction
  rotate_to_normal(2.0f*dot(normal, dir)*normal - dir, v);
  return v;
}

__inline__ __device__ float3 sample_Blinn_normal(const float3& normal, float shininess, unsigned int& t)
{
  // Get random numbers
  float cos_theta = powf(rnd(t), 1.0f/(shininess + 2.0f));
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate sampled half-angle vector as if the z-axis were the normal
  float sin_theta = sqrtf(fmaxf(1.0f - cos_theta*cos_theta, 0.0f));
  float3 hv = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal
  rotate_to_normal(normal, hv);
  return hv;
}

__inline__ __device__ float3 sample_Beckmann_normal(const float3& normal, float width, unsigned int& t)
{
  // Get random numbers
  float tan_theta_sqr = -width*width*logf(1.0f - rnd(t));
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate sampled half-angle vector as if the z-axis were the normal
  float cos_theta_sqr = 1.0f/(1.0f + tan_theta_sqr);
  float cos_theta = sqrtf(cos_theta_sqr);
  float sin_theta = sqrtf(1.0f - cos_theta_sqr);
  float3 hv = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal
  rotate_to_normal(normal, hv);
  return hv;
}

__inline__ __device__ float3 sample_GGX_normal(const float3& normal, float roughness, unsigned int& t)
{
  // Get random numbers
  float xi1 = rnd(t);
  float tan_theta_sqr = roughness*roughness*xi1/(1.0f - xi1);
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate sampled half-angle vector as if the z-axis were the normal
  float cos_theta_sqr = 1.0f/(1.0f + tan_theta_sqr);
  float cos_theta = sqrtf(cos_theta_sqr);
  float sin_theta = sqrtf(1.0f - cos_theta_sqr);
  float3 hv = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal
  rotate_to_normal(normal, hv);
  return hv;
}

#endif // SAMPLER_H