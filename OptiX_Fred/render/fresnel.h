#ifndef FRESNEL_H
#define FRESNEL_H

#include <optix.h>
#include "complex.h"

__host__ __device__ __inline__ float3 sqrtf(const float3& v)
{
  return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

// Helper functions for computing Fresnel reflectance
__device__ __inline__ float fresnel_r_s(float cos_theta1, float cos_theta2, float ior1_over_ior2)
{
	// Compute the perpendicularly polarized component of the Fresnel reflectance
	return (ior1_over_ior2*cos_theta1 - cos_theta2) / (ior1_over_ior2*cos_theta1 + cos_theta2);
}

__device__ __inline__ float fresnel_r_p(float cos_theta1, float cos_theta2, float ior1_over_ior2)
{
	// Compute the parallelly polarized component of the Fresnel reflectance
	return (cos_theta1 - ior1_over_ior2*cos_theta2) / (cos_theta1 + ior1_over_ior2*cos_theta2);
}

__device__ __inline__ float fresnel_R(float cos_theta1, float cos_theta2, float ior1_over_ior2)
{
	// Compute the Fresnel reflectance using fresnel_r_s(...) and fresnel_r_p(...)
	const float r_s = fresnel_r_s(cos_theta1, cos_theta2, ior1_over_ior2);
	const float r_p = fresnel_r_p(cos_theta1, cos_theta2, ior1_over_ior2);
	return (r_s*r_s + r_p*r_p)*0.5f;
}

__host__ __device__ __inline__ float fresnel_R(float cos_theta, float ior1_over_ior2)
{
  const float sin_theta_t_sqr = ior1_over_ior2*ior1_over_ior2*(1.0f - cos_theta*cos_theta);
  if(sin_theta_t_sqr >= 1.0f) return 1.0f;
  const float cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
  return fresnel_R(cos_theta, cos_theta_t, ior1_over_ior2);
}

__host__ __device__ __inline__ float3 fresnel_complex_R(float cos_theta, const float3& eta_sq, const float3& kappa_sq)
{
  const float cos_theta_sqr = cos_theta*cos_theta;
  const float sin_theta_sqr = 1.0f - cos_theta_sqr;
  const float tan_theta_sqr = sin_theta_sqr/cos_theta_sqr;

  const float3 z_real = eta_sq - kappa_sq - sin_theta_sqr;
  const float3 z_imag_sqr = 4.0f*eta_sq*kappa_sq;
  const float3 abs_z = sqrtf(z_real*z_real + z_imag_sqr);
  const float3 two_a = sqrtf(2.0f*(abs_z + z_real));

  float3 c1 = abs_z + cos_theta_sqr;
  float3 c2 = two_a*cos_theta;
  const float3 R_s = (c1 - c2)/(c1 + c2);

  c1 = abs_z + sin_theta_sqr*tan_theta_sqr;
  c2 = two_a*sin_theta_sqr/cos_theta;
  const float3 R_p = R_s*(c1 - c2)/(c1 + c2);

  return (R_s + R_p)*0.5f;
}

__host__ __device__ __inline__ float fresnel_diffuse(float n1_over_n2)
{
  return -1.440f*n1_over_n2*n1_over_n2 + 0.710f*n1_over_n2 + 0.668f + 0.0636f/n1_over_n2;
}

__host__ __device__ __inline__ float3 fresnel_diffuse(const float3& n1_over_n2)
{
  return -1.440f*n1_over_n2*n1_over_n2 + 0.710f*n1_over_n2 + 0.668f + 0.0636f/n1_over_n2;
}

__host__ __device__ __inline__ float two_C1(float n)
{
  float r;
  if(n >= 1.0f)
    r = -9.23372f + n*(22.2272f + n*(-20.9292f + n*(10.2291f + n*(-2.54396f + n*0.254913f))));
  else
    r = 0.919317f + n*(-3.4793f + n*(6.75335f + n*(-7.80989f + n*(4.98554f - n*1.36881f))));
  return r;
}

__host__ __device__ __inline__ float three_C2(float n)
{
  float r;
  if(n >= 1.0f)
  {
    r = -1641.1f + n*(1213.67f + n*(-568.556f + n*(164.798f + n*(-27.0181f + n*1.91826f))));
    r += (((135.926f/n) - 656.175f)/n + 1376.53f)/n;
  }
  else
    r = 0.828421f + n*(-2.62051f + n*(3.36231f + n*(-1.95284f + n*(0.236494f + n*0.145787f))));
  return r;
}

__host__ __device__ __inline__ float C_phi(float ni)
{
  return 0.25f*(1.0f - two_C1(ni));
}

__host__ __device__ __inline__ float C_E(float ni)
{
  return 0.5f*(1.0f - three_C2(ni));
}

// Helper functions for computing Fresnel reflectance with m_complex numbers
__device__ __inline__ m_complex fresnel_r_s(m_complex cos_theta1, m_complex cos_theta2, m_complex ior1_over_ior2)
{
  // Compute the perpendicularly polarized component of the Fresnel reflectance
  return (ior1_over_ior2*cos_theta1 - cos_theta2) / (ior1_over_ior2*cos_theta1 + cos_theta2);
}

__device__ __inline__ m_complex fresnel_r_p(m_complex cos_theta1, m_complex cos_theta2, m_complex ior1_over_ior2)
{
  // Compute the parallelly polarized component of the Fresnel reflectance
  return (cos_theta1 - ior1_over_ior2*cos_theta2) / (cos_theta1 + ior1_over_ior2*cos_theta2);
}

__device__ __inline__ float fresnel_R(m_complex cos_theta1, m_complex cos_theta2, m_complex ior1_over_ior2)
{
  // Compute the Fresnel reflectance using fresnel_r_s(...) and fresnel_r_p(...)
  const m_complex r_s = fresnel_r_s(cos_theta1, cos_theta2, ior1_over_ior2);
  const m_complex r_p = fresnel_r_p(cos_theta1, cos_theta2, ior1_over_ior2);
  return (r_s.re*r_s.re + r_s.im*r_s.im + r_p.re*r_p.re + r_p.im*r_p.im)*0.5f;
}

__host__ __device__ __inline__ float fresnel_R(m_complex cos_theta_i, m_complex ior1_over_ior2)
{
  const m_complex cos_theta_t = sqrt(1.0f - ior1_over_ior2*ior1_over_ior2*(1.0f - cos_theta_i*cos_theta_i));
  return fresnel_R(cos_theta_i, cos_theta_t, ior1_over_ior2);
}

__device__ __inline__ m_complex fresnel_r_s(float cos_theta1, m_complex cos_theta2, m_complex ior1_over_ior2)
{
  // Compute the perpendicularly polarized component of the Fresnel reflectance
  return (ior1_over_ior2*cos_theta1 - cos_theta2) / (ior1_over_ior2*cos_theta1 + cos_theta2);
}

__device__ __inline__ m_complex fresnel_r_p(float cos_theta1, m_complex cos_theta2, m_complex ior1_over_ior2)
{
  // Compute the parallelly polarized component of the Fresnel reflectance
  return (cos_theta1 - ior1_over_ior2*cos_theta2) / (cos_theta1 + ior1_over_ior2*cos_theta2);
}

__device__ __inline__ float fresnel_R(float cos_theta1, m_complex cos_theta2, m_complex ior1_over_ior2)
{
  // Compute the Fresnel reflectance using fresnel_r_s(...) and fresnel_r_p(...)
  const m_complex r_s = fresnel_r_s(cos_theta1, cos_theta2, ior1_over_ior2);
  const m_complex r_p = fresnel_r_p(cos_theta1, cos_theta2, ior1_over_ior2);
  return (abs_sqr(r_s) + abs_sqr(r_p))*0.5f;
}

__host__ __device__ __inline__ float fresnel_R(float cos_theta_i, m_complex ior1_over_ior2)
{
  const m_complex cos_theta_t = sqrt(1.0f - ior1_over_ior2*ior1_over_ior2*(1.0f - cos_theta_i*cos_theta_i));
  return fresnel_R(cos_theta_i, cos_theta_t, ior1_over_ior2);
}

__host__ __device__ __inline__ m_complex fresnel_R_complex(float cos_theta_i, m_complex ior1_over_ior2)
{
  const m_complex ior2_over_ior1 = 1.0f/ior1_over_ior2;
  const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
  const m_complex cos_theta_t_sqr = 1.0f - ior1_over_ior2*ior1_over_ior2*(1.0f - cos_theta_i_sqr);
  const m_complex cos_theta_t = sqrt(cos_theta_t_sqr);
  return (cos_theta_i_sqr - cos_theta_t_sqr)/(cos_theta_i_sqr + cos_theta_t_sqr + (ior1_over_ior2 + ior2_over_ior1)*cos_theta_i*cos_theta_t);
}

__host__ __device__ __inline__ m_complex fresnel_T_complex(float cos_theta_i, m_complex ior1_over_ior2)
{
  const m_complex ior2_over_ior1 = 1.0f/ior1_over_ior2;
  const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
  const m_complex cos_theta_t_sqr = 1.0f - ior1_over_ior2*ior1_over_ior2*(1.0f - cos_theta_i_sqr);
  const m_complex cos_theta_t = sqrt(cos_theta_t_sqr);
  const m_complex i_m_t = cos_theta_i*cos_theta_t;
  const m_complex i2_p_t2 = cos_theta_i_sqr + cos_theta_t_sqr;
  const m_complex numerator = (cos_theta_i_sqr + i_m_t)*(1.0f + ior2_over_ior1);
  return numerator/(i_m_t*(1.0f + ior2_over_ior1*ior2_over_ior1) + ior2_over_ior1*i2_p_t2);
}

#endif // FRESNEL_H