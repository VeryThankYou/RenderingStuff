#ifndef MICROFACET_H
#define MICROFACET_H

#include "fresnel.h"
#include "sampler.h"

// "Microfacet Models for Refraction through Rough Surfaces" [Walter et al. 2007]

__device__ __inline__ float blinn_G(float cos_theta_i, float cos_theta_o, float cosines)
{
  cos_theta_i = fabs(cos_theta_i);
  return fminf(cosines, fminf(2.0f, 2.0f*cos_theta_i/cos_theta_o));
}

__device__ __inline__ float smith_G1(float cos_theta, float alpha_b_sqr)
{
  float cos_theta_sqr = cos_theta*cos_theta;
  float tan_theta_sqr = (1.0f - cos_theta_sqr)/cos_theta_sqr;
  float a_sqr = 1.0f/(alpha_b_sqr*tan_theta_sqr);
  float a = sqrtf(a_sqr);
  return a < 1.6f ? (3.535f*a + 2.181f*a_sqr)/(1.0f + 2.276f*a + 2.577f*a_sqr) : 1.0f;
}

__device__ __inline__ float smith_G_blinn(float cos_theta_i, float cos_theta_o, float s)
{
  float alpha_b_sqr = 1.0f/(0.5f*s + 1.0f);
  return smith_G1(cos_theta_i, alpha_b_sqr)*smith_G1(cos_theta_o, alpha_b_sqr);
}

__device__ __inline__ float smith_G_beckmann(float cos_theta_i, float cos_theta_o, float width)
{
  float alpha_b_sqr = width*width;
  return smith_G1(cos_theta_i, alpha_b_sqr)*smith_G1(cos_theta_o, alpha_b_sqr);
}

__device__ __inline__ float ggx_G1(float cos_theta, float alpha_g_sqr)
{
  float cos_theta_sqr = cos_theta*cos_theta;
  float tan_theta_sqr = (1.0f - cos_theta_sqr)/cos_theta_sqr;
  return 2.0f/(1.0f + sqrtf(1.0f + alpha_g_sqr*tan_theta_sqr));
}

__device__ __inline__ float ggx_G(float cos_theta_i, float cos_theta_o, float roughness)
{
  float alpha_b_sqr = roughness*roughness;
  return ggx_G1(cos_theta_i, alpha_b_sqr)*ggx_G1(cos_theta_o, alpha_b_sqr);
}

__device__ __forceinline__ float sqr(float x) { return x*x; }

// Returns a float3 with Fresnel term (F), n dot wh, and the remaining BSDF weight
__device__ __inline__ float3 rough_bsdf(const float3& x, const float3& wi, const float3& wo, const float3& n,
                                        float cos_theta_i, float cos_theta_o, const float ior1_over_ior2)
{
  // f_r = F D G / (4 cos_theta_o cos_theta_i)

  if(fabsf(cos_theta_i) < 1.0e-5f)
    return make_float3(0.0f);

  bool transmit = cos_theta_i*cos_theta_o < 0.0f;
  float3 wh = transmit ? -(ior1_over_ior2*wi + wo) : copysignf(1.0f, cos_theta_i)*(wo + wi);
  wh = normalize(wh);

  // Check cosine ratios (early exit)
  float wo_dot_wh = dot(wo, wh);
  float n_dot_wh = dot(n, wh);
  float cos_ratio_o = wo_dot_wh/cos_theta_o;
  if(cos_ratio_o <= 0.0f || n_dot_wh < 1.0e-5f)
    return make_float3(0.0f);
  float wi_dot_wh = dot(wi, wh);
  float cos_ratio_i = wi_dot_wh/cos_theta_i;
  if(cos_ratio_i <= 0.0f)
    return make_float3(0.0f);

  // Compute Fresnel reflectance
  float F = 1.0f;
  if(transmit)
    F = fresnel_R(fabsf(wi_dot_wh), fabsf(wo_dot_wh), ior1_over_ior2);
  else
  {
    float sin_theta_t_sqr = ior1_over_ior2*ior1_over_ior2*(1.0f - wi_dot_wh*wi_dot_wh);
    if(sin_theta_t_sqr < 1.0f)
    {
      float cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
      F = fresnel_R(fabsf(wi_dot_wh), cos_theta_t, ior1_over_ior2);
    }
  }

  float cosine_weight = transmit
    ? fabsf(wi_dot_wh)*cos_ratio_o/sqr(ior1_over_ior2*wi_dot_wh + wo_dot_wh)
    : 0.25f/fabsf(cos_theta_o);
  return make_float3(transmit ? 1.0f - F : F, n_dot_wh, cosine_weight);
}

__device__ __inline__ float blinn_bsdf_cos(const float3& x, const float3& wi, const float3& wo,
                                           const float3& n, float cos_theta_o, const float ior1_over_ior2, const float s)
{
  float cos_theta_i = dot(wi, n);
  float3 bsdf = rough_bsdf(x, wi, wo, n, cos_theta_i, cos_theta_o, ior1_over_ior2);
  const float F = bsdf.x;
  const float n_dot_wh = bsdf.y;
  const float cosine_weight = bsdf.z;
  float G = blinn_G(cos_theta_i, cos_theta_o, cosine_weight);
  float D = M_1_PIf*(s + 2.0f)*0.5f*powf(n_dot_wh, s);
  return F*D*G; // bsdf * cos_theta_i
}

__device__ __inline__ float blinn_smith_G_bsdf_cos(const float3& x, const float3& wi, const float3& wo,
                                                   const float3& n, float cos_theta_o, const float ior1_over_ior2, const float s)
{
  float cos_theta_i = dot(wi, n);
  float3 bsdf = rough_bsdf(x, wi, wo, n, cos_theta_i, cos_theta_o, ior1_over_ior2);
  const float F = bsdf.x;
  const float n_dot_wh = bsdf.y;
  const float cosine_weight = bsdf.z;
  float G = smith_G_blinn(cos_theta_i, cos_theta_o, s)*cosine_weight;
  float D = M_1_PIf*(s + 2.0f)*0.5f*powf(n_dot_wh, s);
  return F*D*G; // bsdf * cos_theta_i
}

__device__ __inline__ float beckmann_bsdf_cos(const float3& x, const float3& wi, const float3& wo, const float3& n,
                                              float cos_theta_o, const float ior1_over_ior2, const float s)
{
  float cos_theta_i = dot(wi, n);
  float3 bsdf = rough_bsdf(x, wi, wo, n, cos_theta_i, cos_theta_o, ior1_over_ior2);
  const float F = bsdf.x;
  const float n_dot_wh = bsdf.y;
  const float cosine_weight = bsdf.z;
  float ctm_sqr = n_dot_wh*n_dot_wh;
  float ttm_sqr = (1.0f - ctm_sqr)/ctm_sqr;
  float G = smith_G_beckmann(cos_theta_i, cos_theta_o, s)*cosine_weight;
  float D = M_1_PIf/(s*s*ctm_sqr*ctm_sqr)*expf(-ttm_sqr/(s*s));
  return F*D*G; // bsdf * cos_theta_i
}

__device__ __inline__ float ggx_bsdf_cos(const float3& x, const float3& wi, const float3& wo, const float3& n,
                                         float cos_theta_o, const float ior1_over_ior2, const float s)
{
  float cos_theta_i = dot(wi, n);
  float3 bsdf = rough_bsdf(x, wi, wo, n, cos_theta_i, cos_theta_o, ior1_over_ior2);
  const float F = bsdf.x;
  const float n_dot_wh = bsdf.y;
  const float cosine_weight = bsdf.z;
  float ctm_sqr = n_dot_wh*n_dot_wh;
  float ttm_sqr = (1.0f - ctm_sqr)/ctm_sqr;
  float G = ggx_G(cos_theta_i, cos_theta_o, s)*cosine_weight;
  float D = M_1_PIf*s*s/sqr(ctm_sqr*(s*s + ttm_sqr));
  return F*D*G; // bsdf * cos_theta_i
}

// Given a sampled half vector, the function returns the direction of the refracted ray in the output argument wi and
// a float2 with its associated cos_theta_i and cosine weight
__device__ float2 rough_refract(float3& wi, const float3& wo, const float3& wh, const float3& n,
                                float cos_theta_o, const float ior1_over_ior2, const float xi)
{
  if(fabsf(cos_theta_o) < 1.0e-5f)
    return make_float2(0.0f);

  float wo_dot_wh = dot(wo, wh);
  float n_dot_wh = dot(n, wh);

  // Early exit check on chi
  float cos_ratio_o = wo_dot_wh/cos_theta_o;
  if(cos_ratio_o <= 0.0f || n_dot_wh < 1.0e-5f)
    return make_float2(0.0f);

  // Compute Fresnel reflectance
  float sin_theta_t_sqr = ior1_over_ior2*ior1_over_ior2*(1.0f - wo_dot_wh*wo_dot_wh);
  float cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
  float R = sin_theta_t_sqr >= 1.0f ? 1.0f : fresnel_R(fabsf(wo_dot_wh), cos_theta_t, ior1_over_ior2);

  // Russian roulette
  wi = xi < R ? 2.0f*wo_dot_wh*wh - wo : ior1_over_ior2*(wh*wo_dot_wh - wo) - wh*cos_theta_t;
  float cos_theta_i = dot(wi, n);
  if(fabsf(cos_theta_i) < 1.0e-5f)
    return make_float2(0.0f);

  // Early exit check on chi
  float cos_ratio_i = dot(wi, wh)/cos_theta_i;
  if(cos_ratio_i <= 0.0f)
    return make_float2(0.0f);

  float cosine_weight = cos_ratio_o/n_dot_wh;
  return make_float2(cos_theta_i, cosine_weight);
}

__device__ float blinn_refract(float3& wi, const float3& wo, const float3& n,
                               float cos_theta_o, const float ior1_over_ior2, const float s, unsigned int& t)
{
  float3 wh = sample_Blinn_normal(n, s, t);
  float2 cosines = rough_refract(wi, wo, wh, n, cos_theta_o, ior1_over_ior2, rnd(t));
  const float cos_theta_i = cosines.x;
  const float cosine_weight = cosines.y;
  return blinn_G(cos_theta_i, cos_theta_o, cosine_weight);
}

__device__ float blinn_smith_G_refract(float3& wi, const float3& wo, const float3& n,
                                       float cos_theta_o, const float ior1_over_ior2, const float s, unsigned int& t)
{
  float3 wh = sample_Blinn_normal(n, s, t);
  float2 cosines = rough_refract(wi, wo, wh, n, cos_theta_o, ior1_over_ior2, rnd(t));
  const float cos_theta_i = cosines.x;
  const float cosine_weight = cosines.y;
  return smith_G_blinn(cos_theta_i, cos_theta_o, s)*cosine_weight;
}

__device__ float beckmann_refract(float3& wi, const float3& wo, const float3& n,
                                  float cos_theta_o, const float ior1_over_ior2, const float s, unsigned int& t)
{
  float3 wh = sample_Beckmann_normal(n, s, t);
  float2 cosines = rough_refract(wi, wo, wh, n, cos_theta_o, ior1_over_ior2, rnd(t));
  const float cos_theta_i = cosines.x;
  const float cosine_weight = cosines.y;
  return smith_G_beckmann(cos_theta_i, cos_theta_o, s)*cosine_weight;
}

__device__ float ggx_refract(float3& wi, const float3& wo, const float3& n,
                             float cos_theta_o, const float ior1_over_ior2, const float s, unsigned int& t)
{
  float3 wh = sample_GGX_normal(n, s, t);
  float2 cosines = rough_refract(wi, wo, wh, n, cos_theta_o, ior1_over_ior2, rnd(t));
  const float cos_theta_i = cosines.x;
  const float cosine_weight = cosines.y;
  return ggx_G(cos_theta_i, cos_theta_o, s)*cosine_weight;
}

// Returns Fresnel term in F and a float2 with n dot wh and the remaining BSDF weight
__device__ __inline__ float2 rough_bsdf(const float3& x, const float3& wi, const float3& wo, const float3& n,
                                        float cos_theta_i, float cos_theta_o, const complex3& recip_ior, float3& F)
{
  // f_r = F D G / (4 cos_theta_o cos_theta_i)

  if(fabsf(cos_theta_i) < 1.0e-5f)
    return make_float2(0.0f);

  float ior1_over_ior2 = (recip_ior.x.re + recip_ior.x.re + recip_ior.x.re)/3.0f;
  bool transmit = cos_theta_i*cos_theta_o < 0.0f;
  float3 wh = transmit ? -(ior1_over_ior2*wi + wo) : copysignf(1.0f, cos_theta_i)*(wo + wi);
  wh = normalize(wh);

  // Check cosine ratios (early exit)
  float wo_dot_wh = dot(wo, wh);
  float n_dot_wh = dot(n, wh);
  float cos_ratio_o = wo_dot_wh/cos_theta_o;
  if(cos_ratio_o <= 0.0f || n_dot_wh < 1.0e-5f)
    return make_float2(0.0f);
  float wi_dot_wh = dot(wi, wh);
  float cos_ratio_i = wi_dot_wh/cos_theta_i;
  if(cos_ratio_i <= 0.0f)
    return make_float2(0.0f);

  // Compute Fresnel reflectance
  float abs_wi_dot_wh = fabsf(wi_dot_wh);
  F = make_float3(fresnel_R(abs_wi_dot_wh, recip_ior.x),
                  fresnel_R(abs_wi_dot_wh, recip_ior.y),
                  fresnel_R(abs_wi_dot_wh, recip_ior.z));
  F = transmit ? 1.0f - F : F;

  float cosine_weight = transmit
    ? fabsf(wi_dot_wh)*cos_ratio_o/sqr(ior1_over_ior2*wi_dot_wh + wo_dot_wh)
    : 0.25f/fabsf(cos_theta_o);
  return make_float2(n_dot_wh, cosine_weight);
}

__device__ __inline__ float3 blinn_bsdf_cos(const float3& x, const float3& wi, const float3& wo, const float3& n,
                                            float cos_theta_o, const complex3& recip_ior, const float s)
{
  float cos_theta_i = dot(wi, n);
  float3 F;
  float2 bsdf = rough_bsdf(x, wi, wo, n, cos_theta_i, cos_theta_o, recip_ior, F);
  const float n_dot_wh = bsdf.x;
  const float cosine_weight = bsdf.y;
  float G = blinn_G(cos_theta_i, cos_theta_o, cosine_weight);
  float D = M_1_PIf*(s + 2.0f)*0.5f*powf(n_dot_wh, s);
  return F*D*G; // bsdf * cos_theta_i
}

__device__ __inline__ float3 blinn_smith_G_bsdf_cos(const float3& x, const float3& wi, const float3& wo, const float3& n,
                                                    float cos_theta_o, const complex3& recip_ior, const float s)
{
  float cos_theta_i = dot(wi, n);
  float3 F;
  float2 bsdf = rough_bsdf(x, wi, wo, n, cos_theta_i, cos_theta_o, recip_ior, F);
  const float n_dot_wh = bsdf.x;
  const float cosine_weight = bsdf.y;
  float G = smith_G_blinn(cos_theta_i, cos_theta_o, s)*cosine_weight;
  float D = M_1_PIf*(s + 2.0f)*0.5f*powf(n_dot_wh, s);
  return F*D*G; // bsdf * cos_theta_i
}

__device__ __inline__ float3 beckmann_bsdf_cos(const float3& x, const float3& wi, const float3& wo, const float3& n,
                                               float cos_theta_o, const complex3& recip_ior, const float s)
{
  float cos_theta_i = dot(wi, n);
  float3 F;
  float2 bsdf = rough_bsdf(x, wi, wo, n, cos_theta_i, cos_theta_o, recip_ior, F);
  const float n_dot_wh = bsdf.x;
  const float cosine_weight = bsdf.y;
  float ctm_sqr = n_dot_wh*n_dot_wh;
  float ttm_sqr = (1.0f - ctm_sqr)/ctm_sqr;
  float G = smith_G_beckmann(cos_theta_i, cos_theta_o, s)*cosine_weight;
  float D = M_1_PIf/(s*s*ctm_sqr*ctm_sqr)*expf(-ttm_sqr/(s*s));
  return F*D*G; // bsdf * cos_theta_i
}

__device__ __inline__ float3 ggx_bsdf_cos(const float3& x, const float3& wi, const float3& wo, const float3& n,
                                          float cos_theta_o, const complex3& recip_ior, const float s)
{
  float cos_theta_i = dot(wi, n);
  float3 F;
  float2 bsdf = rough_bsdf(x, wi, wo, n, cos_theta_i, cos_theta_o, recip_ior, F);
  const float n_dot_wh = bsdf.x;
  const float cosine_weight = bsdf.y;
  float ctm_sqr = n_dot_wh*n_dot_wh;
  float ttm_sqr = (1.0f - ctm_sqr)/ctm_sqr;
  float G = ggx_G(cos_theta_i, cos_theta_o, s)*cosine_weight;
  float D = M_1_PIf*s*s/sqr(ctm_sqr*(s*s + ttm_sqr));
  return F*D*G; // bsdf * cos_theta_i
}

// Given a sampled half vector, the function returns the direction of the refracted ray in the output argument wi and
// the Fresnel weight in F as well as a float2 with its associated cos_theta_i and cosine weight
__device__ float2 rough_refract(float3& wi, const float3& wo, const float3& wh, const float3& n,
                                float cos_theta_o, const complex3& recip_ior, const float xi, float3& F)
{
  if(fabsf(cos_theta_o) < 1.0e-5f)
    return make_float2(0.0f);

  float wo_dot_wh = dot(wo, wh);
  float n_dot_wh = dot(n, wh);

  // Early exit check on chi
  float cos_ratio_o = wo_dot_wh/cos_theta_o;
  if(cos_ratio_o <= 0.0f || n_dot_wh < 1.0e-5f)
    return make_float2(0.0f);

  // Compute Fresnel reflectance
  float abs_wo_dot_wh = fabsf(wo_dot_wh);
  float3 R = make_float3(fresnel_R(abs_wo_dot_wh, recip_ior.x),
                         fresnel_R(abs_wo_dot_wh, recip_ior.y),
                         fresnel_R(abs_wo_dot_wh, recip_ior.z));
  float prob = (R.x + R.y + R.z)/3.0f;
  F = xi < prob ? R/prob : (1.0f - R)/(1.0f - prob);

  // Russian roulette
  float ior1_over_ior2 = (recip_ior.x.re + recip_ior.y.re + recip_ior.z.re)/3.0f;
  float sin_theta_t_sqr = ior1_over_ior2*ior1_over_ior2*(1.0f - wo_dot_wh*wo_dot_wh);
  float cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
  wi = xi < prob ? 2.0f*wo_dot_wh*wh - wo : ior1_over_ior2*(wh*wo_dot_wh - wo) - wh*cos_theta_t;
  float cos_theta_i = dot(wi, n);
  if(fabsf(cos_theta_i) < 1.0e-5f)
    return make_float2(0.0f);

  // Early exit check on chi
  float cos_ratio_i = dot(wi, wh)/cos_theta_i;
  if(cos_ratio_i <= 0.0f)
    return make_float2(0.0f);

  float cosine_weight = cos_ratio_o/n_dot_wh;
  return make_float2(cos_theta_i, cosine_weight);
}

__device__ float3 blinn_refract(float3& wi, const float3& wo, const float3& n,
                               float cos_theta_o, const complex3& recip_ior, const float s, unsigned int& t)
{
  float3 wh = sample_Blinn_normal(n, s, t);
  float3 F;
  float2 cosines = rough_refract(wi, wo, wh, n, cos_theta_o, recip_ior, rnd(t), F);
  const float cos_theta_i = cosines.x;
  const float cosine_weight = cosines.y;
  return F*blinn_G(cos_theta_i, cos_theta_o, cosine_weight);
}

__device__ float3 blinn_smith_G_refract(float3& wi, const float3& wo, const float3& n,
                                       float cos_theta_o, const complex3& recip_ior, const float s, unsigned int& t)
{
  float3 wh = sample_Blinn_normal(n, s, t);
  float3 F;
  float2 cosines = rough_refract(wi, wo, wh, n, cos_theta_o, recip_ior, rnd(t), F);
  const float cos_theta_i = cosines.x;
  const float cosine_weight = cosines.y;
  return F*smith_G_blinn(cos_theta_i, cos_theta_o, s)*cosine_weight;
}

__device__ float3 beckmann_refract(float3& wi, const float3& wo, const float3& n,
                                  float cos_theta_o, const complex3& recip_ior, const float s, unsigned int& t)
{
  float3 wh = sample_Beckmann_normal(n, s, t);
  float3 F;
  float2 cosines = rough_refract(wi, wo, wh, n, cos_theta_o, recip_ior, rnd(t), F);
  const float cos_theta_i = cosines.x;
  const float cosine_weight = cosines.y;
  return F*smith_G_beckmann(cos_theta_i, cos_theta_o, s)*cosine_weight;
}

__device__ float3 ggx_refract(float3& wi, const float3& wo, const float3& n,
                             float cos_theta_o, const complex3& recip_ior, const float s, unsigned int& t)
{
  float3 wh = sample_GGX_normal(n, s, t);
  float3 F;
  float2 cosines = rough_refract(wi, wo, wh, n, cos_theta_o, recip_ior, rnd(t), F);
  const float cos_theta_i = cosines.x;
  const float cosine_weight = cosines.y;
  return F*ggx_G(cos_theta_i, cos_theta_o, s)*cosine_weight;
}
#endif // MICROFACET_H