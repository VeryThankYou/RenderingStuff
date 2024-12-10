#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>

__host__ __device__ __inline__ float luminance_NTSC(const float3& color)
{
  return dot(color, make_float3(0.2989f, 0.5866f, 0.1145f));
}

extern "C" __global__ void __raygen__env_luminance()
{
  const LaunchParams& lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int image_idx = launch_idx.y * launch_dims.x + launch_idx.x;
  const float3 uv = (make_float3(launch_idx) + 0.5f)/make_float3(launch_dims);
  const float theta = uv.y*M_PIf;
  const float3 texel = make_float3(tex2D<float4>(lp.envmap, uv.x, uv.y));
  lp.env_luminance[image_idx] = luminance_NTSC(texel)*sinf(theta);
}

extern "C" __global__ void __raygen__env_marginal()
{
  const LaunchParams& lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  float c_f_sum = 0.0f;
  for(unsigned int i = 0; i < lp.env_width; ++i)
  {
    unsigned int idx = i + launch_idx.y*lp.env_width;
    c_f_sum += lp.env_luminance[idx];
  }
  lp.marginal_f[launch_idx.y] = c_f_sum/lp.env_width;
}

extern "C" __global__ void __raygen__env_pdf()
{
  const LaunchParams& lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int image_idx = launch_idx.y*launch_dims.x + launch_idx.x;
  lp.conditional_pdf[image_idx] = lp.env_luminance[image_idx]/lp.marginal_f[launch_idx.y];
  float cdf_sum = 0.0f;
  for(unsigned int i = 0; i <= launch_idx.x; ++i)
  {
    unsigned int idx = i + launch_idx.y*launch_dims.x;
    cdf_sum += lp.env_luminance[idx];
  }
  cdf_sum /= launch_dims.x;
  lp.conditional_cdf[image_idx] = cdf_sum/lp.marginal_f[launch_idx.y];
  if(launch_idx == launch_dims - make_uint3(1u))
    lp.conditional_cdf[image_idx] = 1.0f;  // handle numerical instability

  if(launch_idx.x == 0)
  {
    float m_f_sum = 0.0f;
    for(unsigned int i = 0; i < launch_dims.y; ++i)
    {
      m_f_sum += lp.marginal_f[i];
      if(i == launch_idx.y)
        cdf_sum = m_f_sum;
    }
    m_f_sum /= launch_dims.y;
    cdf_sum /= launch_dims.y;
    lp.marginal_pdf[launch_idx.y] = lp.marginal_f[launch_idx.y]/m_f_sum;
    lp.marginal_cdf[launch_idx.y] = cdf_sum/m_f_sum;
    if(launch_idx.y == launch_dims.y - 1u)
      lp.marginal_cdf[launch_idx.y] = 1.0f; // handle numerical instability
  }
}

__device__ __inline__ unsigned int cdf_bsearch_marginal(const float xi)
{
  const LaunchParams& lp = launch_params;
  unsigned int table_size = lp.env_height;
  unsigned int middle = table_size = table_size>>1;
  unsigned int odd = 0;
  while(table_size > 0)
  {
    odd = table_size&1;
    table_size = table_size>>1;
    unsigned int tmp = table_size + odd;
    middle = xi > lp.marginal_cdf[middle]
      ? middle + tmp
      : (xi < lp.marginal_cdf[middle - 1] ? middle - tmp : middle);
  }
  return middle;
}

__device__ __inline__ unsigned int cdf_bsearch_conditional(const float xi, unsigned int offset)
{
  const LaunchParams& lp = launch_params;
  uint2 table_size = make_uint2(lp.env_width, lp.env_height);
  unsigned int middle = table_size.x = table_size.x>>1;
  unsigned int odd = 0;
  while(table_size.x > 0)
  {
    odd = table_size.x&1;
    table_size.x = table_size.x>>1;
    unsigned int tmp = table_size.x + odd;
    middle = xi > lp.conditional_cdf[middle + offset*lp.env_width]
      ? middle + tmp
      : (xi < lp.conditional_cdf[middle - 1 + offset*lp.env_width] ? middle - tmp : middle);
  }
  return middle;
}

__device__ __inline__ float sample_environment(const float3& pos, float3& dir, float3& L_e, unsigned int& t)
{
  const float M_2PIPIf = 2.0f*M_PIf*M_PIf;

  const LaunchParams& lp = launch_params;
  uint2 count = make_uint2(lp.env_width, lp.env_height);
  const float xi1 = rnd(t), xi2 = rnd(t);

  unsigned int v_idx = cdf_bsearch_marginal(xi1);
  float dv = v_idx > 0
    ? (xi1 - lp.marginal_cdf[v_idx - 1])/(lp.marginal_cdf[v_idx] - lp.marginal_cdf[v_idx - 1])
    : xi1/lp.marginal_cdf[v_idx];
  float pdf_m = lp.marginal_pdf[v_idx];
  float v = (v_idx + dv)/count.y;

  unsigned int u_idx = cdf_bsearch_conditional(xi2, v_idx);
  unsigned int uv_idx_prev = u_idx - 1 + v_idx*lp.env_width;
  unsigned int uv_idx = u_idx + v_idx*lp.env_width;
  float du = u_idx > 0
    ? (xi2 - lp.conditional_cdf[uv_idx_prev])/(lp.conditional_cdf[uv_idx] - lp.conditional_cdf[uv_idx_prev])
    : xi2/lp.conditional_cdf[uv_idx];
  float pdf_c = lp.conditional_pdf[uv_idx];
  float u = (u_idx + du)/count.x;

  float probability = pdf_m*pdf_c;
  float theta = v*M_PIf;
  float phi = (2.0f*u - 0.96f)*M_PIf;
  float sin_theta, cos_theta, sin_phi, cos_phi;
  sincosf(theta, &sin_theta, &cos_theta);
  sincosf(phi, &sin_phi, &cos_phi);
  dir = make_float3(sin_theta*sin_phi, cos_theta, sin_theta*cos_phi);
  L_e = make_float3(tex2D<float4>(lp.envmap, u, v));
  return sin_theta*M_2PIPIf/probability;
}