#pragma once

#ifdef PASS_PAYLOAD_POINTER
__device__ __inline__ float3 rainbow(float f)
{
  const float dx = 0.8f;
  float g = (6.0f - 2.0f*dx)*f + dx;
  float R = fmaxf(0.0f, (3.0f - abs(g - 4.0f) - abs(g - 5.0f))*0.5f);
  float G = fmaxf(0.0f, (4.0f - abs(g - 2.0f) - abs(g - 4.0f))*0.5f);
  float B = fmaxf(0.0f, (3.0f - abs(g - 1.0f) - abs(g - 2.0f))*0.5f);
  return make_float3(R, G, B);
}

__device__ __inline__ float3 hsv2rgb(float h, float s, float v)
{
  float h6 = h*6.0;
  float frac = h6 - floor(h6);
  float4 ell = v*make_float4(1.0 - s, 1.0 - s*frac, 1.0 - s*(1.0 - frac), 1.0);
  return h6 < 1.0 ? make_float3(ell.w, ell.z, ell.x)
    : (h6 < 2.0 ? make_float3(ell.y, ell.w, ell.x)
      : (h6 < 3.0 ? make_float3(ell.x, ell.w, ell.z)
        : (h6 < 4.0 ? make_float3(ell.x, ell.y, ell.w)
          : (h6 < 5.0 ? make_float3(ell.z, ell.x, ell.w)
            : make_float3(ell.w, ell.x, ell.y)))));
}

__device__ __inline__ float3 val2rainbow(float f)
{
  float t = clamp((log10(f) + 5.5f)/5.5f, 0.0f, 1.0f);
  float h = clamp((1.0f - t)*2.0f, 0.0f, 0.65f);
  return hsv2rgb(h, 1.0f, 1.0f);
}

extern "C" __global__ void __raygen__bsdf()
{
  const LaunchParams& lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int frame = lp.subframe_index;
  unsigned int image_idx = launch_idx.y*launch_dims.x + launch_idx.x;
  unsigned int t = tea<16>(image_idx, frame);

  // Generate camera ray (the center of each pixel is at (0.5, 0.5))
  const float2 jitter = make_float2(rnd(t), rnd(t));
  const float2 idx = make_float2(launch_idx.x, launch_idx.y);
  const float2 res = make_float2(launch_dims.x, launch_dims.y);
  float2 x = 2.0f*make_float2(launch_idx.x, launch_idx.y)/float(launch_dims.y) - 1.0f;

  // Trace camera ray
  PayloadRadiance payload;
  payload.result = make_float3(0.0f);
  payload.depth = 0;
  payload.seed = t;
  payload.emit = 1;
  float tmin = 1.0e-4f;
  float tmax = 1.0e16f;

#define REFLECTION

#if defined(PERSPECTIVE)
  const float2 ip_coords = (idx + jitter)/res*2.0f - 1.0f;
  const float3 direction = normalize(ip_coords.x*lp.U + ip_coords.y*lp.V + lp.W);
  traceRadiance(lp.handle, lp.eye, direction, tmin, tmax, &payload);
  float3 result = 0.5f*payload.result + 0.5f;
#elif defined(NORMALS)
  const float2 ip_coords = (idx + jitter)/res*2.0f - 1.0f;
  const float3 direction = lp.lights[0].direction;
  const float3 origin = make_float3(ip_coords.x, ip_coords.y, 0.0f)*lp.beam_factor - 10.0f*direction;
  traceRadiance(lp.handle, origin, direction, tmin, tmax, &payload);
  float3 result = 0.5f*payload.hit_normal + 0.5f;
#else

  float3 result = make_float3(0.0f);
  const float2 ip_coords = (idx + jitter)/res*2.0f - 1.0f;
  const float3 direction = lp.lights[0].direction;
  const float3 origin = make_float3(ip_coords.x, ip_coords.y, 0.0f)*lp.beam_factor - 10.0f*direction;
  traceRadiance(lp.handle, origin, direction, tmin, tmax, &payload);
  //if(payload.depth > 1)
  {
#ifdef REFLECTION
    if(payload.result.z > 0.0f)
    {
#else
    if(prd.result.z < 0.0f)
    {
#endif
      // normal distribution
      //uint2 new_idx = make_uint2(res*(0.5f + make_float2(payload.hit_normal.x, payload.hit_normal.y)*0.5f));
      //result = make_float3(0.2f);

      // geometric optics
      float mo_dot_wo = fabsf(dot(payload.result, payload.hit_normal));
      uint2 new_idx = make_uint2(res*(0.5f + make_float2(payload.result.x, payload.result.y)*0.5f));
      float denom = lp.surface_area*fmaxf(fabsf(direction.z*payload.mi_dot_n), 1.0e-8f);
      result = make_float3(mo_dot_wo/denom);

      // scalar diffraction theory
      //payload.result.z = 0.0f;
      //uint2 new_idx = make_uint2(res*(0.5f + make_float2(payload.hit_normal.x, payload.hit_normal.y)*0.5f));
      //float denom = total_area*fmaxf(abs(direction.z*payload.hit_normal.z), 1.0e-8f);
      //result = make_float4(payload.result*2.0f/denom, 0.0f);

      image_idx = new_idx.y*launch_dims.x + new_idx.x;
      x = 2.0f*make_float2(new_idx.x, new_idx.y)/float(launch_dims.y) - 1.0f;
    }
    }
#endif
  float3 accum_result = make_float3(lp.accum_buffer[image_idx]);
  accum_result += result;
  lp.accum_buffer[image_idx] = make_float4(accum_result, 1.0f);
#if defined(NORMALS) || defined(PERSPECTIVE)
  lp.frame_buffer[image_idx] = make_rgba(accum_result/static_cast<float>(frame + 1));
#else
  //atomicAdd(&lp.accum_buffer[image_idx].x, fmaxf(result.x, 0.0f));
  float3 output = dot(x, x) < 1.0f ? val2rainbow(accum_result.x/static_cast<float>(frame + 1)) : make_float3(0.0f);
  lp.frame_buffer[image_idx] = make_rgba(output);
#endif
  }
#endif