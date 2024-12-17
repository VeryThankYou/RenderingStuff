//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <optix.h>
#include "structs.h"

__forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
  const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
  void* ptr = reinterpret_cast<void*>(uptr);
  return ptr;
}

__forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
  const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void traceRadiance(
  OptixTraversableHandle handle,
  const float3&          ray_origin,
  const float3&          ray_direction,
  float                  tmin,
  float                  tmax,
  PayloadRadiance*       payload)
{
#ifdef PASS_PAYLOAD_POINTER
  unsigned int u0, u1;
  packPointer(payload, u0, u1);
#else
  unsigned int u0 = 0u, u1 = 0u, u2 = 0u, u3 = payload->depth, u4 = payload->seed, u5 = payload->emit;
#endif
  optixTrace(
    handle,
    ray_origin,
    ray_direction,
    tmin,
    tmax,
    0.0f,                   // rayTime
    OptixVisibilityMask(1),
    OPTIX_RAY_FLAG_NONE,
    RAY_TYPE_RADIANCE,      // SBT offset
    RAY_TYPE_COUNT,         // SBT stride
    RAY_TYPE_RADIANCE,      // missSBTIndex
#ifdef PASS_PAYLOAD_POINTER
    u0, u1);
#else
    u0, u1, u2, u3, u4, u5);

  payload->result.x = __int_as_float(u0);
  payload->result.y = __int_as_float(u1);
  payload->result.z = __int_as_float(u2);
  payload->depth = u3;
  payload->seed = u4;
  payload->emit = u5;
#endif
}

static __forceinline__ __device__ bool traceOcclusion(
  OptixTraversableHandle handle,
  const float3&          ray_origin,
  const float3&          ray_direction,
  float                  tmin,
  float                  tmax)
{
#ifdef PASS_PAYLOAD_POINTER
  unsigned int occluded = 0u, u1 = 0u;
#else
  unsigned int occluded = 0u, u1 = 0u, u2 = 0u, u3 = 0u, u4 = 0u, u5 = 0u;
#endif
  optixTrace(
    handle,
    ray_origin,
    ray_direction,
    tmin,
    tmax,
    0.0f,                    // rayTime
    OptixVisibilityMask(1),
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
    RAY_TYPE_OCCLUSION,      // SBT offset
    RAY_TYPE_COUNT,          // SBT stride
    RAY_TYPE_OCCLUSION,      // missSBTIndex
#ifdef PASS_PAYLOAD_POINTER
    occluded, u1);
#else
    occluded, u1, u2, u3, u4, u5);
#endif
  return occluded;
}

static __forceinline__ __device__ float traceFeeler(
  OptixTraversableHandle handle,
  const float3&          ray_origin,
  const float3&          ray_direction,
  float                  tmin,
  float                  tmax,
  PayloadFeeler*         payload)
{
#ifdef PASS_PAYLOAD_POINTER
  unsigned int u0, u1;
  packPointer(payload, u0, u1);
#else
  unsigned int occluded = 0u, u1 = 0u, u2 = 0u, u3 = 0u, u4 = 0u, u5 = 0u;
#endif
  optixTrace(
    handle,
    ray_origin,
    ray_direction,
    tmin,
    tmax,
    0.0f,                    // rayTime
    OptixVisibilityMask(1),
    OPTIX_RAY_FLAG_NONE,
    RAY_TYPE_OCCLUSION,      // SBT offset
    RAY_TYPE_COUNT,          // SBT stride
    RAY_TYPE_OCCLUSION,      // missSBTIndex
#ifdef PASS_PAYLOAD_POINTER
    u0, u1);

  return payload->occlusion;
#else
    occluded, u1, u2, u3, u4, u5);

  payload->dist = __int_as_float(u1);
  payload->normal.x = __int_as_float(u2);
  payload->normal.y = __int_as_float(u3);
  payload->normal.z = __int_as_float(u4);
  payload->n1_over_n2 = __int_as_float(u5);
  return occluded;
#endif
}

__forceinline__ __device__ PayloadRadiance* getPayload()
{
  const unsigned int u0 = optixGetPayload_0();
  const unsigned int u1 = optixGetPayload_1();
  return reinterpret_cast<PayloadRadiance*>(unpackPointer(u0, u1));
}

__forceinline__ __device__ PayloadFeeler* getFeelerPayload()
{
  const unsigned int u0 = optixGetPayload_0();
  const unsigned int u1 = optixGetPayload_1();
  return reinterpret_cast<PayloadFeeler*>(unpackPointer(u0, u1));
}

__forceinline__ __device__ void setPayloadResult(const float3& p)
{
  optixSetPayload_0(__float_as_int(p.x));
  optixSetPayload_1(__float_as_int(p.y));
  optixSetPayload_2(__float_as_int(p.z));
}

__forceinline__ __device__ void setPayloadDepth(unsigned int d)
{
  optixSetPayload_3(d);
}

__forceinline__ __device__ void setPayloadSeed(unsigned int t)
{
  optixSetPayload_4(t);
}

__forceinline__ __device__ void setPayloadEmit(unsigned int e)
{
  optixSetPayload_5(e);
}

__forceinline__ __device__ void setPayloadOcclusion(bool occluded)
{
  optixSetPayload_0(static_cast<unsigned int>(occluded));
}

__forceinline__ __device__ void setPayloadDistance(float dist)
{
  optixSetPayload_1(__float_as_int(dist));
}

__forceinline__ __device__ void setPayloadNormal(const float3& n)
{
  optixSetPayload_2(__float_as_int(n.x));
  optixSetPayload_3(__float_as_int(n.y));
  optixSetPayload_4(__float_as_int(n.z));
}

__forceinline__ __device__ void setPayloadRelIOR(float n1_over_n2)
{
  optixSetPayload_5(__float_as_int(n1_over_n2));
}

__forceinline__ __device__ float3 getPayloadResult()
{
  float3 result;
  result.x = __int_as_float(optixGetPayload_0());
  result.y = __int_as_float(optixGetPayload_1());
  result.z = __int_as_float(optixGetPayload_2());
  return result;
}

__forceinline__ __device__ unsigned int getPayloadDepth()
{
  return optixGetPayload_3();
}

__forceinline__ __device__ unsigned int getPayloadSeed()
{
  return optixGetPayload_4();
}

__forceinline__ __device__ unsigned int getPayloadEmit()
{
  return optixGetPayload_5();
}
