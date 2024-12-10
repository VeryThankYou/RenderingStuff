// 02562 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2020
// Copyright (c) DTU Compute 2020

#ifndef AREALIGHT_H
#define AREALIGHT_H

#include <optix.h>
#include <cuda/random.h>
#include "cdf_bsearch.h"
#include "sampler.h"
#include "structs.h"

#define UNIFORM_SAMPLING

__device__ __inline__ uint3 get_light_triangle(unsigned int idx, float3& v0, float3& v1, float3& v2)
{
  const LaunchParams& lp = launch_params;
  const uint3& face = lp.light_idxs[idx];
  v0 = lp.light_verts[face.x];
  v1 = lp.light_verts[face.y];
  v2 = lp.light_verts[face.z];
  return face;
}

__device__ __inline__ void sample_center(const float3& pos, float3& dir, float3& L, float& dist)
{
  // Compute output given the following information.
  //
  // Input:  pos    (observed surface position in the scene)
  //
  // Output: dir    (direction toward the light)
  //         L      (radiance received from the direction dir)
  //         dist   (distance to the sampled position on the light source)
  //
  // Relevant data fields that are available (see above):
  // lp.light_verts    (vertex positions for the indexed face set representing the light source)
  // lp.light_norms    (vertex normals for the indexed face set representing the light source)
  // lp.light_idxs     (vertex indices for each triangle in the indexed face set)
  // lp.light_emission (radiance emitted by each triangle of the light source)
  //
  // Hint: (a) Find the face normal for each triangle (using the function get_light_triangle) and
  //        use these to add up triangle areas and find the average normal.
  //       (b) OptiX includes a function normalize(v) which returns the 
  //       vector v normalized to unit length.
  const LaunchParams& lp = launch_params;
  unsigned int triangles = lp.light_idxs.count;
  float3 center = make_float3(0.0f);
  float3 normal = make_float3(0.0f);
  L = make_float3(0.0f);
  for(unsigned int i = 0; i < triangles; ++i)
  {
    float3 v0, v1, v2;
    get_light_triangle(i, v0, v1, v2);
    center += (v0 + v1 + v2)/3.0f;

    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    float3 n = cross(e0, e1);
    float area = 0.5f*length(n);
    normal += n;
    L += lp.light_emission[i]*area;
  }
  center /= static_cast<float>(triangles);
  normal = normalize(normal);

  dir = center - pos;
  float sqr_dist = dot(dir, dir);
  dist = sqrtf(sqr_dist);
  dir /= dist;

  float cos_theta_prime = fmaxf(dot(normal, -dir), 0.0f);
  L *= cos_theta_prime/sqr_dist;
}

__device__ __inline__ void sample(const float3& pos, float3& dir, float3& L, float& dist, unsigned int& t)
{
  // Compute output given the following information.
  //
  // Input:  pos    (observed surface position in the scene)
  //
  // Output: dir    (direction toward the light)
  //         L      (radiance received from the direction dir)
  //         dist   (distance to the sampled position on the light source)
  //
  // Relevant data fields that are available (see above):
  // lp.light_verts         (vertex positions for the indexed face set representing the light source)
  // lp.light_norms         (vertex normals for the indexed face set representing the light source)
  // lp.light_idxs          (vertex indices for each triangle in the indexed face set)
  // lp.light_emission      (radiance emitted by each triangle in the indexed face set)
  // lp.light_area          (total surface area of light source)
  // lp.light_face_area_cdf (discrete cdf for sampling a triangle index using binary search)
  //
  // Hint: (a) Get random numbers using rnd(t).
  //       (b) There is a cdf_bsearch function available for doing binary search.
  const LaunchParams& lp = launch_params;

  // sample a triangle
#ifdef UNIFORM_SAMPLING
  unsigned int triangle_id = cdf_bsearch(rnd(t), lp.light_face_area_cdf);
#else
  unsigned int triangles = lp.light_idxs.count;
  unsigned int triangle_id = static_cast<unsigned int>(rnd(t)*triangles);
#endif
  float3 v0, v1, v2;
  uint3 face = get_light_triangle(triangle_id, v0, v1, v2);

  // sample a point in the triangle
  float3 bary = sample_barycentric(t);
  float3 light_pos = bary.x*v0 + bary.y*v1 + bary.z*v2;

  // compute the sample normal
  float3 n;
  const float3& n0 = lp.light_norms[face.x];
  const float3& n1 = lp.light_norms[face.y];
  const float3& n2 = lp.light_norms[face.z];
  n = normalize(bary.x*n0 + bary.y*n1 + bary.z*n2);

  // Find distance and direction
  dir = light_pos - pos;
  const float sqr_dist = dot(dir, dir);
  dist = sqrt(sqr_dist);
  dir /= dist;

  // Compute emitted radiance

#ifdef UNIFORM_SAMPLING
  L = lp.light_emission[triangle_id]*(fmaxf(dot(n, -dir), 0.0f)/sqr_dist*lp.light_area);
#else
  float light_area = triangles*0.5f*length(cross(v1 - v0, v2 - v0));
  L = lp.light_emission[triangle_id]*(fmaxf(dot(n, -dir), 0.0f)/sqr_dist*light_area);
#endif
}

#endif // AREALIGHT_H