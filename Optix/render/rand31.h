// Code by Jeppe Revall Frisvad
// Based on Hui-Ching Tang, An analysis of linear congruential random number generators when
// multiplier restrictions exist, European Journal of Operational Research 182(2):820-828, 2007.
// Copyright (c) 2023 DTU Compute

#pragma once

// Generate random unsigned int in [0, 2^31)
static __host__ __device__ __inline__ unsigned int lcg31(unsigned int& prev)
{
  const unsigned int LCG_A = 1977654935u;
  prev = (LCG_A * prev) & 0x7FFFFFFF;
  return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd31(unsigned int& prev)
{
  return static_cast<float>(lcg31(prev)) / static_cast<float>(0x80000000);
}

