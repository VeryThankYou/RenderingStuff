// Code by Jeppe Revall Frisvad
// Copyright (c) 2023 DTU Compute

#pragma once

template<unsigned int N>
static __host__ __device__ __inline__ unsigned long long tea48(unsigned int val0, unsigned int val1)
{
  unsigned long long v0 = val0;
  unsigned long long v1 = val1;
  unsigned long long s0 = 0;

  for(unsigned int n = 0; n < N; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }
  return ((v0<<16)&0xFFFFFFFFFFFF0000) | 0x330E;
}

// Generate random unsigned int in [0, 2^48)
static __host__ __device__ __inline__ unsigned long long lcg48(unsigned long long& prev)
{
  const unsigned long long LCG_A = 0x5DEECE66D;
  const unsigned long long LCG_C = 0x00000000B;
  prev = (LCG_A * prev + LCG_C) & 0x0000FFFFFFFFFFFF;
  return prev;
}

// Generate random double in [0, 1)
static __host__ __device__ __inline__ double drnd48(unsigned long long& prev)
{
  return static_cast<double>(lcg48(prev)) / static_cast<double>(0x0001000000000000);
}

// Generate random float in [0, 1]
static __host__ __device__ __inline__ float rnd48(unsigned long long& prev)
{
  return static_cast<float>(lcg48(prev)) / static_cast<float>(0x0000FFFFFFFFFFFF);
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd48_half_open(unsigned long long& prev)
{
  lcg48(prev);
  return static_cast<float>((prev&0x0000FFFFFF000000) == 0x0000FFFFFF000000 ? prev&0x0000FFFFFF7FFFFF : prev) / static_cast<float>(0x0001000000000000);
}
