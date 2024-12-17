// 02562 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2021
// Copyright (c) DTU Compute 2021

#pragma once

__device__ __inline__ unsigned int cdf_bsearch(const float xi, const BufferView<float>& cdf)
{
  unsigned int table_size = cdf.count;
  unsigned int middle = table_size = table_size>>1;
  unsigned int odd = 0;
  while(table_size > 0)
  {
    odd = table_size&1;
    table_size = table_size>>1;
    unsigned int tmp = table_size + odd;
    middle = xi > cdf[middle]
      ? middle + tmp
      : (xi < cdf[middle - 1] ? middle - tmp : middle);
  }
  return middle;
}
