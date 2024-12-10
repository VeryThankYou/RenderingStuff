// 02562 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2020
// Copyright (c) DTU Compute 2020

#ifndef DIRECTIONAL_H
#define DIRECTIONAL_H

#include <optix.h>

struct Directional
{
  float3 direction;
  float3 emission;
};

#endif // DIRECTIONAL_H