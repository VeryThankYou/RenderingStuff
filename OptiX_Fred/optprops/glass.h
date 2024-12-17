#ifndef GLASS_H
#define GLASS_H

#include "Medium.h"

__declspec(dllexport) Medium deep_crown_glass();
__declspec(dllexport) Medium crown_glass();
__declspec(dllexport) Medium crown_flint_glass();
__declspec(dllexport) Medium light_flint_glass();
__declspec(dllexport) Medium dense_barium_flint_glass();
__declspec(dllexport) Medium dense_flint_glass();

#endif // GLASS_H
