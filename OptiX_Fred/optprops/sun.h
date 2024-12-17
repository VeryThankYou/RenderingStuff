#ifndef SUN_H
#define SUN_H

#include "Medium.h"

//CGLA::Vec2f sun_position(double day, double time, double latitude);
__declspec(dllexport) Medium mean_solar_irrad();
//Medium solar_irrad(double day, double time, double latitude, const CGLA::Vec3f& up, CGLA::Vec3f& direction);
//Medium direct_sun(double day, double time, double latitude, const CGLA::Vec3f& up, CGLA::Vec3f& direction);
//Medium atmosphere(double day, double time, double latitude, const CGLA::Vec3f& up, CGLA::Vec3f& direction);

#endif // SUN_H
