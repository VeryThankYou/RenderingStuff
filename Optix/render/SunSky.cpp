
// Based on NVIDIA's OptiX implementation of 2009.
// Code modified by Jeppe Revall Frisvad, 2013 and 2021.

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include "sampler.h"
#include "SunSky.h"

const float PreethamSunSky::cie_table[38][4] =
{
  {380.f, 0.0002f, 0.0000f, 0.0007f},
  {390.f, 0.0024f, 0.0003f, 0.0105f},
  {400.f, 0.0191f, 0.0020f, 0.0860f},
  {410.f, 0.0847f, 0.0088f, 0.3894f},
  {420.f, 0.2045f, 0.0214f, 0.9725f},

  {430.f, 0.3147f, 0.0387f, 1.5535f},
  {440.f, 0.3837f, 0.0621f, 1.9673f},
  {450.f, 0.3707f, 0.0895f, 1.9948f},
  {460.f, 0.3023f, 0.1282f, 1.7454f},
  {470.f, 0.1956f, 0.1852f, 1.3176f},

  {480.f, 0.0805f, 0.2536f, 0.7721f},
  {490.f, 0.0162f, 0.3391f, 0.4153f},
  {500.f, 0.0038f, 0.4608f, 0.2185f},
  {510.f, 0.0375f, 0.6067f, 0.1120f},
  {520.f, 0.1177f, 0.7618f, 0.0607f},

  {530.f, 0.2365f, 0.8752f, 0.0305f},
  {540.f, 0.3768f, 0.9620f, 0.0137f},
  {550.f, 0.5298f, 0.9918f, 0.0040f},
  {560.f, 0.7052f, 0.9973f, 0.0000f},
  {570.f, 0.8787f, 0.9556f, 0.0000f},

  {580.f, 1.0142f, 0.8689f, 0.0000f},
  {590.f, 1.1185f, 0.7774f, 0.0000f},
  {600.f, 1.1240f, 0.6583f, 0.0000f},
  {610.f, 1.0305f, 0.5280f, 0.0000f},
  {620.f, 0.8563f, 0.3981f, 0.0000f},

  {630.f, 0.6475f, 0.2835f, 0.0000f},
  {640.f, 0.4316f, 0.1798f, 0.0000f},
  {650.f, 0.2683f, 0.1076f, 0.0000f},
  {660.f, 0.1526f, 0.0603f, 0.0000f},
  {670.f, 0.0813f, 0.0318f, 0.0000f},

  {680.f, 0.0409f, 0.0159f, 0.0000f},
  {690.f, 0.0199f, 0.0077f, 0.0000f},
  {700.f, 0.0096f, 0.0037f, 0.0000f},
  {710.f, 0.0046f, 0.0018f, 0.0000f},
  {720.f, 0.0022f, 0.0008f, 0.0000f},

  {730.f, 0.0010f, 0.0004f, 0.0000f},
  {740.f, 0.0005f, 0.0002f, 0.0000f},
  {750.f, 0.0003f, 0.0001f, 0.0000f}
};

const PreethamSunSky::Datum PreethamSunSky::data[] = 
{
  {0.38f, 1655.9f, -1.f, -1.f},
  {0.39f, 1623.37f, -1.f, -1.f},
  {0.4f, 2112.75f, -1.f, -1.f},
  {0.41f, 2588.82f, -1.f, -1.f},
  {0.42f, 2582.91f, -1.f, -1.f},
  {0.43f, 2423.23f, -1.f, -1.f},
  {0.44f, 2676.05f, -1.f, -1.f},
  {0.45f, 2965.83f, 0.003f, -1.f},
  {0.46f, 3054.54f, 0.006f, -1.f},
  {0.47f, 3005.75f, 0.009f, -1.f},

  {0.48f, 3066.37f, 0.014f, -1.f},
  {0.49f, 2883.04f, 0.021f, -1.f},
  {0.5f, 2871.21f, 0.03f, -1.f},
  {0.51f, 2782.5f, 0.04f, -1.f},
  {0.52f, 2710.06f, 0.048f, -1.f},
  {0.53f, 2723.36f, 0.063f, -1.f},
  {0.54f, 2636.13f, 0.075f, -1.f},
  {0.55f, 2550.38f, 0.085f, -1.f},
  {0.56f, 2506.02f, 0.103f, -1.f},
  {0.57f, 2531.16f, 0.12f, -1.f},

  {0.58f, 2535.59f, 0.12f, -1.f},
  {0.59f, 2513.42f, 0.115f, -1.f},
  {0.6f, 2463.15f, 0.125f, -1.f},
  {0.61f, 2417.32f, 0.12f, -1.f},
  {0.62f, 2368.53f, 0.105f, -1.f},
  {0.63f, 2321.21f, 0.09f, -1.f},
  {0.64f, 2282.77f, 0.079f, -1.f},
  {0.65f, 2233.98f, 0.067f, -1.f},
  {0.66f, 2197.02f, 0.057f, -1.f},
  {0.67f, 2152.67f, 0.048f, -1.f},

  {0.68f, 2109.79f, 0.036f, -1.f},
  {0.69f, 2072.83f, 0.028f, 0.028f},
  {0.7f, 2024.04f, 0.023f, 0.023f},
  {0.71f, 1987.08f, 0.018f, 0.018f},
  {0.72f, 1942.72f, 0.014f, 0.014f},
  {0.73f, 1907.24f, 0.011f, 0.011f},
  {0.74f, 1862.89f, 0.01f, 0.01f},
  {0.75f, 1825.92f, 0.009f, 0.009f}
};


PreethamSunSky::PreethamSunSky()
  : _sun_theta( M_PIf ),
    _sun_phi(   0.0f ),
    _turbidity( 2.4f ),
    _dirty( true )
{
  _p.sky_up = make_float3( 0.0f, 1.0f, 0.0f );
  _p.overcast = 0.0f;
}

void PreethamSunSky::init()
{
  if( !_dirty ) return;

  _dirty = false;
  

  _p.c0 = make_float3 (  0.1787f * _turbidity - 1.4630f,
                        -0.0193f * _turbidity - 0.2592f,
                        -0.0167f * _turbidity - 0.2608f );

  _p.c1 = make_float3 ( -0.3554f * _turbidity + 0.4275f,
                        -0.0665f * _turbidity + 0.0008f,
                        -0.0950f * _turbidity + 0.0092f );

  _p.c2 = make_float3 ( -0.0227f * _turbidity + 5.3251f,
                        -0.0004f * _turbidity + 0.2125f,
                        -0.0079f * _turbidity + 0.2102f );

  _p.c3 = make_float3 (  0.1206f * _turbidity - 2.5771f,
                        -0.0641f * _turbidity - 0.8989f,
                        -0.0441f * _turbidity - 1.6537f );

  _p.c4 = make_float3 ( -0.0670f * _turbidity + 0.3703f,
                        -0.0033f * _turbidity + 0.0452f,
                        -0.0109f * _turbidity + 0.0529f );

  const float sun_theta_2 = _sun_theta * _sun_theta;
  const float sun_theta_3 = sun_theta_2 * _sun_theta;

  const float xi = ( 4.0f / 9.0f - _turbidity / 120.0f ) * 
                   ( M_PIf - 2.0f * _sun_theta );

  float3 zenith;
  // Preetham paper is in kilocandellas -- we want candellas
  zenith.x = ( ( 4.0453f * _turbidity - 4.9710f) * tan(xi) - 0.2155f * _turbidity + 2.4192f ) * 1000.0f;
  zenith.y = _turbidity * _turbidity * ( 0.00166f*sun_theta_3 - 0.00375f*sun_theta_2 + 0.00209f*_sun_theta ) +
                          _turbidity * (-0.02903f*sun_theta_3 + 0.06377f*sun_theta_2 - 0.03202f*_sun_theta + 0.00394f) +
                                       ( 0.11693f*sun_theta_3 - 0.21196f*sun_theta_2 + 0.06052f*_sun_theta + 0.25886f);
  zenith.z = _turbidity * _turbidity * ( 0.00275f*sun_theta_3 - 0.00610f*sun_theta_2 + 0.00317f*_sun_theta ) +
                          _turbidity * (-0.04214f*sun_theta_3 + 0.08970f*sun_theta_2 - 0.04153f*_sun_theta + 0.00516f) +
                                       ( 0.15346f*sun_theta_3 - 0.26756f*sun_theta_2 + 0.06670f*_sun_theta + 0.26688f);


  float cos_sun_theta = cosf( _sun_theta );

  float3 divisor_Yxy = ( 1.0f + _p.c0 * expf(_p.c1) ) * 
                       ( 1.0f + _p.c2 * expf(_p.c3*_sun_theta) + _p.c4 * cos_sun_theta * cos_sun_theta );
  
  _p.inv_divisor_Yxy =  zenith / divisor_Yxy;

  // 
  // Direct sunlight
  //
  _p.sun_color = sunColor();
  _p.sun_dir = spherical_direction(sinf(_sun_theta), cos_sun_theta, _sun_phi);
  rotate_to_normal(_p.sky_up, _p.sun_dir);
}

// lambda is wavelength in micro meters
float PreethamSunSky::calculateAbsorption( float sun_theta, float m, float lambda, float turbidity, float k_o, float k_wa ) 

{
  float alpha = 1.3f;                             // wavelength exponent
  float beta  = 0.04608f * turbidity - 0.04586f;  // turbidity coefficient
  float ell   = 0.35f;                            // ozone at NTP (cm)
  float w     = 2.0f;                             // precipitable water vapor (cm)

  float rayleigh_air   = expf( -0.008735f * m * powf( lambda, -4.08f ) );
  float aerosol        = expf( -beta * m * powf( lambda, -alpha ) );
  float ozone          = k_o  > 0.0f ? expf( -k_o*ell*m )                                              : 1.0f;
  float water_vapor    = k_wa > 0.0f ? expf( -0.2385f*k_wa*w*m/powf( 1.0f + 20.07f*k_wa*w*m, 0.45f ) ) : 1.0f;

  return rayleigh_air*aerosol*ozone*water_vapor;
}

float3 PreethamSunSky::sunColor()
{
  init();

  const float elevation_angle = 93.885f  - _sun_theta*180.0f/M_PIf;
  if(elevation_angle < 0.0f)
    return make_float3(0.0f);

  // optical mass
  const float cos_sun_theta = cos( _sun_theta );
  const float m = 1.0f / ( cos_sun_theta + 0.15f * powf(elevation_angle, -1.253f) );

  float results[38];
  for( int i = 0; i < 38; ++i ) {
    results[i]  = data[i].sun_spectral_radiance * 10000.0f // convert from 1/cm^2 to 1/m^2;
                                                / 1000.0f; // convert from micrometer to nanometer

    results[i] *= calculateAbsorption( _sun_theta, m, data[i].wavelength, _turbidity, data[i].k_o, data[i].k_wa );
  }


  float X = 0.0f, Y = 0.0f, Z = 0.0f;
  for( int i = 0; i < 38; ++i ) {
    X += results[i]*cie_table[i][1] * 10.0f; 
    Y += results[i]*cie_table[i][2] * 10.0f; 
    Z += results[i]*cie_table[i][3] * 10.0f; 
  }
  return XYZ2rgb(make_float3(X, Y, Z))*_p.tonemap; // return result in W/m^2/sr (using a scale to handle dynamic range)
}

float3 PreethamSunSky::skyColor( const float3 & direction, bool CEL )
{
  init();

  // Preetham skylight model
  float3 preetham_color = make_float3(0.0f);
  if(_p.overcast < 1.0f) {
    if(CEL && dot(direction, _p.sun_dir) > 0.999988541392f) // 94.0f / sqrtf(94.0f*94.0f + 0.45f*0.45f)
      preetham_color = _p.sun_color;
    else
      preetham_color = clear_sky_color(direction, _p);
  }

  // CIE standard overcast sky model
  float3 overcast_color = overcast_sky_color(direction, _p);

  // return linear combo of the two
  return lerp(preetham_color, overcast_color, _p.overcast);
}

