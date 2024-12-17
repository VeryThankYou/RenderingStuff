#ifndef MILK_H
#define MILK_H

#include "Medium.h"
#include "LorenzMie.h"

struct LogNormalParticleDistrib : public LorenzMie::ParticleDistrib
{
  double r_vs;  // volume-to-surface mean particle radius
  double c_s;   // particle radius standard deviation
};

__declspec(dllexport)
void milk(Medium& m,
          LogNormalParticleDistrib& fat,     // Distribution of fat globules
          LogNormalParticleDistrib& casein); // Distribution of casein micelles

__declspec(dllexport)
void milk(Medium& m,
          LogNormalParticleDistrib& fat,     // Distribution of fat globules
          LogNormalParticleDistrib& casein,  // Distribution of casein micelles
          double fat_weight,                 // milk fat weight-%
          double protein_weight = 3.4);      // protein weight-%

__declspec(dllexport)
void milk(Medium& m,
          double fat_weight,                 // milk fat weight-%
          double protein_weight = 3.4,       // protein weight-%
          double fat_r_vs = 475.0e-9,        // volume-to-surface mean radius of fat globules
          double fat_c_s = 0.285,            // radius standard deviation of fat globules
          double casein_r_vs = 43.0e-9,      // volume-to-surface mean radius of casein micelles
          double casein_c_s = 0.23);         // radius standard deviation of casein micelles

__declspec(dllexport)
void homogenized_milk(Medium& m,
                      double fat_weight,            // milk fat weight-%
                      double protein_weight = 3.4,  // protein weight-%
                      double pressure = 20.0,       // Homogenization pressure
                      double casein_r_vs = 43.0e-9, // volume-to-surface mean radius of casein micelles
                      double casein_c_s = 0.23);    // radius standard deviation of casein micelles

__declspec(dllexport)
void unhomogenized_milk(Medium& m,
                        double fat_weight,            // milk fat weight-%
                        double protein_weight = 3.4,  // protein weight-%
                        double casein_r_vs = 43.0e-9, // volume-to-surface mean radius of casein micelles
                        double casein_c_s = 0.23);    // radius standard deviation of casein micelles

#endif // MILK_H
