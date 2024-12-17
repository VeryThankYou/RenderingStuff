#include <iostream>
#include <complex>
#include <cmath>
#include "LorenzMie.h"
#include "orange_juice.h"

using namespace std;
using namespace LorenzMie;

namespace
{
#ifndef M_PI
  const double M_PI = 3.14159265358979323846;
#endif

  // Data from Hale and Querry [1973] and Pope and Fry [1997]
  complex<double> refrac_water[] = { complex<double>(1.341, 3.393e-10),
                                     complex<double>(1.339, 2.110e-10),
                                     complex<double>(1.338, 1.617e-10),
                                     complex<double>(1.337, 3.302e-10),
                                     complex<double>(1.336, 4.309e-10),
                                     complex<double>(1.335, 8.117e-10),
                                     complex<double>(1.334, 1.742e-9),
                                     complex<double>(1.333, 2.473e-9),
                                     complex<double>(1.333, 3.532e-9),
                                     complex<double>(1.332, 1.062e-8),
                                     complex<double>(1.332, 1.410e-8),
                                     complex<double>(1.331, 1.759e-8),
                                     complex<double>(1.331, 2.406e-8),
                                     complex<double>(1.331, 3.476e-8),
                                     complex<double>(1.330, 8.591e-8),
                                     complex<double>(1.330, 1.474e-7),
                                     complex<double>(1.330, 1.486e-7)  };

  // Percent transmittance of clarified orange juice [Meydav et al. 1977].
  // Measurements are usually done using cuvettes of internal width of 1 cm.
  double orange_juice_tra[8] = { 21.2, 65.6, 87.3, 92.9, 95.4, 97.1, 98.2, 98.8 };

  // Absorbance spectrum of the carotenoid fraction in orange juice [Melendez-Martinez et al. 2011].
  // The first array range:
  //   0 = thermally treated orange juice (carotenoid content: 6.34 mg/L)
  //   1 = ultra frozen orange juice (carotenoid content: 24.08 mg/L, presumably closer to fresh orange juice)
  double carotenoids_abs[2][17] = { { 0.465, 0.713, 0.855, 0.728, 0.424, 0.0675, 0.0113, 0.00375,
                                      0.0, 0.0, 0.0, 0.0113, 0.0, 0.0, 0.0, 0.0, 0.0 },
                                    { 0.383, 0.653, 0.990, 0.975, 0.623, 0.0750, 0.0225, 0.0150,
                                      0.0195, 0.0150, 0.113, 0.0225, 0.0075, 0.0, 0.0, 0.0, 0.0 } };

  template<class T> void init_spectrum(Color<T>& c, unsigned int no_of_samples)
  {
    c.resize(no_of_samples);
    c.wavelength = 375.0;
    c.step_size = 25.0;
  }

  // Orange particle size distribution [Sentandreu et al. 2011].
  // The first array range:
  //   0 = particle sizes in micrometers
  //   1 = volume fraction in whole juice (untreated)
  double orange_vol_distrib[2][32] = { { 1.36, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                         14.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
                                         140.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 
                                         1000.0, 1400.0, 1700.0 }, 
                                       { 0.0, 0.083, 0.11, 0.17, 0.22, 0.28, 0.32, 0.35, 0.39, 0.41,
                                         0.44, 0.44, 0.46, 0.61, 0.83, 1.08, 1.28, 1.42, 1.50, 1.60,
                                         1.75, 2.38, 4.28, 6.50, 7.50, 7.78, 7.44, 6.67, 5.89,
                                         4.76, 1.39, 0.0 } };

  // Number density distribution of particles in the orange particle cloud.
  ParticleDistrib orange_cloud(double vol_frac)
  {
    ParticleDistrib distrib;
    distrib.r_min = log10(orange_vol_distrib[0][0]);
    distrib.r_max = log10(orange_vol_distrib[0][31]);
    distrib.N.resize(31);
    double volume = 0.0;
    for(unsigned int i = 0; i < distrib.N.size(); ++i)
    {
      double dr = (orange_vol_distrib[0][i + 1] - orange_vol_distrib[0][i])*0.5e-6;
      double r = orange_vol_distrib[0][i]*0.5e-6 + dr*0.5;
      distrib.N[i] = (orange_vol_distrib[1][i] + orange_vol_distrib[1][i + 1])*0.5e-2;
      distrib.N[i] *= log10(orange_vol_distrib[0][i + 1]) - log10(orange_vol_distrib[0][i]);
      volume += distrib.N[i];
      distrib.N[i] /= 4.0/3.0*M_PI*r*r*r*dr;
    }
    distrib.N *= vol_frac/volume;
    volume = 0.0;
    for(unsigned int i = 0; i < distrib.N.size(); ++i)
    {
      double dr = (orange_vol_distrib[0][i + 1] - orange_vol_distrib[0][i])*0.5e-6;
      double r = orange_vol_distrib[0][i]*0.5e-6 + dr*0.5;
      volume += distrib.N[i]*4.0/3.0*M_PI*r*r*r*dr;
    }
    return distrib;
  }

  // Modified optical_props function for integrating over the orange
  // particle size distribution.
  void optical_props_orange(ParticleDistrib* p, 
                            double wavelength, 
                            const complex<double>& host_refrac,
                            const complex<double>* particle_refrac)
  {
    if(!p)
    {
      cerr << "Error: Particle distribution p is a null pointer.";
      return;
    }
    p->ext = 0.0;
    p->sca = 0.0;
    p->abs = 0.0;
    p->g = 0.0;
    p->ior = 0.0;
    for(unsigned int i = 0; i < p->N.size(); ++i)
    {
      double dr = (orange_vol_distrib[0][i + 1] - orange_vol_distrib[0][i])*0.5e-6;
      double r = orange_vol_distrib[0][i]*0.5e-6 + dr*0.5;
      double C_t, C_s, C_a, g, ior;
      particle_props(C_t, C_s, C_a, g, ior, 
                     r, wavelength,
                     host_refrac,
                     particle_refrac ? *particle_refrac : p->refrac_idx);
      
      double sigma_s = C_s*p->N[i]*dr;
      p->ext += C_t*p->N[i]*dr;
      p->sca += sigma_s;
      p->abs += C_a*p->N[i]*dr;
      p->g += g*sigma_s;
      p->ior += ior*p->N[i]*dr;
    }
    if(p->sca > 0.0)
      p->g /= p->sca;
  }
}

Medium orange_juice()
{
  // Orange juice color is mainly due to the carotenoid contents [Rummens 1970, Melendez-Martinez et al. 2011]
  // Viscosities of fresh juice and juice host have been measured [Hernandez et al. 1992]
  // Non-Newtonian properties of orange juice have been analyzed [Mizrahi and Berk 1972]

  const double temperature = 20.0;    // degrees Celsius [10, 80]
  const double soluble_solids = 10.0; // degrees Brix [2, 14] (fresh and commercial orange juices are in this range [Hernandez et al. 1992, Cen et al. 2006]).
  const double orange_volume = 0.10;  // volume fraction [0, 0.24] (measured to be 8% by Hernandez et al. [1992], see also Mizrahi and Berk [1970])

  // Formula for orange juice density which works for temperatures in the range 10 to 80 degrees Celsius [Ramos et al. 1998].
  // In freshly squeezed and pasteurized orange juice the soluble solids concentration is ~10 degrees Brix [Hernandez et al. 1992].
  const double juice_density = (1025.42 - 0.3289*temperature + 3.2819*soluble_solids + 0.0178*soluble_solids*soluble_solids)*1.0e-3; // g/mL

  // Real part of the refractive index of the orange juice host as a function of the soluble
  // solids concentration (using two measurements from Genovese and Lozano [2006])
  double n_h_slope = (1.363 - 1.347)/(20.0 - 10.0);
  double n_h_delta = n_h_slope*soluble_solids;

  // Carotenoid concentration in juice as a whole (available from several references [Higby 1962, Cortes et al. 2006, Melendez-Martinez et al. 2011])
  //double carotenoids = 1.36717;          // mg/100 g in untreateed juice [Cortes et al. 2006]
  double carotenoids = 1.19537;            // mg/100 g in pasteurized juice [Cortes et al. 2006]
  carotenoids *= juice_density*1.0e-3;     // to get concentration in g/100 mL
  //double carotenoids = 24.0e-4;          // g/100 mL [Melendez-Martinez et al. 2011] 

  // Carotenoid concentration in particles (they exist in the particles [Melendez-Martinez et al. 2009, Melendez-Martinez et al. 2011])
  carotenoids /= orange_volume;

  // Real part of the refractive index of orange particles as a function of wavelength 
  // (using two measurements from Ray et al. [1983])
  double n_p_slope = (1.4963 - 1.5120)/(589.0 - 425.0);
  double n_p_intercept = 1.4963 - n_p_slope*589.0;
  n_p_slope *= 1.0e9; // 1/nm -> 1/m

  ParticleDistrib orange = orange_cloud(orange_volume);
  orange.refrac_idx = complex<double>(1.5, 0.0);

  const int no_of_samples = 17;
  Medium m;
  complex<double> refrac_host[no_of_samples];
  complex<double> refrac_orange[no_of_samples];
  Color< complex<double> >& ior = m.get_ior(spectrum);
  Color<double>& extinct = m.get_extinction(spectrum);
  Color<double>& scatter = m.get_scattering(spectrum);
  Color<double>& absorp = m.get_absorption(spectrum);
  Color<double>& asymmetry = m.get_asymmetry(spectrum);

  // Initialize medium
  init_spectrum(ior, no_of_samples);
  init_spectrum(extinct, no_of_samples);
  init_spectrum(scatter, no_of_samples);
  init_spectrum(absorp, no_of_samples);
  init_spectrum(asymmetry, no_of_samples);

  for(unsigned int i = 0; i < no_of_samples; ++i)
  {
    double wavelength = i*25.0e-9 + 375.0e-9;
    
    // Adding soluble solids and host absorption as measured in clarified orange juice [Meydav et al. 1977].
    // Blending absorption with water absorption as we get away from the absorption peak 
    // (the curve is too limited in precision when values get close to zero).
    unsigned int abs_idx = i > 7 ? 7 : i;
    double host_abs = -log10(orange_juice_tra[abs_idx]*1.0e-2)/1.0e-2*log(10.0);
    double n_imag = host_abs*wavelength/(4.0*M_PI);
    double x = (i - abs_idx)/(no_of_samples - 8.0);
    refrac_host[i] = complex<double>(refrac_water[i].real() + n_h_delta, n_imag*(1.0 - x) + refrac_water[i].imag()*x);

    // Absorption coefficient (1/m) of a 1 g/100 mL solution of carotenoids [Melendez-Martinez et al. 2011].
    double abs_coef = (carotenoids_abs[0][i]/6.34e-4)/1.0e-2*log(10.0);

    // Adding absorption of carotenoids to particle refractive index.
    abs_coef *= carotenoids;
    refrac_orange[i] = complex<double>(n_p_slope*wavelength + n_p_intercept, abs_coef*wavelength/(4.0*M_PI));
    
    //cerr << "Wavelength: " << wavelength << " (idx: " << i << ")" << endl;
    //cerr << "Refractive Index of host medium: " << refrac_host[i] << endl;
    //cerr << "Refractive index of orange particles: " << refrac_orange[i] << endl;
    
    optical_props_orange(&orange, wavelength, refrac_host[i], &refrac_orange[i]);
    //cerr << "Orange properties: " << orange.ext << " " << orange.sca << " " << orange.abs << " " << orange.g << endl;
    
    double host_absorp = 4.0*M_PI*refrac_host[i].imag()/wavelength;
    //cerr << "Host extinction: " << host_absorp << endl;

    extinct[i] = host_absorp + orange.ext;
    scatter[i] = orange.sca;
    absorp[i] = extinct[i] - scatter[i];
    ior[i] = complex<double>(refrac_host[i].real(), absorp[i]*wavelength/(4.0*M_PI));
    if(scatter[i] != 0.0)
      asymmetry[i] = (orange.sca*orange.g)/scatter[i];
    
    //cout << "Extinction coefficient (" << wavelength*1e9 << "nm): " << extinct[i] << endl
    //     << "Scattering coefficient (" << wavelength*1e9 << "nm): " << scatter[i] << endl
    //     << "Absorption coefficient (" << wavelength*1e9 << "nm): " << absorp[i] << endl
    //     << "Ensemble asymmetry parameter (" << wavelength*1e9 << "nm):    " << asymmetry[i] << endl;

    //if(i < 16)
    //{
    //  cout << endl << "<press enter for next set of values>" << endl;
    //  cin.get();
    //}
  }
  m.name = "OrangeJuice";
  m.turbid = true;
  return m;
}
