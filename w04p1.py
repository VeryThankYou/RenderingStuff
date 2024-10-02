import numpy as np

"""
Part 1
A small 25Wlight bulb has an efficiency of 20%. How many photons are approximately emitted per second?
Assume in the calculations that we only use average photons of wavelength 500 nm.
"""

l = 500 * 10**(-9)
effect = 25
efficiency = 0.2
h = 6.626 * 10**(-34)
c = 2.9979 * 10**(8)
energy_per_photon = h*c/l

num_photons = effect * efficiency / energy_per_photon
print(num_photons)
# 1.2585527849170446e+19

"""
Part 2
A light bulb (2.4 V and 0.7 A), which is approximately sphere-shaped with a diameter of 1 cm, emits light
equally in all directions. Find the following entities (ideal conditions assumed)
"""
voltage = 2.4
current = 0.7
d = 0.01
# Radiant Flux
# We assume efficiency = 1
Phi = voltage * current
print(f"Radiant flux = {Phi}")

# Radiant Intensity
I = Phi/(4 * np.pi)
print(f"Radiant Intensity = {I}")

# Radiant Exitance
A = 4 * np.pi * (d/2)**2
M = Phi/A
print(f"Radiant Exitance = {M}")

# Emitted energy in 5 minutes
t = 60*5
E = Phi*t
print(f"Emitted Energy in 5 minutes = {E}")

"""
Part 3
The light bulb from above is observed by an eye, which has an opening of the pupil of 6 mm and a distance
of 1 m from the light bulb. Find the irradiance received in the eye.
"""
dist = 1
pupulsize = 6e-3
pupil_A = (pupulsize/2)**2 * np.pi
#A_big_light = 4 * ((d/2) + dist)**2 * np.pi
#ratio  = pupil_A/A_big_light
#P_eye = ratio * Phi
#Irradiance = P_eye/pupil_A
Irradiance = I / dist **2

print(f"The irradiance received in the eye = {Irradiance}")

"""
Part 4
A 200 W spherically shaped light bulb (20% efficiency) emits red light of wavelength 650 nm equally in all
directions. The light bulb is placed 2 m above a table. Calculate the irradiance at the table.
Photometric quantities can be calculated from radiometric ones based on the equation
Photometric = Radiometric · 685 · V (λ)
in which V (λ) is the luminous efficiency curve.
At 650 nm, the luminous efficiency curve has a value of 0.1. Calculate the illuminance.
"""

effect = 200
efficiency = 0.2
l = 650e-9
dist = 2
Vl = 0.1
Phi = effect
I = Phi/(4 * np.pi)
Irradiance = I / dist **2
print(f"The irradiance received by the table = {Irradiance}")
Illuminance = Irradiance * 685 * Vl
print(f"The illuminance received by the table = {Illuminance}")

"""
Part 5
In a simple arrangement the luminous intensity of an unknown light source is determined from a known
light source. The light sources are placed 1 m from each other and illuminate a double sided screen placed
between the light sources. The screen is moved until both sides are equally illuminated as observed by a
photometer. At the position of match, the screen is 35 cm from the known source with luminous intensity
Is = 40 lm/sr = 40 cd and 65 cm from the unknown light source. What is the luminous intensity Ix of the
unknown source?
"""
li_1 = 40
d_1 = 0.35
d_2 = 1 - d_1
Irradiance = li_1 / d_1**2
li_2 = Irradiance * d_2**2

print(f"The luminous intensity of the unknown light source is = {li_2}")

"""
Part 6
The radiance L from a diffuse light source (emitter) of 10×10 cm is 5000 W/(sr m2). Calculate the radiosity
(radiant exitance). How much energy is emitted from the light source?
"""
L = 5000
B = L * np.pi
print(f"The diffuse radiosity of the light source is {B}")
l = 0.1
A = l*l
Phi = B*A
print(f"The power of the light source is {Phi}")

"""
Part 7
The radiance L = 6000 cos θ W/(m2 sr) for a non-diffuse emitter of area 10 by 10 cm. Find the radiant
exitance. Also, find the power of the entire light source.
"""
import sympy as sp
theta = sp.Symbol("theta")
phi = sp.Symbol("phi")
L = 6000 * sp.cos(theta)
M = sp.integrate(sp.integrate(L*sp.cos(theta)*sp.sin(theta), (theta, 0, sp.pi/2)), (phi, 0, 2*sp.pi))
print(f"The radiant exitance of the light source is {M.evalf()}")

l = 0.1
A = l*l
Phi = M.evalf() * A
print(f"The power of the light source is {Phi}")