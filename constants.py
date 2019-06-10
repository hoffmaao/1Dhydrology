# Copyright (C) 2019 by Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of Andrew's 1D hydrology model
#
# The full text of the license can be found in the file LICENSE in the
# source directory or at <http://www.gnu.org/licenses/>.

r"""Physical constants

This module constains physical constants used throughout the library such
as the acceleration due to gravity, the density, of ice and water, etc.

"""
pcs={}
# Seconds per day
pcs["day"] = 60.0 * 60.0 * 24.0
# Seconds per year
pcs["year"] = 60.0 * 60.0 * 24.0 * 365.0
# Density of water 
pcs["rho_water"] = 1000.0 # kg / m^3
# Density of ice 
pcs["rho_ice"] = 910.0 # kg / m^3
# Gravitational acceleration 
pcs["g"] = 9.8 # m / s^2
# Flow rate factor of ice
pcs["A"] = 2.25e-25 # 1 / Pa^3 * s
# Average bump height
pcs["h_r"] = 0.1 # m
# Typical spacing between bumps
pcs["l_r"] = 2.0 # m
# Sheet width under channel
pcs["l_c"] = 2.0 # m
# Sheet conductivity
pcs["k"] = 0.005 # m^(7/4) / kg^(1/2)
# Channel conductivity 
pcs["k_c"] = 0.195 # m^(3/2) / kg^(1/2)
# Specific heat capacity of ice 
pcs["c_w"] = 4.22e3 # J / (kg * K)
# Pressure melting coefficient 
pcs["c_t"] = 7.5e-8 # J / (kg * K)
# Latent heat 
pcs["L"] = 3.34e5 # J / kg
# Void storage ratio
pcs["e_v"] = 1e-3
# Exponents
pcs["alpha"] = 5.0 / 4.0
pcs["beta"] = 3.0 / 2.0
pcs["delta" = pcs["beta"] - 2.0
