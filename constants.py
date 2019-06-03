# Copyright (C) 2018-2019 Andrew Hoffman hoffmaao@uw.edu
# Constants for hydrology model

### Create a dictionary of physical constants
pcs = {}

# Seconds per day
pcs['spd'] = 60.0 * 60.0 * 24.0
# Seconds in a year
pcs['spy'] = 60.0 * 60.0 * 24.0 * 365.0                    
# Density of water (kg / m^3)
pcs['rho_water'] = 1000.0  
# Density of ice (kg / m^3)
pcs['rho_ice'] = 910.0
# Gravitational acceleration (m / s^2)
pcs['g'] = 9.81 
# Flow rate factor of ice (1 / Pa^3 * s) 
pcs['A'] = 5.0e-25
# Average bump height (m)
pcs['h_r'] = 0.1
# Typical spacing between bumps (m)
pcs['l_r'] = 8.0
# Sheet width under channel (m)
pcs['l_c'] = 2.0          
# Sheet conductivity (m^(7/4) / kg^(1/2))
pcs['k'] = 1e-2
# Channel conductivity (m^(7/4) / kg^(1/2))
pcs['k_c'] = 1e-1
# Specific heat capacity of ice (J / (kg * K))
pcs['c_w'] = 4.22e3
# Pressure melting coefficient (J / (kg * K))
pcs['c_t'] = 7.5e-8
# Latent heat (J / kg)
pcs['L'] = 3.34e5
# Void storage ratio
pcs['e_v'] = 1e-3
# Exponents 
pcs['alpha'] = 5. / 4.
pcs['beta'] = 3. / 2.
pcs['delta'] = pcs['beta'] - 2.0