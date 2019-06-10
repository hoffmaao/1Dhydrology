import firedrake
import numpy as np
from constants import *
from hs_solver import *
from phi_solver import *
from model import *


########### Domain Geometry ############
proj_dir = "/Volumes/hoffmaao/data/rd06/projects/thwiates_modeling/1Dhydrology/"
Lx = 50e3
nx = 48
b0 = 0
b1 = 100
mesh = firedrake.IntervalMesh(nx, Lx)

degree = 1

V_cg = firedrake.FunctionSpace(mesh, "CG", degree)

# bed/surface topography, ice velocity and melt rates

H = firedrake.Function(V_cg)
B = firedrake.Function(V_cg)

x = firedrake.SpatialCoordinate(mesh)

H = firedrake.interpolate(100 * firedrake.sqrt(x[0]), V_cg)
B = firedrake.interpolate(firedrake.Constant(b0) - (b0 - b1) * x[0] / Lx, V_cg)

ub = firedrake.Function(V_cg)
m = firedrake.Function(V_cg)

ub = firedrake.interpolate(firedrake.Constant(100.0 / pcs["spy"]), V_cg)
m = firedrake.interpolate(firedrake.Constant(0.01 / pcs["spy"]), V_cg)


H_out = firedrake.File(proj_dir + "inputs/H.pvd")
B_out = firedrake.File(proj_dir + "inputs/B.pvd")

ub_out = firedrake.File(proj_dir + "inputs/ub.pvd")
m_out = firedrake.File(proj_dir + "inputs/m.pvd")

H_out.write(H)
B_out.write(B)
ub_out.write(ub)
m_out.write(m)


########### Model Initializtion ############


h_init = firedrake.Function(V_cg)
h_init = firedrake.interpolate(firedrake.Constant(0.05), V_cg)

S_init = firedrake.Function(V_cg)
phi_init = firedrake.Function(V_cg)


phi_init = pcs["g"] * H * pcs["rho_ice"]

# firedrake.File(in_dir +"phi_0.xml") >> phi_init


# Load the boundary facet function
# boundaries = firedrake.FacetFunction('size_t', mesh)
# firedrake.File(proj_dir + "inputs/boundaries.xml") >> boundaries

# Load potential at 0 pressure
phi_m = firedrake.Function(V_cg)
phi_m = B * pcs["g"] * pcs["rho_water"]

# ice overburden pressure
p_i = firedrake.Function(V_cg)
p_i = H * pcs["g"] * pcs["rho_ice"]
# Enforce 0 pressure bc at margin
bc = firedrake.DirichletBC(V_cg, phi_m, "on_boundary")
phi_init = firedrake.Function(V_cg)
phi_init = phi_m + p_i + 0.001


pcs["k"] = 5e-4
pcs["k_c"] = 0.05


model_inputs = {}
model_inputs["phi_m"] = phi_m
model_inputs["p_i"] = H * pcs["g"] * pcs["rho_ice"]
model_inputs["phi_0"] = B * pcs["g"] * pcs["rho_water"]
model_inputs["mesh"] = mesh
model_inputs["H"] = H
model_inputs["B"] = B
model_inputs["u_b"] = ub
model_inputs["m"] = m
model_inputs["h_init"] = h_init
model_inputs["S_init"] = S_init
model_inputs["phi_init"] = phi_init
model_inputs["d_bcs"] = [bc]
model_inputs["maps_dir"] = proj_dir + "maps/"
model_inputs["out_dir"] = proj_dir + "outputs/"
model_inputs["constants"] = pcs
model_inputs["n_bc"] = []


model = Glads1DModel(model_inputs)

T = 10.0 * pcs["spd"]
dt = 60.0 * 30.0  # 30 minute timesteps
i = 0


############# Run simulation ################

while model.t < T:
    model.step(dt)
    if i % 24 == 0:
        model.write_pvds()
