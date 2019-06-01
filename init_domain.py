import numpy as np
from dolfin import *
from constants import *
from model import 




########### Domain Geometry ############
mesh_dir = '/Volumes/hoffmaao/data/rd09/projects/thwaites_modeling/1Dhydrology'
xmin = 0.0
xmax = 10000.0
dx = 100.0
mesh = InvervalMesh(xmax,xmin,dx)
x = SpatialCoordinate(mesh)
b0,db = 0,100

V_cg = FunctionSapce(mesh,"CG",1)

# bed and surface topography
H=Function(V_cg)
H.interpolate(100*sqrt(x))


B=Function(V_cg)
B.interpolate(b0-db*x/xmax)

B_out = File(mesh_dir + "S.pvd")
H_out = File(mesh_dir + "h.pvd")

