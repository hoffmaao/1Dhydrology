import firedrake
from constants import *
from hs_solver import *
from phi_solver import *



class Glads1DModel():

  def __init__(self, model_inputs, in_dir = None):

    ### Initialize model inputs

    self.mesh = model_inputs['mesh']
    self.V_cg = firedrake.FunctionSpace(self.mesh, "CG", 2)

    self.model_inputs = model_inputs
    
    # If an input directory is specified, load model inputs from there. 
    # Otherwise use the specified model inputs dictionary.
    if in_dir:
      model_inputs = self.load_inputs(in_dir)
      
    # Ice thickness    
    self.H = self.model_inputs['H']
    # Bed elevation
    self.B = self.model_inputs['B']
    # Basal sliding speed
    self.u_b = self.model_inputs['u_b']
    # Melt rate
    self.m = self.model_inputs['m']
    # Cavity gap height
    self.h = self.model_inputs['h_init']
    # Potential
    self.phi_prev = self.model_inputs['phi_init']
    # Potential at 0 pressure
    self.phi_m = self.model_inputs['phi_m']
    # Ice overburden pressure
    self.p_i = self.model_inputs['p_i']
    # Potential at overburden pressure
    self.phi_0 = self.model_inputs['phi_0']
    # Dirichlet boundary conditions
    self.d_bcs = self.model_inputs['d_bcs']
    # Channel areas
    self.S = self.model_inputs['S_init']
    # Output directory
    self.out_dir = self.model_inputs['out_dir']
    # Directory storing maps that are used to deal with CR functions in parallel
    self.maps_dir = self.model_inputs['maps_dir']

    
    # If there is a dictionary of physical constants specified, use it. 
    # Otherwise use the defaults. 
    if 'constants' in self.model_inputs :
      self.pcs = self.model_inputs['constants']
    else :
      self.pcs = pcs
    
    self.n_bcs = []
    if 'n_bcs' in self.model_inputs:
      self.n_bcs = model_inputs['n_bcs']

    ### Create some fields

    self.V_cg = firedrake.FunctionSpace(self.mesh, "CG", 1)


    # Potential
    self.phi = firedrake.Function(self.V_cg)
    # Effective pressure
    self.N = firedrake.Function(self.V_cg)
    # Stores the value of S**alpha. 
    self.S_alpha = firedrake.Function(self.V_cg)
    self.update_S_alpha()
    # Water pressure
    self.p_w = firedrake.Function(self.V_cg)
    # Pressure as a fraction of overburden
    self.pfo = firedrake.Function(self.V_cg)
    # Current time
    self.t = 0.0
    

    ### Output files
    #self.ff_out = firedrake.FacetFunctionDouble(self.mesh)
    self.S_out = firedrake.File(self.out_dir + "S.pvd")
    self.h_out = firedrake.File(self.out_dir + "h.pvd")
    self.phi_out = firedrake.File(self.out_dir + "phi.pvd")
    self.pfo_out = firedrake.File(self.out_dir + "pfo.pvd")


    ### Create the solver objects
    # Potential solver    
    self.phi_solver = PhiSolver(self)
    # Gap height solver
    self.hs_solver = HSSolver(self)
    

  # Steps the potential, gap height, and water height forward by dt  
  def step(self, dt):
    # Step the potential forward by dt with h fixed
    self.phi_solver.step(dt)
    # Step h forward by dt with phi fixed
    self.hs_solver.step(dt)
    
    
  # Load all model inputs from a directory except for the mesh and initial 
  # conditions on h, h_w, and phi
  def load_inputs(self, in_dir):
    # Bed
    B = firedrake.Function(self.V_cg)
    firedrake.File(in_dir + "B.pvd") >> B
    # Ice thickness
    H = firedrake.Function(self.V_cg)
    firedrake.File(in_dir + "H.pvd") >> H
    # Melt
    m = firedrake.Function(self.V_cg)
    firedrake.File(in_dir + "m.pvd") >> m
    # Sliding speed
    u_b = firedrake.Function(self.V_cg)
    firedrake.File(in_dir + "u_b.pvd") >> u_b
    # Potential at 0 pressure
    phi_m = firedrake.Function(self.V_cg)
    firedrake.File(in_dir + "phi_m.pvd") >> phi_m
    # Potential at overburden pressure
    phi_0 = firedrake.Function(self.V_cg)
    firedrake.File(in_dir + "phi_0.pvd") >> phi_0
    # Ice overburden pressure
    p_i = firedrake.Function(self.V_cg)
    firedrake.File(in_dir + "p_i.pvd") >> p_i
   
    self.model_inputs['B'] = B
    self.model_inputs['H'] = H
    self.model_inputs['m'] = m
    self.model_inputs['u_b'] = u_b
    self.model_inputs['phi_m'] = phi_m
    self.model_inputs['phi_0'] = phi_0
    self.model_inputs['p_i'] = p_i
     
  
  # Update the effective pressure to reflect current value of phi
  def update_N(self):
    self.phi=firedrake.interpolate(self.phi,self.V_cg)
    self.phi_0=firedrake.interpolate(self.phi_0,self.V_cg)

    self.N.vector().set_local(self.phi_0.vector().array() - self.phi.vector().array())
    self.N.vector().apply("insert")
    
  
  # Update the water pressure to reflect current value of phi
  def update_pw(self):
    self.p_w=firedrake.interpolate(self.p_w,self.V_cg)
    self.phi=firedrake.interpolate(self.phi,self.V_cg)
    self.phi_m=firedrake.interpolate(self.phi_m,self.V_cg)



    self.p_w.vector().set_local(self.phi.vector().array() - self.phi_m.vector().array())
    self.p_w.vector().apply("insert")
    
  
  # Update the pressure as a fraction of overburden to reflect the current 
  # value of phi
  def update_pfo(self):
    # Update water pressure
    self.update_pw()
  
    # Compute overburden pressure
    self.pfo=firedrake.interpolate(self.pfo,self.V_cg)
    self.p_w=firedrake.interpolate(self.p_w,self.V_cg)
    self.p_i=firedrake.interpolate(self.p_i,self.V_cg)
    
    self.pfo.vector().set_local(self.p_w.vector().array() / self.p_i.vector().array())
    self.pfo.vector().apply("insert")
    
  
  # Updates all fields derived from phi
  def update_phi(self):
    #phi_tmp=firedrake.interpolate(self.phi,self.V_cg)
    self.phi_prev=firedrake.interpolate(self.phi_prev,self.V_cg)
    phi_tmp=firedrake.interpolate(self.phi,self.V_cg)
    self.phi_prev.assign(phi_tmp)
    self.update_N()
    self.update_pfo()
  
  
  # Update S**alpha to reflect current value of S
  def update_S_alpha(self):
    alpha = self.pcs['alpha']
    self.S_alpha.vector().set_local(self.S.vector().array()**alpha)
    
  
  # Write h, S, pfo, and phi to pvd files
  def write_pvds(self):
    self.S=firedrake.interpolate(self.S,self.V_cg)
    self.h=firedrake.interpolate(self.h,self.V_cg)
    self.S_out << self.S
    self.h_out << self.h
    self.phi_out << self.phi
    self.pfo_out << self.pfo

