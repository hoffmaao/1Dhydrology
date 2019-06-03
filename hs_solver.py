import firedrake
from scipy.integrate import ode
import numpy as np

"""Solves ODEs for the sheet height h and channel area S."""

class HSSolver():

  def __init__(self, model):
    
    ### Get a few fields and parameters from the model
    
    # Effective pressure
    N = firedrake.interpolate(model.N,model.V_cg)
    # hydropotential
    phi = firedrake.interpolate(model.phi,model.V_cg) 
    # Cavity gap height
    h = firedrake.interpolate(model.h,model.V_cg)
    # Channel height
    S = firedrake.interpolate(model.S,model.V_cg)
    # Initial model time
    t0 = model.t
    # Rate factor
    A = model.pcs['A']
    # Distance between bumps
    l_r = model.pcs['l_r']
    # Bump height
    h_r = model.pcs['h_r']
    # Density of ice
    rho_i = model.pcs['rho_ice']
    # Latent heat
    L = model.pcs['L']
    # Sheet conductivity
    k = model.pcs['k']
    # Channel conductivity
    k_c = model.pcs['k_c']
    # Sheet width under channel
    l_c = model.pcs['l_c']
    # Exponent
    alpha = model.pcs['alpha']
    delta = model.pcs['delta']
    # Regularization parameter
    phi_reg = firedrake.Constant(1e-16)   
    
    
    ### Static arrays used in the ODE rhs
    
    # Vector for sliding speed
    u_b_n = firedrake.interpolate(model.u_b,model.V_cg)
    u_b_n = u_b_n.vector().array()
    h0 = firedrake.interpolate(model.h,model.V_cg)#model.h.vector().array()#firedrake.interpolate(model.h,model.V_cg)
    h0 = h0.vector().array()
    # Initial channel areas
    S0 = firedrake.interpolate(model.S,model.V_cg)
    S0 = S0.vector().array()
    # Length of h vector
    h_len = len(h0)

    ### Set up the sheet height and channel area ODEs
    
    # Right hand side for the gap height ODE
    def h_rhs(t, h_n) :
      # Ensure that the sheet height is positive
      h_n[h_n < 0.0] = firedrake.Constant(0.0)
      # Get effective pressures
      N_n = N.vector().array()
      # Sheet opening term
      w_n = u_b_n * (h_r - h_n) / l_r
      # Ensure that the opening term is non-negative
      w_n[w_n < 0.0] = firedrake.Constant(0.0)
      # Sheet closure term
      v_n = firedrake.Constant(A) * h_n * N_n**firedrake.Constant(3.0)

      # Return the time rate of change of the sheet
      dhdt = w_n - v_n
      return dhdt
      
    # Right hand side for the channel area ODE
    def S_rhs(t, S_n):
      # Ensure that the channel area is positive
      S_n[S_n < 0.0] = firedrake.Constant(0.0)
      # Get effective pressures
      N_n = N.vector().array()
      # Get midpoint values of sheet thickness
      h_n = h.vector().array()
      # Array form of the derivative of the potential 
      phi_grad = model.phi.dx(0)
      phi_s = firedrake.interpolate(phi_grad,model.V_cg)
      #phi_s = phi.dx(0) # This is one obvious problem (How do we define derivatives of arrays?)
      
      # Along channel flux
      Q_n = -firedrake.Constant(k_c) * S_n**firedrake.Constant(alpha) * np.abs(phi_s.dat.data + phi_reg)**firedrake.Constant(delta) * phi_s.dat.data
      # Flux of sheet under channel
      q_n = -firedrake.Constant(k) * h_n**firedrake.Constant(alpha) * np.abs(phi_s.dat.data + phi_reg)**firedrake.Constant(delta) * phi_s.dat.data
      # Dissipation melting due to turbulent flux
      # Creep closure
      Xi_n = abs(Q_n * phi_s.dat.data) + np.abs(l_c * q_n * phi_s.dat.data)
      v_c_n = firedrake.Constant(A) * S_n * N_n**3
      # Total opening rate
      v_o_n = Xi_n / (rho_i * L) # firedrake.conditional(firedrake.And(firedrake.lt(Xi_n / (rho_i * L),0.0),firedrake.eq(S_n,0.0)),0.0,Xi_n / (rho_i * L))
      # Disallow negative opening rate where the channel area is 0
      dsdt = (v_o_n - v_c_n)
      return dsdt
      
    # Combined right hand side for h and S
    def rhs(t, Y):
      Ys = np.split(Y, [h_len])
      h_n = Ys[0]
      S_n = Ys[1]
      
      dsdt = S_rhs(t, S_n)
      dhdt = h_rhs(t, h_n)
      
      return np.hstack((dhdt, dsdt))
    
    # ODE solver initial condition
    Y0 = np.hstack((h0, S0))
    # Set up ODE solver
    ode_solver = ode(rhs).set_integrator('vode', method = 'adams', max_step = 60.0 * 5.0)
    ode_solver.set_initial_value(Y0, t0)


    ### Set local variables
    self.Y0=Y0    
    self.rhs=rhs
    self.S_rhs= S_rhs
    self.ode_solver = ode_solver
    self.model = model
    self.h_len = h_len
    
  def step(self, dt):
    # Step h and S forward   
    self.ode_solver.integrate(self.model.t + dt)

    # Retrieve values from the ODE solver    
    Y = np.split(self.ode_solver.y, [self.h_len])
    self.model.h.vector().set_local(Y[0])
    self.model.h.vector().apply("insert")
    self.model.S.vector().set_local(Y[1])
    self.model.S.vector().apply("insert")
    
    # Update S**alpha
    self.model.update_S_alpha()
  
    # Update the model time
    self.model.t = self.ode_solver.t