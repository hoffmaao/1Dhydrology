import firedrake
from scipy.integrate import ode

"""Solves for the sheet height h."""

class HSolver():

  def __init__(self, model):
    
    ### Get a few fields and parameters from the model
    
    # Basal sliding speed
    u_b = model.u_b
    # Cavity gap height
    h = model.h
    # Effective pressure
    N = model.N   
    # Initial model time
    t0 = model.t
    # Rate factor
    A = model.constants['A']
    # Distance between bumps
    l_r = model.constants['l_r']
    # Bump height
    h_r = model.constants['h_r']
    

    ### Set up the gap height ODE

    # Get some of the static fields as arrays
    u_b_n = u_b.vector().array()
    h0 = h.vector().array()
    
    # Right hand side for the gap height ODE
    def rhs(t, h_n) :
      # Ensure that the sheet height is positive
      h_n[h_n < 0.0] = 0.0
      # Sheet opening term
      w_n = u_b_n * (h_r - h_n) / l_r
      # Ensure that the opening term is non-negative
      w_n[w_n < 0.0] = 0.0
      # Sheet closure term
      v_n = A * h_n * N.vector().array()**3
      # Return the time rate of change of the sheet
      dhdt = w_n - v_n
      return dhdt
      
    # Set up integrator
    ode_solver = ode(rhs).set_integrator('vode', method = 'adams', max_step = 60.0 * 5.0)
    ode_solver.set_initial_value(h0, t0)


    ### Set local variables    
    
    self.ode_solver = ode_solver
    self.model = model
    

  # Step the gap height h forward by dt
  def solve(self, dt):
    # Step the gap height ODE forward
    self.ode_solver.integrate(self.model.t + dt)
    # Update the gap height function
    self.model.h.vector()[:] = self.ode_solver.y
    # Update the model time
    self.model.t = self.ode_solver.t