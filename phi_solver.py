import firedrake


""" Solves phi with h and S fixed."""


class PhiSolver(object):
    def __init__(self, model):

        # melt rate
        m = model.m
        # Sheet height
        h = model.h
        # Channel area
        S = model.S
        # This function stores the value of S**alpha. 
        S_alpha = model.S_alpha
        # hydropotential at zero bed elvation
        phi_m = model.phi_m
        # Basal sliding speed
        u_b = model.u_b
        # Potential
        phi = model.phi
        # Potential at previous time step
        phi_prev = model.phi_prev
        # Potential at overburden pressure
        phi_0 = model.phi_0
        # Representative width of water system
        width = model.width
        # Density of ice
        rho_ice = model.pcs["rho_ice"]
        # Density of water
        rho_water = model.pcs["rho_water"]
        # Rate factor
        A = model.pcs["A"]
        # Sheet conductivity
        k = model.pcs["k"]
        # Channel conductivity
        k_c = model.pcs["k_c"]
        # Bump height
        h_r = model.pcs["h_r"]
        # Distance between bumps
        l_r = model.pcs["l_r"]
        # Sheet width under channel
        l_c = model.pcs["l_c"]
        # Clapeyron slope
        c_t = model.pcs["c_t"]
        # Specific heat capacity of water
        c_w = model.pcs["c_w"]
        # Latent heat
        L = model.pcs["L"]
        # Void storage ratio
        e_v = model.pcs["e_v"]
        # Gravitational acceleration
        g = model.pcs["g"]

        # Exponents
        alpha = model.pcs["alpha"]
        delta = model.pcs["delta"]
        # pcs in front of storage term
        c1 = e_v / (rho_water * g)
        # Regularization parameter
        phi_reg = firedrake.Constant(1e-15)

        ### Set up the sheet model

        # Expression for effective pressure in terms of potential
        N = phi_0 - phi
        # Derivative of phi
        dphi_tmp = phi.dx(0)

        dphi_ds = firedrake.interpolate(dphi_tmp, model.V_cg)
        # Flux vector
        q_s = (
            -firedrake.Constant(k)
            * h ** alpha
            * abs(phi.dx(0) + phi_reg) ** (delta)
            * phi.dx(0)
        )
        # Opening term
        w = firedrake.conditional(
            firedrake.gt(h_r - h, 0.0), u_b * (h_r - h) / l_r, 0.0
        )
        # Closing term
        v = firedrake.Constant(A) * h * N ** 3
        # Time step
        dt = firedrake.Constant(1.0)

        ### Set up the channel model

        # Discharge through channels
        Q_c = (
            -firedrake.Constant(k_c)
            * S_alpha
            * abs(phi.dx(0) + firedrake.Constant(phi_reg)) ** delta
            * phi.dx(0)
        )
        # Approximate discharge of sheet in direction of channel
        q_c = (
            -firedrake.Constant(k)
            * h ** alpha
            * abs(phi.dx(0) + firedrake.Constant(phi_reg)) ** delta
            * phi.dx(0)
        )* l_c

        # Energy dissipation
        Xi = abs(Q_c * phi.dx(0)) + abs(firedrake.Constant(l_c) * q_c * dphi_ds)

        # pressure melting
        pw = phi - phi_m
        pw_s = pw.dx(0)
        pw_s = firedrake.interpolate(pw_s, model.V_cg)
        f = firedrake.conditional(firedrake.Or(S>0.0,q_c*pw.dx(0)> 0.0),1.0,0.0)
        II_n = -c_t * c_w * rho_water * 0.3 * (Q_c + f * l_c * q_c) * pw_s
        # Total opening rate (dissapation of potential energy and pressure melting)
        w_c = ((Xi - II_n) / firedrake.Constant(L)) * firedrake.Constant(
            (1.0 / rho_ice) - (1.0 / rho_water))

        # closing term assocaited with creep closure
        v_c = firedrake.Constant(A) * S * N ** firedrake.Constant(3.0)
        
        ### Set up the PDE for the potential ###
        theta = firedrake.TestFunction(model.V_cg)

        # Constant in front of storage term
        C1 = firedrake.Constant(c1) * width
        # Storage term
        F_s = C1 * (phi - phi_prev) * theta * firedrake.dx
        
        tmp=firedrake.assemble(width)
        # Sheet contribution to PDE
        F_s += (
            dt
            * (-theta.dx(0) * q_s * width + (w - v - m) * width * theta)
            * firedrake.dx
        )
        # Add any non-zero Neumann boundary conditions
        for (m, c) in model.n_bcs:
            F_s += dt * firedrake.Constant(c) * theta * m

        # Channel contribution to PDE
        F_c = dt * ((-theta.dx(0)) * Q_c + (w_c - v_c) * theta("+")) * firedrake.dx
        # Variational form
        F = F_s + F_c
        # Get the Jacobian
        dphi = firedrake.TrialFunction(model.V_cg)
        J = firedrake.derivative(F, phi, dphi)

        ### Assign local variables

        self.F = F
        self.tmp = tmp
        self.J = J
        self.model = model
        self.dt = dt

    # Steps the potential forward by dt. Returns true if the  converged or false if it
    # had to use a smaller relaxation parameter.
    def step(self, dt):

        self.dt.assign(dt)
        tmp = firedrake.assemble(self.tmp)
        try:

            # Solve for potential
            firedrake.solve(
                self.F == 0,
                self.model.phi,
                self.model.d_bcs,
                J=self.J,
                solver_parameters={
                    "snes_monitor": None,
                    "snes_view": None,
                    "ksp_monitor_true_residual": None,
                    "snes_converged_reason": None,
                    "ksp_converged_reason": None,
                },
            )  # , solver_parameters = self.model.newton_params)

            # Derive values from the new potential
            self.model.update_phi()
        except:

            # Try the solve again with a lower relaxation param
            firedrake.solve(
                self.F == 0,
                self.model.phi,
                self.model.d_bcs,
                J=self.J,
                solver_parameters={
                    "snes_type": "newtonls",
                    "snes_rtol": 5e-11,
                    "snes_atol": 5e-10,
                    "pc_type": "lu",
                    "snes_max_it": 50,
                    "mat_type": "aij",
                },
            )  # , solver_parameters = self.model.newton_params)

            # Derive values from potential
            self.model.update_phi()

            # Didn't converge with standard params
            return False

        # Did converge with standard params
        return True
