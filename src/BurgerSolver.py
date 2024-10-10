import numpy as np

class BurgerSolver:
    def __init__(self, nx=100, L=2*np.pi, dt=0.001, T=1.0):
        """
        Initialize the solver with spatial grid, time step, and total simulation time.
        """
        self.nx = nx  # Number of grid points
        self.L = L    # Domain length
        self.dx = L / nx  # Spatial step size
        self.dt = dt  # Time step size
        self.T = T    # Total simulation time
        self.nt = int(T / dt)  # Number of time steps
        self.x = np.linspace(0, L, nx, endpoint=False)  # Spatial grid

        # Initialize velocity field u
        self.u = np.sin(self.x)

    def lax_friedrichs(self):
        """
        Lax-Friedrichs scheme to update the velocity field u.
        """
        u_new = np.zeros_like(self.u)
        u_new[1:-1] = 0.5 * (self.u[2:] + self.u[:-2]) - 0.5 * self.dt / self.dx * (self.u[2:]**2 / 2 - self.u[:-2]**2 / 2)
        u_new[0] = 0.5 * (self.u[1] + self.u[-1]) - 0.5 * self.dt / self.dx * (self.u[1]**2 / 2 - self.u[-1]**2 / 2)
        u_new[-1] = u_new[0]  # Periodic boundary condition
        self.u = u_new

    def solve(self):
        """
        Solve the Burgers equation using the Lax-Friedrichs scheme.
        """
        for _ in range(self.nt):
            self.lax_friedrichs()

        return self.u

    def get_solution(self):
        """
        Return the final velocity field.
        """
        return self.u
