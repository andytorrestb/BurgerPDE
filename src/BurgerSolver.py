import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

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
        self.x = np.linspace(-0.5*L, 0.5*L, nx, endpoint=False)  # Spatial grid

        # Initialize velocity field u with the piecewise ramp function
        self.u = self.piecewise_ramp(self.x, a=1)
        self.plot_initial_u('frames/frame_0000.png')  # Save the initial condition as frame 0000

    def piecewise_ramp(self, x, a):
        """
        Define a piecewise ramp function for the initial velocity field.
        The function is split into 3 segments:
        1. u(x) = 0 for x < L/3
        2. u(x) increases linearly from 0 to 1 between L/3 <= x < 2L/3
        3. u(x) = 1 for x >= 2L/3
        """

        # Define conditions for the piecewise function
        conditions = [
            (x < 0),  # First third of the domain
            (x >= 0) & (x < 1/a),  # Middle third of the domain
            (x >= 1/a)  # Last third of the domain
        ]

        # Define the corresponding functions for each condition
        functions = [
            0,  # u(x) = 0 for the first third
            lambda x: a*x,  # Linear ramp for the middle third
            1  # u(x) = 1 for the last third
        ]

        # Apply the conditions and corresponding functions to x using np.piecewise
        u = np.piecewise(x, conditions, functions)
        
        return u

    def lax_friedrichs(self):
        """
        Lax-Friedrichs scheme to update the velocity field u with boundary conditions.
        """
        u_new = np.zeros_like(self.u)
        u_new[1:-1] = 0.5 * (self.u[2:] + self.u[:-2]) - 0.5 * self.dt / self.dx * (self.u[2:]**2 / 2 - self.u[:-2]**2 / 2)

        # Enforce boundary conditions:
        u_new[0] = 0  # Left boundary: u(0, t) = 0
        u_new[-1] = 1  # Right boundary: u(L, t) = 1
        
        self.u = u_new

    def solve(self):
        """
        Solve the Burgers equation using the Lax-Friedrichs scheme, 
        and save each time step as an individual frame (image).
        """
        if not os.path.exists('frames'):
            os.makedirs('frames')  # Create frames directory if not exists

        for t_step in range(1, self.nt + 1):
            self.lax_friedrichs()
            # Save each frame as a PNG file
            filename = f'frames/frame_{t_step:04d}.png'
            self.plot_u(filename, t_step * self.dt)

    def get_solution(self):
        """
        Return the final velocity field.
        """
        return self.u

    def plot_initial_u(self, filename='frames/initial_u_profile.png'):
        """
        Plot the initial velocity field u(x, 0) and save the plot to a file.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.x, self.u, label="Initial u(x, 0)", color="blue", lw=2)
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Initial Velocity Field u(x, 0)')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def plot_u(self, filename, time):
        """
        Plot the velocity field u(x) at a given time step and save the plot to a file.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.x, self.u, label=f'u(x, t={time:.7f})', color="blue", lw=2)
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title(f'Velocity Field u(x) at t = {time:.7f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def create_mp4(self, output_file='simulation.mp4', fps=10):
        """
        Use ffmpeg to combine the frames into an MP4 file.
        """
        # Run ffmpeg command to create the mp4 video
        command = [
            'ffmpeg', '-framerate', str(fps), '-i', 'frames/frame_%04d.png',  # Input image files
            '-r', '30',  # Output framerate
            '-pix_fmt', 'yuv420p',  # Pixel format for wide compatibility
            output_file  # Output file
        ]

        try:
            subprocess.run(command, check=True)
            print(f"MP4 video created: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating MP4 video: {e}")

