import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from BurgerSolver import BurgerSolver

class BurgerAnimation:
    def __init__(self, solver: BurgerSolver):
        self.solver = solver
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.solver.x, self.solver.u, lw=2)
        self.ax.set_xlim(0, self.solver.L)
        self.ax.set_ylim(-1.5, 1.5)
        
        # Add a title placeholder for displaying the time
        self.time_text = self.ax.text(0.5, 1.05, '', transform=self.ax.transAxes, ha='center')

    def update(self, frame):
        """
        Update the graph at each frame of the animation. The frame corresponds to the current time step.
        """
        # Perform a time step in the solver
        self.solver.lax_friedrichs()
        
        # Update the line data with the new solution
        self.line.set_ydata(self.solver.get_solution())
        
        # Calculate the current time based on the frame and solver's time step
        current_time = frame * self.solver.dt
        
        # Update the title to show the current time
        self.time_text.set_text(f'Time: {current_time:.6f} s')
        
        return self.line, self.time_text

    def animate(self):
        # Ensure the results directory exists
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Save the animation in the results folder
        animation_path = os.path.join(results_dir, 'burger_inviscid_sine_wave.gif')
        ani = FuncAnimation(self.fig, self.update, frames=self.solver.nt, blit=True, interval=200)
        ani.save(animation_path, writer='pillow')

        print(f"Animation saved at {animation_path}")

    def show(self):
        plt.show()
