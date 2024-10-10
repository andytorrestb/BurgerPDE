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

    def update(self, frame):
        self.solver.lax_friedrichs()
        self.line.set_ydata(self.solver.get_solution())
        return self.line,

    def animate(self):
        # Ensure the results directory exists
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Save the animation in the results folder
        animation_path = os.path.join(results_dir, 'burger_inviscid_sine_wave.gif')
        ani = FuncAnimation(self.fig, self.update, frames=self.solver.nt, blit=True, interval=50)
        ani.save(animation_path, writer='pillow')

        print(f"Animation saved at {animation_path}")

    def show(self):
        plt.show()
