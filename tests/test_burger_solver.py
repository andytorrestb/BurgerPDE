import sys
import os
import unittest
import numpy as np

# Add src/ folder to system path to find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import modules from src/ directory
from BurgerSolver import BurgerSolver
from Animation import BurgerAnimation

class TestBurgerSolver(unittest.TestCase):
    
    # def test_initial_condition(self):
    #     solver = BurgerSolver()
    #     u_initial = np.sin(solver.x)
    #     self.assertTrue(np.allclose(solver.u, u_initial), "Initial condition should be a sine wave.")
    
    def test_solver_step(self):
        solver = BurgerSolver()
        u_initial = solver.u.copy()
        solver.lax_friedrichs()
        self.assertEqual(solver.u.shape, u_initial.shape, "Solver should preserve the shape of u.")
    
    def test_full_solution_and_animation(self):
        """
        Test the complete process of solving the Burger equation and generating the animation.
        The animation will be saved in the results folder.
        """
        # Create the solver instance with a custom time step
        solver = BurgerSolver(dt=1e-7, nx=200, T=5e-5, L=10)
        
        # Run the solver to simulate the Burger equation
        solver.solve()

        # Verify the solution after solving
        u_final = solver.get_solution()
        self.assertIsNotNone(u_final, "Final solution should not be None.")
        self.assertEqual(u_final.shape, solver.u.shape, "Final solution should have the correct shape.")

        # Ensure the results directory exists
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Create the animation
        anim = BurgerAnimation(solver)
        anim.animate()

        # Check if the animation file was created
        animation_path = os.path.join(results_dir, "burger_inviscid_sine_wave.gif")
        self.assertTrue(os.path.exists(animation_path), f"Animation should be created at {animation_path}.")

        solver.create_mp4()

        # Optional Cleanup after test (if needed)
        if os.path.exists(animation_path):
            # Commented out for now, for manual inspection.
            # os.remove(animation_path)
            print(f"Test passed. Animation file: {animation_path}")
        else:
            print(f"Test failed. Animation file not found: {animation_path}")

# To allow running the tests directly when executing the script
if __name__ == '__main__':
    unittest.main()
