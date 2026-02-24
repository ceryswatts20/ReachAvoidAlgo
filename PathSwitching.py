import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from ManipulatorDynamics import ManipulatorDynamics
from Simulator import Simulator
from ReachabilityCalculator import ReachabilityCalculator
import HelperFunctions

if __name__ == "__main__":
    try:
        # Load all parameters
        robot_type, m, L, q_start, q_end, min_tau_loaded, max_tau_loaded = HelperFunctions.load_parameters_from_file('parameters.txt')

        print("\n--- Loaded Parameters ---")
        print("Robot Type:", robot_type)
        print("Masses (m):", m)
        print("Lengths (L):", L)
        print("Path Start (q_start_rad):", q_start)
        print("Path End (q_end_rad):", q_end)
        print("Min Torques (min_tau):", min_tau_loaded)
        print("Max Torques (max_tau):", max_tau_loaded)
        
        print(f"\n--- Lipschitz Continuity Check ---")
        # Path
        q_s_dot_path = q_end - q_start
        q_s = lambda s: q_start + s * q_s_dot_path
        # Check Lipschitz continuity of the path
        lipschitz = HelperFunctions.is_lipschitz_continuous(q_s)
        # If the path is Lipschitz continuous, print the constant
        if lipschitz[0]:
            print(f"The path is Lipschitz continuous with constant L = {lipschitz[1]:.4f}")
        else:
            print("The path is not Lipschitz continuous.")
            # Stop the program if the path is not Lipschitz continuous, as the algorithm relies on this property
            exit(1)
        
        # --- Initialise Dynamics and Simulation ---
        robot_dynamics = ManipulatorDynamics(m, L, q_start, q_end, robot_type)
        simulator = Simulator(min_tau_loaded, max_tau_loaded, robot_dynamics)
        
        # Paths and switching points
        
        # 1. Calculate the reach-avoid set for qA(s)
        
        
        # 2. Calculate the reach-avoid set for qB(s)
        # Not needed in this example
        
        # 3. Calculate the reach-avoid set for the 2nd path segment
        
        
        # 4. Calculate the reach-avoid set for the 3rd path segment
        
        
        # 5. Implement a controller to go from the start of the path switching path to the 1st switching point, p(1), i.e qA(1)
        
        
        # 6. Implement a controller to go from the 1st switching point, p(1), i.e qB(0), to the 2nd switching point, p(2)
        
        
        # 7. Implement a controller to go from the 2nd switching point, p(2), to the 3rd switching point, p(3)
        
        
        # 8. Implement a controller to go from the 3rd switching point, p(3), to the end of the path switching path
        
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")