import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

from ManipulatorDynamics import ManipulatorDynamics
from BoundarySimulator import BoundarySimulator
from ReachabilityCalculator import ReachabilityCalculator

def load_parameters_from_file(filepath='parameters.txt'):
    # Dictionary to hold extracted parameters
    parameters = {}
    current_key = None
    # Map comment headers to parameter keys
    key_map = {
        "Min joint torques": "min_torques",
        "Max joint torques": "max_torques",
        "Path start point": "q_start_deg",
        "Path end point": "q_end_deg",
        "2 DOF link masses": "m",
        "2 DOF link lengths": "L",
    }
    try:
        with open(filepath, 'r') as f:
            # For each line in the file
            for line in f:
                # Strip whitespace
                line = line.strip()
                # If the line is not empty or is a comment
                if not line or line.startswith('%'):
                    # Removes the '%' and any leading/trailing whitespace
                    header_text = line.lstrip('%').strip()
                    # Check if this header is one we care about i.e., in key_map
                    if header_text in key_map:
                        # Set the current key to the corresponding parameter name
                        current_key = key_map[header_text]
                    continue
                # If we have a current key
                if current_key:
                    try:
                        # Convert the line into a numpy array of floats
                        values = np.array([float(val.strip()) for val in line.split(',')])
                        # Store the values in the parameters dictionary
                        parameters[current_key] = values
                        # Reset current_key
                        current_key = None
                    except ValueError:
                        print(f"Warning: Could not parse values for '{current_key}' from line: '{line}'")
                        current_key = None
                else:
                    print(f"Warning: Data line '{line}' found without a preceding header comment. Skipping.")
        
        # Check if all required parameters are present
        required_keys = ["m", "L", "q_start_deg", "q_end_deg", "min_torques", "max_torques"]
        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Missing required parameter: '{key}' in '{filepath}'")

        m = parameters["m"]
        L = parameters["L"]
        # Convert degrees to radians for joint angles
        q_start_rad = np.deg2rad(parameters["q_start_deg"])
        q_end_rad = np.deg2rad(parameters["q_end_deg"])
        min_tau = parameters["min_torques"]
        max_tau = parameters["max_torques"]

        return m, L, q_start_rad, q_end_rad, min_tau, max_tau
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure it exists.")
        raise
    except Exception as e:
        print(f"An error occurred while loading parameters: {e}")
        raise


if __name__ == "__main__":
    try:
        # Load all parameters
        m, L, q_start, q_end, min_tau_loaded, max_tau_loaded = load_parameters_from_file('parameters.txt')

        print("\n--- Loaded Parameters ---")
        print("Masses (m):", m)
        print("Lengths (L):", L)
        print("Path Start (q_start_rad):", q_start)
        print("Path End (q_end_rad):", q_end)
        print("Min Torques (min_tau):", min_tau_loaded)
        print("Max Torques (max_tau):", max_tau_loaded)

        # --- Initialize Dynamics and Simulation ---
        robot_dynamics = ManipulatorDynamics(m, L, q_start, q_end)
        boundaries = BoundarySimulator(min_tau_loaded, max_tau_loaded, robot_dynamics)

        # Lots of s values
        s_star = np.linspace(0, 1, 101)
        # List to store found sdot values
        V_u = []

        print("\nCalculating Velocity Limit Curve (V_u)...")
        current_sdot_guess = 30.0
        # Iterate through each s value and find the sdot where VLC = 0
        for s_val in s_star:
            # Define the upper boundary function for fsolve
            upper_bound= lambda sdot_to_solve: \
                boundaries.calculate_upper_boundary(s_val, sdot_to_solve)
            try:
                sdot_found = fsolve(upper_bound, current_sdot_guess,
                                    full_output=False,
                                    factor=0.1)[0]
                V_u.append(sdot_found)
                current_sdot_guess = sdot_found
                
            except Exception as e:
                print(f"Warning: fsolve failed to converge for s = {s_val:.3f}. ")

        # Convert lists to numpy arrays for easier handling
        V_u = np.array(V_u)
        V_l = np.zeros_like(V_u)
        
        # Convert sets into formulas C_u and C_l
        print("\nCreating computable functions C_u(s) and C_l(s)...")

        # C_l: a function that always returns 0
        C_l = lambda s: np.zeros_like(s)
        # C_u: interpolate V_u for values within [0, 1]
        # bounds_error = Raise an error for out-of-bounds access, i.e s is not in [0, 1]
        C_u = interp1d(s_star, V_u, kind='linear', bounds_error=True)
        # Create a finer set of s-values for a smoother plot
        s_fine = np.linspace(0, 1, 500)
        
        
        # --- Initialize the ReachabilityCalculator and test S(x) ---
        reach_calc = ReachabilityCalculator(C_u, C_l, boundaries, s_star)

        print("\n--- Demonstrating S(x) calculation ---")

        # Test S(x) on the upper boundary
        s_test_upper = 0.5
        x_upper = np.array([s_test_upper, C_u(s_test_upper)])
        S_upper = reach_calc.calculate_S(x_upper)
        print(f"For state x = [{x_upper[0]:.3f}, {x_upper[1]:.3f}] on C_u, S(x) = {S_upper:.4f}")

        # Test S(x) on the lower boundary
        s_test_lower = 0.5
        x_lower = np.array([s_test_lower, C_l(s_test_lower)])
        S_lower = reach_calc.calculate_S(x_lower)
        print(f"For state x = [{x_lower[0]:.3f}, {x_lower[1]:.3f}] on C_l, S(x) = {S_lower:.4f}")
        
        
        # Enable LaTeX rendering for all text in figures
        #plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(12, 7))
        # Plot the upper and lower boundary sets
        plt.plot(s_star, V_u, 'b-', label='Upper Boundary Set, $V_u$')
        plt.plot(s_star, V_l, 'g-', label='Lower Boundary Set, $V_l$')
        # Plot the upper and lower boundart functions C_u and C_l
        plt.plot(s_fine, C_u(s_fine), 'r--', label='Upper Boundary Function, $C_u(s)$')
        plt.plot(s_fine, C_l(s_fine), 'm--', label='Lower Boundary Function, $C_l(s)$')
        # Shade the constraint set X (from V_l to V_u, from s=0 to s=1)
        plt.fill_between(s_star, V_l, V_u, where=(V_u >= V_l), color='lightgrey', alpha=0.3, label='Constraint Set (X)')
        
        plt.xlabel('Path Parameter \($s$\)', fontsize=12)
        plt.ylabel('Path Velocity \($\dot{s}$\) (rad/s)', fontsize=12)
        plt.title('Reach Avoid Set Algorithm', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        plt.show()

    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")