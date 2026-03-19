import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the absolute path of this script's directory (Plots/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root directory (ReachAvoidAlgo/)
# Goes up one level from 'scripts/'
project_root_dir = os.path.dirname(current_dir)
# Add the project root directory to sys.path
sys.path.insert(0, project_root_dir)

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

        # --- Initialise Dynamics and Simulation ---
        robot_dynamics = ManipulatorDynamics(m, L, q_start, q_end, robot_type)
        simulator = Simulator(min_tau_loaded, max_tau_loaded, robot_dynamics)
        
        # Check if path is lipschtiz continuous
        q_s_dot_path = q_end - q_start
        q_s = lambda s: q_start + s * q_s_dot_path
        lipschitz = HelperFunctions.is_lipschitz_continuous(q_s)

        print("\nCalculating Velocity Limit Curve (V_u)...")
        # Lots of x1 values
        x1_star = np.linspace(0, 1, 401)
        initial_guess = 30.0
        # Calculate the upper boundary
        V_u = simulator.calculate_upper_boundary(x1_star, initial_guess)
        # Set lower boundary to zero
        V_l = np.zeros_like(V_u)
        
        print("\nCreating computable functions C_u(x1) and C_l(x1)...")
        poly_degree = 6
        C_l = lambda x1: np.zeros_like(x1)
        C_l_coeffs = np.zeros(poly_degree + 1)
        C_u, C_u_coeffs = simulator.create_boundary_function(V_u, lipschitz[1], x1_star)

        print("\n--- Finding roots of S(x) on the boundaries ---")
        reach_calc = ReachabilityCalculator(C_u, C_l, simulator, C_u_coeffs, C_l_coeffs)
        roots_lower, roots_upper = reach_calc.find_S_roots(x1_star)
        roots = sorted(list(set(roots_lower + roots_upper)))
        
        # Print the results
        if roots_lower:
            print("\nFound roots of S(x) on lower boundary at x1 =")
            for root in roots_lower: 
                print(f"  {root:.6f}")
        else:
            print("\nNo roots of S(x) were found on the lower boundary.")
            
        if roots_upper:
            print("\nFound roots of S(x) on upper boundary at x1 =")
            for root in roots_upper: 
                print(f"  {root:.6f}")
        else:
            print("\nNo roots of S(x) were found on the upper boundary.")
            
        if roots:
            print("\nFounds roots of S(x) on boundaries at x1 =")
            for root in roots:
                print(f" {root:.6f}")
        
        print("\n--- Generating Partition I ---")
        tolerance = 1e-6
        # Initialise lists for intervals
        I_in_lower, I_out_lower, I_lower = reach_calc.generate_partition_I(roots_lower, C_l, tolerance)
        I_in_upper, I_out_upper, I_upper = reach_calc.generate_partition_I(roots_upper, C_u, tolerance)
        
        print("\nIntervals where S(x) <= 0 (I_in) on the lower boundary:")
        for interval in I_in_lower:
            print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
        
        print("\nIntervals where S(x) > 0 (I_out) on the lower boundary:")
        for interval in I_out_lower:
            print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
            
        print("\nIntervals where S(x) <= 0 (I_in) on the upper boundary:")
        for interval in I_in_upper:
            print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
        
        print("\nIntervals where S(x) > 0 (I_out) on the upper boundary:")
        for interval in I_out_upper:
            print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
        
        # --- Plotting ---
        print("\nGenerating plot...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        # Create a finer set of x1-values for a smoother plot
        x1_fine = np.linspace(0, 1, 500)
        
        # Plot the upper and lower boundary sets
        if False:
            plt.plot(x1_star, V_u, 'b-', label='Upper Boundary Set, $V_u$')
            plt.plot(x1_star, V_l, 'g-', label='Lower Boundary Set, $V_l$')
            
        
        # Right Subplot - Lower roots
        ax1.plot(x1_fine, C_u(x1_fine), 'r--', label='$C_u(x_1)$')
        ax1.plot(x1_fine, C_l(x1_fine), 'm--', label='$C_l(x_1)$')

        # Plot lower roots
        if roots_lower:
            ax1.vlines(x=roots_lower, ymin=0, ymax=C_u(np.array(roots_lower)), 
                    colors='green', linestyles='solid', label='Roots of $S(x)$')

        # Shade intervals for lower boundary
        for i, interval in enumerate(I_in_lower):
            x1_start = interval[1]
            x1_end = interval[0]
            x1_mask = x1_fine[(x1_fine >= x1_start) & (x1_fine <= x1_end)]
            label = '$S(x) \\leq 0$' if i == 0 else ''
            ax1.fill_between(x1_mask, C_l(x1_mask), C_u(x1_mask), 
                     color='lightgreen', alpha=0.3, label=label)

        for i, interval in enumerate(I_out_lower):
            x1_start = interval[1]
            x1_end = interval[0]
            mask = (x1_fine >= x1_start) & (x1_fine <= x1_end)
            label = '$S(x) > 0$' if i == 0 else ''
            ax1.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine), 
                            where=mask, color='lightcoral', alpha=0.3, label=label)

        ax1.set_xlabel('Path Parameter, $x_1$', fontsize=12)
        ax1.set_ylabel('Path Velocity, $x_2$ (rad/s)', fontsize=12)
        ax1.set_title('Lower Boundary Roots', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True)

        # Right Subplot - Upper Roots
        ax2.plot(x1_fine, C_u(x1_fine), 'r--', label='$C_u(x_1)$')
        ax2.plot(x1_fine, C_l(x1_fine), 'm--', label='$C_l(x_1)$')

        # Plot upper roots
        if roots_upper:
            ax2.vlines(x=roots_upper, ymin=0, ymax=C_u(np.array(roots_upper)), 
                    colors='green', linestyles='solid', label='Roots of $S(x)$')

        # Shade intervals for upper boundary
        for i, interval in enumerate(I_in_upper):
            x1_start = interval[1]
            x1_end = interval[0]
            mask = (x1_fine >= x1_start) & (x1_fine <= x1_end)
            label = '$S(x) \\leq 0$' if i == 0 else ''
            ax2.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine), 
                            where=mask, color='lightgreen', alpha=0.3, label=label)

        for i, interval in enumerate(I_out_upper):
            x1_start = interval[1]
            x1_end = interval[0]
            mask = (x1_fine >= x1_start) & (x1_fine <= x1_end)
            label = '$S(x) > 0$' if i == 0 else ''
            ax2.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine), 
                            where=mask, color='lightcoral', alpha=0.3, label=label)

        ax2.set_xlabel('Path Parameter, $x_1$', fontsize=12)
        ax2.set_ylabel('Path Velocity, $x_2$ (rad/s)', fontsize=12)
        ax2.set_title('Upper Boundary Roots', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        
        # Plot the upper and lower boundart functions C_u and C_l
        # plt.plot(x1_fine, C_u(x1_fine), 'r--', label='Upper Boundary Function, $C_u(x1)$')
        # plt.plot(x1_fine, C_l(x1_fine), 'm--', label='Lower Boundary Function, $C_l(x1)$')
        
        # # If any roots were found
        # if roots:
        #     # Draw the vertical lines from y=0 up to the C_u curve
        #     plt.vlines(x=roots, ymin=0, ymax=C_u(np.array(roots)), colors='green', linestyles='solid', label='Roots of $S(x)$')
        
        # # Shade the intervals
        # for i, interval in enumerate(I_in_upper):
        #     # Extract the start and end of the interval
        #     x1_start = interval[1]
        #     x1_end = interval[0]
    
        #     # Create a mask for the x1-values within the interval
        #     mask = (x1_fine >= x1_start) & (x1_fine <= x1_end)
            
        #     if i == 0:
        #         label = '$S(x) \\leq 0$'
        #     else:
        #         label = ''
    
        #     # Fill the region between C_l and C_u for the interval
        #     plt.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine),where=mask, color='lightgreen', alpha=0.3, label=label)
        
        # for i, interval in enumerate(I_out_upper):
        #     # Extract the start and end of the interval
        #     x1_start = interval[1]
        #     x1_end = interval[0]
    
        #     # Create a mask for the x1-values within the interval
        #     mask = (x1_fine >= x1_start) & (x1_fine <= x1_end)
            
        #     if i == 0:
        #         label = '$S(x) > 0$'
        #     else:
        #         label = ''
    
        #     # Fill the region between C_l and C_u for the interval
        #     plt.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine), where=mask, color='lightcoral', alpha=0.3, label=label)
        
        # # Shade the constraint set X (from C_l to C_u, from x1=0 to x1=1)
        # # plt.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine), 
        # #                  where=(C_u(x1_fine) >= C_l(x1_fine)), color='lightgrey', 
        # #                  alpha=0.8, label='Constraint Set (X)')
        
        # plt.legend(fontsize=10)
        # plt.show()

    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")