import numpy as np
import matplotlib.pyplot as plt

from ManipulatorDynamics import ManipulatorDynamics
from Simulator import Simulator
from ReachabilityCalculator import ReachabilityCalculator
import HelperFunctions

if __name__ == "__main__":
    try:
        # Load all parameters
        m, L, q_start, q_end, min_tau_loaded, max_tau_loaded = HelperFunctions.load_parameters_from_file('parameters.txt')

        print("\n--- Loaded Parameters ---")
        print("Masses (m):", m)
        print("Lengths (L):", L)
        print("Path Start (q_start_rad):", q_start)
        print("Path End (q_end_rad):", q_end)
        print("Min Torques (min_tau):", min_tau_loaded)
        print("Max Torques (max_tau):", max_tau_loaded)

        # --- Initialize Dynamics and Simulation ---
        robot_dynamics = ManipulatorDynamics(m, L, q_start, q_end)
        simulator = Simulator(min_tau_loaded, max_tau_loaded, robot_dynamics)

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
        C_u, C_u_coeffs = Simulator.create_constrained_polynomial(x1_star, V_u, degree=poly_degree)
        C_u_coeffs = np.flip(C_u_coeffs)
        C_l = lambda x1: np.zeros_like(x1)
        C_l_coeffs = np.zeros(poly_degree + 1)
        
        # Derivative of Cu (check this)
        print(f"Cu coefficients: {C_u_coeffs}")
        Cu_dervi_coeffs = [C_u_coeffs[i] * i for i in range(1, len(C_u_coeffs))]
        print(f"Cu derivative coefficients: {Cu_dervi_coeffs}")
        # Values of Cu derivative at x1_star
        Cu_deriv_values = np.polyval(np.flip(Cu_dervi_coeffs), x1_star)
        # Detect rough approx. maximum 
        x1_maximizer_approx = np.max(np.abs(Cu_deriv_values))
        # Refinement step (use an optimiser to find L (todo)
        sadety_diff = x1_maximizer_approx * (x1_star[1] - x1_star[0]) / 2
        print(f"Safety margin: {sadety_diff}")
        
        plt.plot(x1_star, Cu_deriv_values)
        plt.grid()
        plt.show()
        
        C_u_coeffs = np.flip(C_u_coeffs)
        

        print("\n--- Finding roots of S(x) on the boundaries ---")
        reach_calc = ReachabilityCalculator(C_u, C_l, simulator, C_u_coeffs, C_l_coeffs)
        roots = reach_calc.find_S_roots(x1_star)

        # Print the results
        if roots:
            print("\nFound roots of S(x) on Boundaries at x1 =")
            for root in roots: 
                print(f"  {root:.6f}")
        else:
            print("\nNo roots of S(x) were found.")
            
        
        print("\n--- Generating Partition I ---")
        tolerance = 1e-6
        # Initialize lists for intervals
        I_in, I_out, I = reach_calc.generate_partition_I(roots, tolerance)
        
        print("\nIntervals where S(x) <= 0 (I_in):")
        for interval in I_in:
            print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
        
        print("\nIntervals where S(x) > 0 (I_out):")
        for interval in I_out:
            print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
        
        
        # --- Plotting ---
        print("\nGenerating plot...")
        # Enable LaTeX rendering for all text in figures
        #plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(12, 7))
        plt.xlabel('Path Parameter, $x_1$', fontsize=12)
        plt.ylabel('Path Velocity, $x_2$ (rad/s)', fontsize=12)
        plt.title('Reach Avoid Set Algorithm', fontsize=14)
        plt.grid(True)
        
        # Plot the upper and lower boundary sets
        plt.plot(x1_star, V_u, 'b-', label='Upper Boundary Set, $V_u$')
        plt.plot(x1_star, V_l, 'g-', label='Lower Boundary Set, $V_l$')
        # Create a finer set of x1-values for a smoother plot
        x1_fine = np.linspace(0, 1, 500)
        # Plot the upper and lower boundart functions C_u and C_l
        plt.plot(x1_fine, C_u(x1_fine) - sadety_diff, 'r--', label='Upper Boundary Function, $C_u(x1)$')
        plt.plot(x1_fine, C_l(x1_fine), 'm--', label='Lower Boundary Function, $C_l(x1)$')
        
        # If any roots were found
        if roots:
            # Draw the vertical lines from y=0 up to the C_u curve
            plt.vlines(x=roots, ymin=0, ymax=C_u(np.array(roots)), colors='green', linestyles='solid', label='Roots of $S(x)$')
        
        # Shade the intervals
        for i, interval in enumerate(I_in):
            # Extract the start and end of the interval
            x1_start = interval[1]
            x1_end = interval[0]
    
            # Create a mask for the x1-values within the interval
            mask = (x1_fine >= x1_start) & (x1_fine <= x1_end)
            
            if i == 0:
                label = '$S(x) \leq 0$'
            else:
                label = ''
    
            # Fill the region between C_l and C_u for the interval
            plt.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine),where=mask, color='lightgreen', alpha=0.3, label=label)
        
        for i, interval in enumerate(I_out):
            # Extract the start and end of the interval
            x1_start = interval[1]
            x1_end = interval[0]
    
            # Create a mask for the x1-values within the interval
            mask = (x1_fine >= x1_start) & (x1_fine <= x1_end)
            
            if i == 0:
                label = '$S(x) > 0$'
            else:
                label = ''
    
            # Fill the region between C_l and C_u for the interval
            plt.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine), where=mask, color='lightcoral', alpha=0.3, label=label)
        
        # Shade the constraint set X (from C_l to C_u, from x1=0 to x1=1)
        # plt.fill_between(x1_fine, C_l(x1_fine), C_u(x1_fine), 
        #                  where=(C_u(x1_fine) >= C_l(x1_fine)), color='lightgrey', 
        #                  alpha=0.8, label='Constraint Set (X)')
        
        plt.legend(fontsize=10)
        plt.show()

    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")