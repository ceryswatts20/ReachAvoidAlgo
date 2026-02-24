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
        
        theta = 5
        # --- Initialise Dynamics and Simulation ---
        robot_dynamics_A = ManipulatorDynamics(m, L, q_start, q_end)
        robot_dynamics_B = ManipulatorDynamics(m, L, q_start, q_end)

        simulator_A = Simulator(min_tau_loaded, max_tau_loaded, robot_dynamics_A)
        simulator_B = Simulator(min_tau_loaded, max_tau_loaded, robot_dynamics_B)

        print("\nCalculating Velocity Limit Curve (V_u)...")
        # Lots of x1 values
        x1_star = np.linspace(0, 1, 401)
        initial_guess = 30.0
        # Calculate the upper boundary
        V_u = simulator_A.calculate_upper_boundary(x1_star, initial_guess)
        # Set lower boundary to zero
        V_l = np.zeros_like(V_u)
        
        V_u_B = simulator_B.calculate_upper_boundary(x1_star, initial_guess)
        
        print("\nCreating computable functions C_u(x1) and C_l(x1)...")
        poly_degree = 6
        C_u, C_u_coeffs = Simulator.create_constrained_polynomial(x1_star, V_u, degree=poly_degree)
        C_u_coeffs = np.flip(C_u_coeffs)
        C_l = lambda x1: np.zeros_like(x1)
        C_l_coeffs = np.zeros(poly_degree + 1)
        
        C_u_B, C_u_B = Simulator.create_constrained_polynomial(x1_star, V_u_B, degree=poly_degree)
        
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
        
        C_u_coeffs = np.flip(C_u_coeffs)
        

        print("\n--- Finding roots of S(x) on the boundaries ---")
        reach_calc = ReachabilityCalculator(C_u, C_l, simulator_A, C_u_coeffs, C_l_coeffs)
        roots = reach_calc.find_S_roots(x1_star)
        
        reach_calc_no_grav = ReachabilityCalculator(C_u_B, C_l, simulator_B, C_u_B, C_l_coeffs)
        roots_no_grav = reach_calc_no_grav.find_S_roots(x1_star)

        # Print the results
        if roots:
            print("\nFound roots of S(x) on Boundaries at x1 =")
            for root in roots: 
                print(f"  {root:.6f}")
        else:
            print("\nNo roots of S(x) were found.")
        
        
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
        plt.plot(x1_star, V_u_B, 'c-', label='Upper Boundary Set No Gravity, $V_{u,no\ grav}$')
        plt.plot(x1_star, V_l, 'm-', label='Lower Boundary Set No Gravity, $V_{l,no\ grav}$')
        # Create a finer set of x1-values for a smoother plot
        x1_fine = np.linspace(0, 1, 500)
        # Plot the upper and lower boundart functions C_u and C_l
        plt.plot(x1_fine, C_u(x1_fine) - sadety_diff, 'r--', label='Upper Boundary Function, $C_u(x1)$')
        plt.plot(x1_fine, C_l(x1_fine), 'm--', label='Lower Boundary Function, $C_l(x1)$')
        plt.plot(x1_fine, C_u_B(x1_fine) - sadety_diff, 'orange', linestyle='--', label='Upper Boundary Function No Gravity, $C_{u,no\ grav}(x1)$')
        plt.plot(x1_fine, C_l(x1_fine), 'brown', linestyle='--', label='Lower Boundary Function No Gravity, $C_{l,no\ grav}(x1)$')
        
        # If any roots were found
        if roots:
            # Draw the vertical lines from y=0 up to the C_u curve
            plt.vlines(x=roots, ymin=0, ymax=C_u(np.array(roots)), colors='green', linestyles='solid', label='Roots of $S(x)$')
        
        
        
        plt.legend(fontsize=10)
        plt.show()

    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")

