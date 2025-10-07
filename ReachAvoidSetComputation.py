import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
        x1_star = np.linspace(0, 1, 101)
        initial_guess = 30.0
        # Calculate the upper boundary
        V_u = simulator.calculate_upper_boundary(x1_star, initial_guess)
        # Set lower boundary to zero
        V_l = np.zeros_like(V_u)
        
        print("\nCreating computable functions C_u(x1) and C_l(x1)...")
        poly_degree = 10
        C_u, C_u_coeffs = Simulator.create_constrained_polynomial(x1_star, V_u, degree=poly_degree)
        C_l = lambda x1: np.zeros_like(x1)
        C_l_coeffs = np.zeros(poly_degree + 1)
        
        print("\n--- Finding roots of S(x) on the boundaries ---")
        reach_calc = ReachabilityCalculator(C_u, C_l, simulator, C_u_coeffs, C_l_coeffs)
        roots = reach_calc.find_S_roots(x1_star)
        
        print("\n--- Generating Partition I ---")
        tolerance = 1e-6
        # Initialize lists for intervals
        I_in, I_out, I = reach_calc.generate_partition_I(roots, tolerance)
        
        print("\n--- Generating Target Set, X_T ---")
        xstar_u = 4
        xstar_l = 0.05
        # Target set as a line segment in state space - [x1, x2_min, x2_max]
        X_T = [0.8, xstar_l, xstar_u]
        
        print("\n--- Simulating System Trajectories, Tb(xstar_u, 0) & Tb(xstar_l, 1) ---")
        t_span = (0.0, 10.0)
        # Initial conditions - top of the target set
        x0_u = np.array([X_T[0], xstar_u])
        # Initial conditions - bottom of the target set
        x0_l = np.array([X_T[0], xstar_l])
        # Integration direction
        direction = 'backward'
        
        # Stopping conditions:
        cross_x_axis = ReachabilityCalculator.event_x2_zero
        cross_y_axis = ReachabilityCalculator.event_x1_zero
        cross_Cu = lambda t, x, C_u=C_u: ReachabilityCalculator.event_Cu_cross(t, x, C_u)
        cross_Cl = lambda t, x, C_l=C_l: ReachabilityCalculator.event_Cl_cross(t, x, C_l)
        # Stop when any of these events occur
        cross_x_axis.terminal = True
        cross_y_axis.terminal = True
        cross_Cu.terminal = True
        cross_Cl.terminal = True
        # Set direction to zero to detect all crossings
        cross_x_axis.direction = 0
        cross_y_axis.direction = 0
        cross_Cu.direction = 0
        cross_Cl.direction = 0
        
        # Define the ode function for upper trajectory
        dynamics_func_upper = lambda t, x, direction=direction, u=0: simulator.get_double_integrator_dynamics(t, x, direction, u)
        # Solve ODE
        T_back_u = solve_ivp(dynamics_func_upper, t_span, x0_u, method='RK45', events=[cross_x_axis, cross_y_axis, cross_Cu, cross_Cl], dense_output=True, rtol=1e-6, atol=1e-8)
        print(T_back_u.message)
        # The last valid time before the event
        t_end = T_back_u.t[-1]
        # Create a dense time array for smooth plotting
        t_dense = np.linspace(T_back_u.t[0], t_end, 500)
        # Evaluate the solution at dense time points and transpose for easier plotting
        # T_star_u = array of (x1, x2) pairs along the trajectory
        T_star_u = T_back_u.sol(t_dense).T
        
        # Define the ode function for lower trajectory
        dynamics_func_lower = lambda t, x, direction=direction, u=1: simulator.get_double_integrator_dynamics(t, x, direction, u)
        # Solve ODE
        T_back_l = solve_ivp(dynamics_func_lower, t_span, x0_l, method='RK45', events=[cross_x_axis, cross_y_axis, cross_Cu, cross_Cl], dense_output=True, rtol=1e-6, atol=1e-8)
        print(T_back_l.message)
        # The last valid time before the event
        t_end = T_back_l.t[-1]
        # Create a dense time array for smooth plotting
        t_dense = np.linspace(T_back_l.t[0], t_end, 500)
        # Evaluate the solution at dense time points and transpose for easier plotting
        # T_star_l = array of (x1, x2) pairs along the trajectory
        T_star_l = T_back_l.sol(t_dense).T
        
        
        # Find the index of the left most point in the trajectory (i.e., minimum x1)
        min_index_u = np.argmin(T_star_u[:,0])
        x_d = T_star_u[min_index_u]
        print(f"Left most state in trajectory Tb(xstar_u, 0): {x_d[0]:.6f}, {x_d[1]:.6f}")
        # Find the index of the left most point in the trajectory (i.e., minimum x1)
        min_index_l = np.argmin(T_star_l[:,0])
        x_a = T_star_l[min_index_l]
        print(f"Left most state in trajectory Tb(xstar_l, 1): {x_a[0]:.6f}, {x_a[1]:.6f}")
        
        print("\n--- Algorithm Logic ---")
        Z_u = np.array([])
        Z_l = np.array([])
        # If x_a and x_d are on the lower boundary
        if reach_calc.is_on_lower_boundary(x_a) and reach_calc.is_on_lower_boundary(x_d):
            print("Both x_a and x_d are on the lower boundary C_l.")
            Z_u = T_star_u
        # If x_a and x_d are on the upper boundary
        elif reach_calc.is_on_upper_boundary(x_a) and reach_calc.is_on_upper_boundary(x_d):
            print("Both x_a and x_d are on the upper boundary C_u.")
            Z_l = T_star_l
        else:
            print("x_a and x_d are on different boundaries.")
            # If x_a is on the lower boundary
            if reach_calc.is_on_lower_boundary(x_a):
                print("x_a is on the lower boundary C_l.")
            else:
                print("x_a is not on the lower boundary C_u.")
                Z_l = T_star_l
            
            # If x_d is on the lower boundary
            if reach_calc.is_on_upper_boundary(x_d):
                print("x_d is on the upper boundary C_l.")
            else:
                print("x_d is not on the upper boundary C_u.")
                Z_u = T_star_u
        
        
        # Plot results
        plt.figure(figsize=(10, 5))
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.title('Reach Avoid Set Computation')
        plt.grid()
        
        # Plot the upper and lower boundary sets
        # plt.plot(x1_star, V_u, 'b-', label='Upper Boundary Set, $V_u$')
        # plt.plot(x1_star, V_l, 'g-', label='Lower Boundary Set, $V_l$')
        # Create a finer set of x1-values for a smoother plot
        x1_fine = np.linspace(0, 1, 500)
        # Plot upper and lower boundary functions
        plt.plot(x1_fine, C_u(x1_fine), 'r-', label='Upper Boundary Function, $C_u(x1)$')
        plt.plot(x1_fine, C_l(x1_fine), 'm-', label='Lower Boundary Function, $C_l(x1)$')
        # Plot targer set X_T
        plt.vlines(X_T[0], X_T[1], X_T[2], colors='orange', label='Target Set, $X_T$')
        plt.plot(X_T[0], xstar_u, 'ko', label='$x^*_u$')
        plt.plot(X_T[0], xstar_l, 'ko', label='$x^*_l$')
        
        # Plot trajectories
        plt.plot(T_star_u[:, 0], T_star_u[:, 1], 'c-', label='Trajectory, $T^*_u$')
        plt.plot(T_star_l[:, 0], T_star_l[:, 1], 'y-', label='Trajectory, $T^*_l$')
        # Plot x_d
        plt.plot(x_d[0], x_d[1], 'ks', label='$x_d$')
        # Plot x_a
        plt.plot(x_a[0], x_a[1], 'k^', label='$x_a$')
        
        Z = HelperFunctions.slice(C_u, I_in[0])
        plt.plot(Z[:, 0], Z[:, 1], 'k-', label='Test Slice')
        Z = HelperFunctions.slice(C_u, I_in[0])
        plt.plot(Z[:, 0], Z[:, 1], 'k-')
        Z = HelperFunctions.slice(T_star_u, I_in[0])
        plt.plot(Z[:, 0], Z[:, 1], 'k-', label='Trajectory Slice')
        
        plt.legend()
        plt.show()
        
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")
