import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

from ManipulatorDynamics import ManipulatorDynamics
from Simulator import Simulator
from ReachabilityCalculator import ReachabilityCalculator
import HelperFunctions

"""
    Event function to stop integration when x1 = 0 or x crosses C_u or C_l.
    
    Args:
        t (float): Current time.
        y (np.ndarray): Current state vector [x1, x2].
        *args: Additional arguments (C_u, C_l).
        
    Returns:
        np.ndarray: Array of values indicating when to stop.
"""
def backwardsStopEvent(t, x, C_u, C_l):
    x1, x2 = x
    
    # Stop if x1 = 0
    value_x1 = x1
    # Stop if crossing C_u
    value_Cu = x2 - C_u(x1)
    # Stop if crossing C_l
    value_Cl = x2 - C_l(x1)
    
    # Values to monitor
    values = np.array([value_x1, value_Cu, value_Cl])
    # Stop when either condition is met
    isterminal = [True, True, True]
    
    return (values, isterminal)

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
        
        
        print("\n--- Simulating System Trajectory, Tb(xstar_u, 0) ---")
        # Initial conditions - top of the target set
        x0 = np.array([X_T[0], xstar_u])
        t_span = (0.0, 10.0)
        # Integration direction
        direction = 'backward'
        # Max decceleration dynamics (L)
        u = 0
        
        # Wrap so SciPy calls dynamics as: dynamics(t, x)
        # def dynamics_func(t, x, direction=direction, u=u):
        #     return simulator.get_double_integrator_dynamics(t, x, direction, u)

        # # Wrap for the event function: event(t, y)
        # def stopFunc(t, y, C_u=C_u, C_l=C_l):
        #     return backwardsStopEvent(t, y, C_u, C_l)
        
        dynamics_func = lambda t, x, direction=direction, u=u: simulator.get_double_integrator_dynamics(t, x, direction, u)
        stopFunc = lambda t, x, C_u=C_u, C_l=C_l: backwardsStopEvent(t, x, C_u, C_l)
        
        # Solve ODE
        T_star_u = solve_ivp(dynamics_func, t_span, x0, method='RK45', events=stopFunc, rtol=1e-6, atol=1e-8)
        
        # Plot results
        plt.figure(figsize=(10, 5))
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('System Trajectory')
        plt.grid()
        
        # Plot the upper and lower boundary sets
        plt.plot(x1_star, V_u, 'b-', label='Upper Boundary Set, $V_u$')
        plt.plot(x1_star, V_l, 'g-', label='Lower Boundary Set, $V_l$')
        # Create a finer set of x1-values for a smoother plot
        x1_fine = np.linspace(0, 1, 500)
        # Plot the upper and lower boundart functions C_u and C_l
        plt.plot(x1_fine, C_u(x1_fine), 'r--', label='Upper Boundary Function, $C_u(x1)$')
        plt.plot(x1_fine, C_l(x1_fine), 'm--', label='Lower Boundary Function, $C_l(x1)$')
        
        plt.legend()
        plt.show()
        
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")
