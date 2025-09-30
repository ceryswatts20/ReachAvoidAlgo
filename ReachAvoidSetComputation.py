import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ManipulatorDynamics import ManipulatorDynamics
from Simulator import Simulator
from ReachabilityCalculator import ReachabilityCalculator
import HelperFunctions

"""
    Event function to stop integration when x1 = 0 or x2 crosses C_u or C_l.
    
    Args:
        t (float): Current time.
        y (np.ndarray): Current state vector [x1, x2].
        *args: Additional arguments (C_u, C_l).
        
    Returns:
        np.ndarray: Array of values indicating when to stop.
"""
def backwardsStopEvent(t, y, *args):
    
    x1, x2 = y
    C_u, C_l = args
    
    # Stop if x1 = 0
    value_x1 = x1
    # Stop if crossing C_u
    value_Cu = x2 - C_u(x1)
    # Stop if crossing C_l
    value_Cl = x2 - C_l(x1)
    
    return np.array([value_x1, value_Cu, value_Cl])

"""
    Solves the ODE for the given initial conditions and parameters.
    
    Args:
        C_u (callable): Upper boundary function.
        C_l (callable): Lower boundary function.
        boundary_simulator (Simulator): Instance of Simulator.
        initial_conditions (np.ndarray): Initial state [x1, x2].
        t_span (tuple): Start and end times for the integration.
        direction (str): Direction of integration ('forward' or 'backward').
        u (int): Control input (0 or 1).
    
    Returns:
        OdeSolution: Solution object containing the results of the ODE integration.
"""
def solve_ode(C_u, C_l, func, initial_conditions, t_span):
    
        # Event handling
        # Stop when either condition is met
        isterminal = [True, True, True]
        # Detect any crossing
        direction = [0, 0, 0]
        
        # Solve ODE
        sol = solve_ivp(func, t_span, initial_conditions, method='RK45', events=backwardsStopEvent, args=(C_u, C_l), rtol=1e-6, atol=1e-8)
        
        return sol

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
        
        
        # Initial conditions
        x0 = np.array([1.0, 0.0])
        t_span = (0.0, 10.0)
        # Integration direction
        direction = 'backward'
        # Max decceleration dynamics (L)
        u = 0
        
        function = lambda t, y: simulator.get_double_integrator_dynamics(y, direction, u)
        sol = solve_ode(C_u, C_l, function, x0, t_span)
        
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
        
        plt.plot(sol.t, sol.y[0], label='x1')
        plt.plot(sol.t, sol.y[1], label='x2')
        
        plt.legend()
        plt.show()
        
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")
