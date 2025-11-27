import numpy as np
from scipy.optimize import minimize, LinearConstraint, fsolve
from ManipulatorDynamics import ManipulatorDynamics

"""
Simulates the system to calculate various system boundaries
"""
class Simulator:
    """
    Initializes Simulator.
    
    Args:
        min_tau (np.ndarray): Minimum joint torques.
        max_tau (np.ndarray): Maximum joint torques.
        manipulator_dynamics_instance (ManipulatorDynamics): Instance of ManipulatorDynamics for dynamics calculations.
    """
    def __init__(self, min_tau: np.ndarray, max_tau: np.ndarray, manipulator_dynamics_instance: ManipulatorDynamics):
        
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.dynamics = manipulator_dynamics_instance
    
    """
    Calculates the joint acceleration bounds at a given (x1, x2)
    
    Args:
        x1 (float): The path parameter, [0, 1].
        x2 (float): The path velocity.
        
    Returns:
        tuple[float, float]: The acceleration bounds (L, U).
    """
    def get_accel_bounds(self, x1: float, x2: float) -> tuple[float, float]:
        lower_joint_accels = np.zeros(2)
        upper_joint_accels = np.zeros(2)
        m_s, c_s, g_s = self.dynamics.get_2_rev_dynamics(x1, x2)
        
        for i in range(2):
            m_val = m_s[i, 0]
            c_val = c_s[i, 0]
            g_val = g_s[i, 0]
            if m_val > 0:
                lower_joint_accels[i] = (self.min_tau[i] - c_val - g_val) / m_val
                upper_joint_accels[i] = (self.max_tau[i] - c_val - g_val) / m_val
            elif m_val < 0:
                lower_joint_accels[i] = (self.max_tau[i] - c_val - g_val) / m_val
                upper_joint_accels[i] = (self.min_tau[i] - c_val - g_val) / m_val
            else:
                lower_joint_accels[i] = -np.inf
                upper_joint_accels[i] = np.inf
        
        
        L = np.max(lower_joint_accels)
        U = np.min(upper_joint_accels)
        
        return L, U
    
    
    # Define the upper boundary function once to avoid redefining it in the loop
    def _upper_bound_function(self, x2_to_solve, x1):
        L, U= self.get_accel_bounds(x1, x2_to_solve)
        vlc = L - U
        
        return vlc
        
    """
    Calculate the upper boundary values by solving for x2 using fsolve for each x1 in x1_array.

    Args:
        x1_array (np.ndarray): Array of x1 values to solve for.
        initial_x2_guess (float): Initial guess for x2.

    Returns:
        np.ndarray: Array of x2 values corresponding to each x1 in x1_array.
    """
    def calculate_upper_boundary(self, x1_array: np.ndarray, initial_x2_guess: float) -> np.ndarray:
        # Array to store found x2 values
        V_u = np.array([])
        
        # Iterate through each x1 value and solve for x2
        current_x2_guess = initial_x2_guess
        for x1_val in x1_array:
            try:
                # Use fsolve to find x2 where VLC = 0
                x2_found = fsolve(self._upper_bound_function, current_x2_guess, args=(x1_val,),full_output=False, factor=0.1)[0]
                
                # Append the found x2 to the result array
                V_u = np.append(V_u, x2_found)
                
                # Update the initial guess for the next iteration
                current_x2_guess = x2_found
                
            except Exception as e:
                print(f"Warning: fsolve failed to converge for x1 = {x1_val:.3f}.")
                # Optionally, append a NaN or other placeholder
                V_u = np.append(V_u, np.nan)
        
        return V_u
    
    
    def get_double_integrator_dynamics(self, t, x: np.ndarray, direction, u: int) -> np.ndarray:
        # Matrix A (2x2)
        A = np.array([[0, 1],
                      [0, 0]])
        # Matrix B (2x1 column vector)
        B = np.array([0, 1])
        # Ensure x is 1D (SciPy passes a 1D y)
        x = np.asarray(x).reshape(2,)
        
        # Acceleration bounds
        L, U = self.get_accel_bounds(x[0], x[1])
        L = float(L)
        U = float(U)
        
        # Integrating forwards in time
        if direction == 'forward':
            # Max decceleration dynamics (L)
            if u == 0:
                return A @ x + B * L
            # Max acceleration dynamics (U)
            elif u == 1:
                return A @ x + B * U
            # Error handling
            else:
                raise ValueError("u must be 0 or 1 for this function.")
        elif direction == 'backward':
            # Min decceleration dynamics (L)
            if u == 0:
                return -A @ x - B * L
            # Max acceleration dynamics (U)
            elif u == 1:
                return -A @ x - B * U
            # Error handling
            else:
                raise ValueError("u must be 0 or 1 for this function.")
        else:
            raise ValueError("direction must be 'forward' or 'backward' for this function.")
    
    """
    Creates a polynomial approximation P(x1) for a given set, v_data, at x1 using
    constrained least squares, ensuring P(x1_i) <= v_data[i] for all i.
    
    Args:
        x1_data (np.ndarray): Discrete x1 values.
        v_data (np.ndarray): Corresponding x2 values from boundary sets V (e.g., velocity limits).
        degree (int): Degree of the polynomial to fit.

    Returns:
        A callable function representing the polynomial and its coefficients.
    """
    @staticmethod
    def create_constrained_polynomial(x1_data, v_data, degree):
        # Create the Vandermonde matrix A, where A @ c gives the polynomial values
        A = np.vander(x1_data, degree + 1)
        
        # Define the objective function for least squares with a small regularization term
        func = lambda c, A=A, v_data=v_data: np.sum((v_data - A @ c)**2) + 0.0005 * np.linalg.norm(c)**2

        # Define the linear inequality constraints: A @ c <= v_data
        # This ensures the polynomial is always at or below the data points
        A_upper = np.vander(x1_data, degree + 1)
        upper_bound_constraint = LinearConstraint(A_upper, lb=-np.inf, ub=v_data)

        # P(x1) >= 0 
        num_points = np.linspace(0, 1, 200)
        A_lower = np.vander(num_points, degree + 1)
        # We express P(x1) >= 0 as: 0 <= P(x1) <= infinity
        lower_bound_constraint = LinearConstraint(A_lower, lb=0, ub=np.inf)
        
        # Set the initial coefficients guess for the optimiser to be all zeros
        c_initial = np.zeros((degree + 1 ))#np.polyfit(x1_data, v_data, degree)
        print(f"Initial polynomial coefficients: {c_initial}")
        
        # Perform the constrained optimization
        result = minimize(func, c_initial, args=(A, v_data), method='SLSQP', constraints=[upper_bound_constraint, lower_bound_constraint])
        
        # Extract the optimal polynomial coefficients
        c_optimal = result.x
        
        # Return both the function and its coefficients
        return lambda x1: np.polyval(c_optimal, x1), c_optimal
    
