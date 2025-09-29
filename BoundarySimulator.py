import numpy as np
from scipy.optimize import minimize, LinearConstraint
from ManipulatorDynamics import ManipulatorDynamics

"""
Simulates the system to calculate various system boundaries
"""
class BoundarySimulator:
    """
    Initializes the VLCSimulator.
    
    Args:
        min_tau (np.ndarray): Minimum joint torques.
        max_tau (np.ndarray): Maximum joint torques.
        manipulator_dynamics_instance (ManipulatorDynamics): Instance of ManipulatorDynamics for dynamics calculations.
    """
    def __init__(self, min_tau: np.ndarray, max_tau: np.ndarray, manipulator_dynamics_instance: ManipulatorDynamics):
        
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.robot = manipulator_dynamics_instance
    
    """
    Calculates the joint acceleration bounds at a given (s, sdot)
    
    Args:
        s (float): The path parameter, [0, 1].
        sdot (float): The path velocity.
    """
    def _calculate_joint_accel_bounds(self, s: float, sdot: float) -> tuple[np.ndarray, np.ndarray]:
        lower_joint_accels = np.zeros(2)
        upper_joint_accels = np.zeros(2)
        m_s, c_s, g_s = self.robot.get_2_rev_dynamics(s, sdot)
        
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
                
        return lower_joint_accels, upper_joint_accels
    
    """
    Calculates the upper boundary function value at (s, sdot)
    
    Args:
        s (float): The path parameter, [0, 1].
        sdot (float): The path velocity.
    """
    def calculate_upper_boundary(self, s: float, sdot: float) -> float:
        l, u = self._calculate_joint_accel_bounds(s, sdot)
        L = np.max(l)
        U = np.min(u)
        vlc = L - U
        
        return vlc
    
    """
    Calculates the acceleration bounds L and U at (s, sdot)
    
    Args:
        s (float): The path parameter, [0, 1].
        sdot (float): The path velocity. 
    """
    def get_accel_bounds(self, s: float, sdot: float) -> tuple[float, float]:
        l, u = self._calculate_joint_accel_bounds(s, sdot)
        L = np.max(l)
        U = np.min(u)
        
        return L, U
    
    
    """
    Objective function for polynomial fitting.
    
    Args:
        A (np.ndarray): Vandermonde matrix.
        c (np.ndarray): Coefficients of the polynomial.
        v_data (np.ndarray): Data points to fit the polynomial to.
    """
    @staticmethod
    def _objective_func(c, A, v_data):
        return np.sum((A @ c - v_data)**2)
    
    """
    Creates a polynomial approximation P(s) for a given set, v_data, at s using
    constrained least squares, ensuring P(s_i) <= v_data[i] for all i.
    
    Args:
        s_data (np.ndarray): Discrete s values.
        v_data (np.ndarray): Corresponding sdot values from boundary sets V (e.g., velocity limits).
        degree (int): Degree of the polynomial to fit.

    Returns:
        A callable function representing the polynomial and its coefficients.
    """
    @staticmethod
    def create_constrained_polynomial(s_data, v_data, degree):
        # Create the Vandermonde matrix A, where A @ c gives the polynomial values
        A = np.vander(s_data, degree + 1)

        # Define the linear inequality constraints: A @ c <= v_data
        # This ensures the polynomial is always at or below the data points
        A_upper = np.vander(s_data, degree + 1)
        upper_bound_constraint = LinearConstraint(A_upper, lb=-np.inf, ub=v_data)

        # P(s) >= 0 
        num_points = np.linspace(0, 1, 200)
        A_lower = np.vander(num_points, degree + 1)
        # We express P(s) >= 0 as: 0 <= P(s) <= infinity
        lower_bound_constraint = LinearConstraint(A_lower, lb=0, ub=np.inf)

        # Use a simple unconstrained fit as the initial guess for the optimizer
        c_initial = np.polyfit(s_data, v_data, degree)

        # Perform the constrained optimization
        result = minimize(BoundarySimulator._objective_func, c_initial, args=(A, v_data),
            method='SLSQP', constraints=[upper_bound_constraint, lower_bound_constraint])
        
        # Extract the optimal polynomial coefficients
        c_optimal = result.x
        
        # Return both the function and its coefficients
        return lambda s: np.polyval(c_optimal, s), c_optimal
