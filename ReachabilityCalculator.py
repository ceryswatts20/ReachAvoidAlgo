import numpy as np
from scipy.interpolate import interp1d

from BoundarySimulator import BoundarySimulator

"""
    Implements reachability calculations, including the S-function, based on
    the velocity limit curves C_u and C_l.
"""
class ReachabilityCalculator:
    
    """
    Initializes the class

    Args:
        C_u_func (callable): The upper velocity limit function C_u(s).
        C_l_func (callable): The lower velocity limit function C_l(s).
        simulator (Simulations): Instance for getting accel bounds L and U.
        s_data (np.ndarray): The discrete s-points used to create C_u.
    """
    def __init__(self, C_u_func: callable, C_l_func: callable, boundarySim: BoundarySimulator, s_data: np.ndarray):
        
        self.C_u = C_u_func
        self.C_l = C_l_func
        self.simulator = boundarySim
        self.s_data = s_data

        # Calculate the right-hand derivatives m_u and m_l
        self.m_l = self._create_derivative_func(self.C_l)
        self.m_u = self._create_derivative_func(self.C_u)

    
    """
    Creates a function for the right-hand derivative.
    """
    def _create_derivative_func(self, func: callable) -> callable:
        # Evaluate the given function at the discrete s points
        y_data = func(self.s_data)
        # Calculate the gradient i.e slope, at each point
        dydx = np.gradient(y_data, self.s_data)
        # Create a new interpolator for the derivative.
        # 'next' kind approximates the right-hand derivative
        deriv_func = interp1d(self.s_data, dydx, kind='next', bounds_error=True)
        
        return deriv_func


    """
    Represents the system dynamics f(x, u(x, lambda)).
    u = 0 corresponds to min acceleration, L and u = 1 to max U
    """
    def f(self, x: np.ndarray, u: int) -> np.ndarray:
        s, s_dot = x
        
        # Min acceleration dynamics (L)
        if u == 0:
            L, _ = self.simulator.get_accel_bounds(s, s_dot)
            return np.array([s_dot, L])
        # Max acceleration dynamics (U)
        elif u == 1:
            _, U = self.simulator.get_accel_bounds(s, s_dot)
            return np.array([s_dot, U])
        else:
            raise ValueError("u must be 0 or 1 for this function.")
        
    """
    Calculates S(x) from Equation (3.59).
    The state x must be on either the upper or lower velocity boundary.
    """
    def calculate_S(self, x: np.ndarray, tolerance: float = 1e-6) -> float:
        s, s_dot = x

        # Check if x is on the upper boundary (V_u)
        if np.abs(s_dot - self.C_u(s)) < tolerance:
            m_u_val = self.m_u(s)
            f_vec = self.f(x, 0)
            # S(x) = [-m_u, 1] . f(x, 0)
            S_val = -m_u_val * f_vec[0] + 1 * f_vec[1]
            
            return S_val
        # Check if x is on the lower boundary (V_l)
        elif np.abs(s_dot - self.C_l(s)) < tolerance:
            m_l_val = self.m_l(s)
            f_vec = self.f(x, 1)
            # S(x) = [m_l, -1] . f(x, 1)
            S_val = m_l_val * f_vec[0] - 1 * f_vec[1]
            
            return S_val
        # If not on a boundary, S(x) is not defined
        else:
            raise ValueError(
                f"State x = [{s:.3f}, {s_dot:.3f}] is not on the velocity "
                f"boundary (C_u(s)={self.C_u(s):.3f}, C_l(s)={self.C_l(s):.3f})."
            )
            