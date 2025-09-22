import numpy as np
from scipy.optimize import root_scalar

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
        C_u_coeffs (np.ndarray): The coefficients of the C_u polynomial.
        C_l_coeffs (np.ndarray): The coefficients of the C_l polynomial.
    """
    def __init__(self, C_u_func: callable, C_l_func: callable, boundarySim: BoundarySimulator, 
                 C_u_coeffs: np.ndarray, C_l_coeffs: np.ndarray):
        
        self.C_u = C_u_func
        self.C_l = C_l_func
        self.boundarySim = boundarySim

        # Analytically derive the polynomial derivatives from the coefficients
        m_u_coeffs = np.polyder(C_u_coeffs)
        m_l_coeffs = np.polyder(C_l_coeffs)

        # Create callable functions for the derivatives
        self.m_u = lambda s: np.polyval(m_u_coeffs, s)
        self.m_l = lambda s: np.polyval(m_l_coeffs, s)


    """
    Represents the system dynamics f(x, u(x, lambda)).
    u = 0 corresponds to min acceleration, L and u = 1 to max U
    """
    def f(self, x: np.ndarray, u: int) -> np.ndarray:
        s, s_dot = x
        
        # Min acceleration dynamics (L)
        if u == 0:
            L, _ = self.boundarySim.get_accel_bounds(s, s_dot)
            return np.array([s_dot, L])
        # Max acceleration dynamics (U)
        elif u == 1:
            _, U = self.boundarySim.get_accel_bounds(s, s_dot)
            return np.array([s_dot, U])
        else:
            raise ValueError("u must be 0 or 1 for this function.")
        
    """
    Calculates S(x) from Equation (3.59).
    The state x must be on either the upper or lower velocity boundary.
    """
    def calculate_S(self, x: np.ndarray, tolerance: float = 1e-6) -> float:
        """
        Calculates S(x) based on which boundary the state x lies on.
        """
        s, s_dot = x

        # Check if x is on the upper boundary (V^u)
        if np.abs(s_dot - self.C_u(s)) < tolerance:
            m_u_val = self.m_u(s)
            # Dynamics with min acceleration L
            f_vec = self.f(x, 0)
            # S(x) = [-m_u, 1] . f(x, 0)
            return -m_u_val * f_vec[0] + 1 * f_vec[1]

        # Check if x is on the lower boundary (V^l)
        elif np.abs(s_dot - self.C_l(s)) < tolerance:
            m_l_val = self.m_l(s)
            # Dynamics with max acceleration U
            f_vec = self.f(x, 1)
            # S(x) = [m_l, -1] . f(x, 1)
            return m_l_val * f_vec[0] - 1 * f_vec[1]

        # If not on a known boundary, raise an error
        else:
            raise ValueError(
                f"State x = [{s:.3f}, {s_dot:.3f}] is not on a known velocity "
                f"boundary (C_u(s)={self.C_u(s):.3f}, C_l(s)={self.C_l(s):.3f})."
            )
    
    
    # Define nested helper functions for S(s) on each boundary
    def S_on_upper_boundary(self, s_val):
        x_state = np.array([s_val, self.C_u(s_val)])
    
        return self.calculate_S(x_state)

    def S_on_lower_boundary(self, s_val):
        x_state = np.array([s_val, self.C_l(s_val)])
    
        return self.calculate_S(x_state)
        
    """
    Finds the roots of S(x)=0 on the upper and lower boundaries.
    
    Args:
        x_points (np.ndarray): The discrete s-points to check for sign changes.

    Returns:
        A tuple containing two lists: (roots_on_upper, roots_on_lower).
    """   
    def find_S_roots(self, x_points: np.ndarray) -> tuple[list, list]:
        # Evaluate S at all s points to find sign changes
        S_values_on_Vu = [self.S_on_upper_boundary(s) for s in x_points]
        S_values_on_Vl = [self.S_on_lower_boundary(s) for s in x_points]

        # Find roots on the upper boundary
        roots_S_upper = []
        # Iterate through consecutive pairs of x_points to find sign changes
        for i in range(len(x_points) - 1):
            # x1 test points
            x1_1, x1_2 = x_points[i], x_points[i+1]
            # Corresponding S values
            S1, S2 = S_values_on_Vu[i], S_values_on_Vu[i+1]
            # Check for a root at x1_1
            if np.isclose(S1, 0):
                # Add the root if it's not already in the list 
                roots_S_upper.append(x1_1)
            # If there's a sign change
            elif S1 * S2 < 0:
                # Use root_scalar to find the root in the interval
                sol = root_scalar(self.S_on_upper_boundary, bracket=[x1_1, x1_2], method='brentq')
                # If the solver converged
                if sol.converged: 
                    # Add the found root
                    roots_S_upper.append(sol.root)
        # Check for a root at the last point
        if np.isclose(S_values_on_Vu[-1], 0): 
            roots_S_upper.append(x_points[-1])

        # Find roots on the lower boundary
        roots_S_lower = []
        # Iterate through consecutive pairs of x_points to find sign changes
        for i in range(len(x_points) - 1):
            # x1 test points
            x1_1, x1_2 = x_points[i], x_points[i+1]
            # Corresponding S values at those points
            S1, S2 = S_values_on_Vl[i], S_values_on_Vl[i+1]
            # Check for a root at x1_1
            if np.isclose(S1, 0): 
                # Add the root if it's not already in the list
                roots_S_lower.append(x1_1)
            # If there's a sign change
            elif S1 * S2 < 0:
                # Use root_scalar to find the root in the interval
                sol = root_scalar(self.S_on_lower_boundary, bracket=[x1_1, x1_2], method='brentq')
                # If the solver converged
                if sol.converged: 
                    # Add the found root
                    roots_S_lower.append(sol.root)
        # Check for a root at the last point
        if np.isclose(S_values_on_Vl[-1], 0): 
            # Add the last point as a root
            roots_S_lower.append(x_points[-1])

        # Return sorted, unique roots
        return sorted(list(set(roots_S_upper))), sorted(list(set(roots_S_lower)))
            