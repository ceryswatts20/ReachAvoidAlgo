import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

import HelperFunctions
from Simulator import Simulator

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
        simulator (Simulator): Instance for getting accel bounds L and U.
        C_u_coeffs (np.ndarray): The coefficients of the C_u polynomial.
        C_l_coeffs (np.ndarray): The coefficients of the C_l polynomial.
    """
    def __init__(self, C_u_func: callable, C_l_func: callable, boundarySim: Simulator, C_u_coeffs: np.ndarray, C_l_coeffs: np.ndarray):
        
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
    
    
    # Define nested helper functions for S(x) on each boundary
    def S_on_upper_boundary(self, s_val):
        x_state = np.array([s_val, self.C_u(s_val)])
    
        return self.calculate_S(x_state)

    def S_on_lower_boundary(self, s_val):
        x_state = np.array([s_val, self.C_l(s_val)])
    
        return self.calculate_S(x_state)
        
    """
    Finds the roots of S(x)=0 on the upper and lower boundaries.
    
    Args:
        x_points (np.ndarray): The discrete x1-points to check for sign changes.

    Returns:
        List containing roots on the upper and lower boundary.
    """   
    def find_S_roots(self, x_points: np.ndarray) -> list:
        # Evaluate S at all s points to find sign changes
        S_values_on_Vu = [self.S_on_upper_boundary(s) for s in x_points]
        S_values_on_Vl = [self.S_on_lower_boundary(s) for s in x_points]

        # Find roots on the upper boundary
        roots_S_upper = set()
        # Iterate through consecutive pairs of x_points to find sign changes
        for i in range(len(x_points) - 1):
            # x1 test points
            x1_1, x1_2 = x_points[i], x_points[i+1]
            # Corresponding S values
            S1, S2 = S_values_on_Vu[i], S_values_on_Vu[i+1]
            # Check for a root at x1_1
            if np.isclose(S1, 0):
                # Add the root if it's not already in the list 
                roots_S_upper.add(x1_1)
            # If there's a sign change
            elif S1 * S2 < 0:
                # Use root_scalar to find the root in the interval
                sol = root_scalar(self.S_on_upper_boundary, bracket=[x1_1, x1_2], method='brentq')
                # If the solver converged
                if sol.converged: 
                    # Add the found root
                    roots_S_upper.add(sol.root)
        # Check for a root at the last point
        if np.isclose(S_values_on_Vu[-1], 0): 
            roots_S_upper.add(x_points[-1])

        # Find roots on the lower boundary
        roots_S_lower = set()
        # Iterate through consecutive pairs of x_points to find sign changes
        for i in range(len(x_points) - 1):
            # x1 test points
            x1_1, x1_2 = x_points[i], x_points[i+1]
            # Corresponding S values at those points
            S1, S2 = S_values_on_Vl[i], S_values_on_Vl[i+1]
            # Check for a root at x1_1
            if np.isclose(S1, 0): 
                # Add the root if it's not already in the list
                roots_S_lower.add(x1_1)
            # If there's a sign change
            elif S1 * S2 < 0:
                # Use root_scalar to find the root in the interval
                sol = root_scalar(self.S_on_lower_boundary, bracket=[x1_1, x1_2], method='brentq')
                # If the solver converged
                if sol.converged: 
                    # Add the found root
                    roots_S_lower.add(sol.root)
        # Check for a root at the last point
        if np.isclose(S_values_on_Vl[-1], 0):   
            # Add the last point as a root
            roots_S_lower.add(x_points[-1])
        
        # Return sorted, unique roots
        return sorted(list(roots_S_upper.union(roots_S_lower)) + [0, 1])
    
    """
    Generates the partition I and the interval sets I_in, I_out based on the roots of S(x).
    
    Args:
        roots (list): List of roots of S(x).
        tolerance (float): Small value to check the sign of S(x) just before the root.
        
    Returns:
        A tuple containing three lists: (I_in, I_out, I).
    """
    def generate_partition_I(self, roots: list, tolerance=1e-6) -> tuple[list, list, list]:
        # Initialize lists for intervals
        I_in, I_out, I = [], [], []
        
        # Generate intervals from the roots
        for i in range(len(roots)-1):
            x1_start = roots[i]
            x1_end = roots[i+1]
            
            # Save the interval in partition I
            I.insert(0, [x1_end, x1_start])
            
            # Check the sign just before the end of the interval
            if i < len(roots) - 1:
                # Small step back from the end of the interval
                x1_before_end = x1_end - tolerance
                # Check the sign of S just before the end of the interval
                s_before = self.calculate_S(np.array([x1_before_end, self.C_u(x1_before_end)]))
                # True if S(x) <= 0, False if S(x) > 0
                interval_sign = (s_before <= 0)
                #print(f"Interval [{x1_start:.6f}, {x1_end:.6f}] - S just before end: {s_before:.6f} - S(x) <= 0: {interval_sign}")
            
            # If S(x) <= 0
            if interval_sign:
                # Add to I_in
                I_in.insert(0, [x1_end, x1_start])
            # If S(x) > 0
            else:
                # Add to I_out
                I_out.insert(0, [x1_end, x1_start])
        return I_in, I_out, I
    
    """
        Checks if a given state x_current = [x1, x2] is within the target set.
        
        Args:
            x_current (np.ndarray): The current state [x1, x2].
            X_T (tuple): The target set defined as (x1_target, x2_min, x2_max).
            
        Returns:
            bool: True if x_current is in the target set, False otherwise.
    """
    @staticmethod
    def is_in_target_set(x_current, X_T):
        x1, x2 = x_current
        target_x1, target_x2_min, target_x2_max = X_T
        
        # Check if x1 is the target set x1
        x1_condition = target_x1 == x1
        # Check if x2 is within the target set i.e less than x2_max and greater than x2_min
        x2_condition = (x2 >= target_x2_min) and (x2 <= target_x2_max)
        
        return x1_condition and x2_condition
    
    """
        Event function to stop integration when crossing a targeted x1 point.
        
        Args:
            t (float): Current time.
            x (np.ndarray): Current state vector [x1, x2].
            
        Returns:
            float: Value indicating when to stop (x1).
    """
    @staticmethod
    def event_x1_cross(t, x, x1_target):
        return x[0] - x1_target
    
    """
        Event function to stop integration when crossing x-axis.
        
        Args:
            t (float): Current time.
            x (np.ndarray): Current state vector [x1, x2].
            
        Returns:
            float: Value indicating when to stop (x2).
    """
    @staticmethod
    def event_x2_zero(t, x):
        return x[1]
    
    """
        Event function to stop integration when crossing the upper boundary.
        
        Args:
            t (float): Current time.
            x (np.ndarray): Current state vector [x1, x2].
            
        Returns:
            float: Value indicating when to stop (x2 - C_u(x1)).
    """
    @staticmethod
    def event_Cu_cross(t, x, C_u):
        # Stop when x2 - Cu(x1) = 0
        x1 = float(x[0])
        x2 = float(x[1])
        return float(x2 - C_u(x1))
    
    """
        Event function to stop integration when crossing the lower boundary.
        
        Args:
            t (float): Current time.
            x (np.ndarray): Current state vector [x1, x2].
            
        Returns:
            float: Value indicating when to stop (x2 - C_l(x1)).
    """
    @staticmethod
    def event_Cl_cross(t, x, C_l):
        # Stop when x2 - Cl(x1) = 0
        x1 = float(x[0])
        x2 = float(x[1])
        return float(x2 - C_l(x1))
    
    """
        Check if the state (x1, x2) lies on the upper boundary curve C_u(x1)
        within a tolerance.
        
        Args:
            x (np.ndarray): The state [x1, x2].
            tol (float): Tolerance for checking equality.
            
        Returns:
            bool: True if (x1, x2) is on C_u(x1), False otherwise.
    """
    def is_on_upper_boundary(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        x1, x2 = x
        cu_val = float(self.C_u(x1))
        
        # If state is above the boundary, raise error
        if x2 > cu_val:
            print(f"State (x1={x1:.6f}, x2={x2:.6f}) is above the upper boundary C_u(x1)={cu_val:.6f}.")
        # If state is within tolerance below or at the boundary, return True
        elif x2 <= cu_val and x2 >= cu_val - tol:
            return True
        
    
    """
        Check if the state (x1, x2) lies on the lower boundary curve C_l(x1)
        within a tolerance.
        
        Args:
            x (np.ndarray): The state [x1, x2].
            tol (float): Tolerance for checking equality.
            
        Returns:
            bool: True if (x1, x2) is on C_l(x1), False otherwise.
    """
    def is_on_lower_boundary(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        x1, x2 = x
        cl_val = float(self.C_l(x1))
        
        # If state is below the boundary, raise error
        if x2 < cl_val:
            print(f"State (x1={x1:.6f}, x2={x2:.6f}) is below the lower boundary C_l(x1)={cl_val:.6f}.")
        # If state is within tolerance above or at the boundary, return True
        elif x2 >= cl_val and x2 <= cl_val + tol:
            return True
        
    """
        TODO: Write method description
        
        Args:
            V (callable): The boundary function (C_u or C_l).
            x_end (float): The end of the interval.
            x_start (float): The start of the interval.
            u (int): Control input, 0 for max decceleration L, 1 for max acceleration U.
            e (float): Small value to perturb y when in I_out intervals.
            
        Returns:
            set: 
    """
    def extend(self, V: callable, x_end: float, x_start: float, u: int, e=1e-4) -> set:
        # Line 1: Inputs not defined in the method signature
        x_points = np.linspace(0, 1, 101)
        # Find roots of S(x) on both boundaries
        roots = self.find_S_roots(x_points)
        # Generate partition
        I_in, I_out, I = self.generate_partition_I(roots)
        # Initialise delta
        delta = (-1)**(u+1) * e
        # Line 3: Initialise Z and y
        # Initialise y
        y = x_start
        # Initialise output set
        Z = set()
        
        # Find intervals x_start and x_end are in
        I_start = I_end = i = 0
        for interval in I:
            # If x_start is in interval save interval position in I
            if x_start[0] >= interval[1] and x_start[0] <= interval[0]:
                I_start = I[i]
                
            # If x_end is in interval, save interval position in I
            if x_end[0] >= interval[1] and x_end[0] <= interval[0]:
                I_end = I[i]
            
        print(f"  [{x_start[0]:.6f}, {x_start[1]:.6f}]")
        print(f"  [{x_end[0]:.6f}, {x_end[1]:.6f}]")
        for interval in I[I_start:I_end]:
            print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
        # ODE setup - defined here to avoid redefining in each loop iteration
        t_span = (0.0, 10.0)
        # Define the ode function
        dynamics_func = lambda t, x, direction='backward', u=u: self.boundarySim.get_double_integrator_dynamics(t, x, direction, u)
        # Stopping conditions
        cross_x_axis = self.event_x2_zero
        leave_interval = lambda t, x, x_target=x_end: self.event_x1_cross(t, x, x_target)
        cross_Cu = lambda t, x, C_u=self.C_u: self.event_Cu_cross(t, x, C_u)
        cross_Cl = lambda t, x, C_l=self.C_l: self.event_Cl_cross(t, x, C_l)
        # Stop when any of these events occur
        cross_x_axis.terminal = True
        leave_interval.terminal = True
        cross_Cu.terminal = True
        cross_Cl.terminal = True
        # Set direction to zero to detect all crossings
        cross_x_axis.direction = 0
        leave_interval.direction = 0
        cross_Cu.direction = 0
        cross_Cl.direction = 0
        
        # Line 4:
        # Loop though each interval from x_start and x_end inclusive
        for interval in I[I_start:I_end]:
            # Line 5: If interval is in I_in
            if interval in I_in:
                print("In I_in")
                # Line 6: If y is on the upper or lower boundary
                if self.is_on_upper_boundary(y) or self.is_on_lower_boundary(y):
                    print("y on boundary")
                    # Extract part of boundary within the interval
                    V_slice = HelperFunctions.slice(V, interval)
                    # Convert from array to set
                    V_slice = set(tuple(row) for row in V_slice)
                    # Line 7: Z = Z and the slice of V over the interval
                    Z = Z.union(V_slice)
                # Line 8: y is not on with boundary
                else:
                    print("y not on boundary")
                    # Integrate backwards in time from y with control u
                    # until crossing a boundary or reaching the interval end
                    # Initial state
                    x0 = np.array([y[0], y[1]])
                    # Solve ODE
                    T_b = solve_ivp(dynamics_func, t_span, x0, method='RK45', events=[cross_x_axis, leave_interval, cross_Cu, cross_Cl], dense_output=True, rtol=1e-6, atol=1e-8)
                    T_b.message
                    # Set the last valid time before the event
                    t_end = T_b.t[-1]
                    # Create a dense time array for smooth plotting
                    t_dense = np.linspace(T_b.t[0], t_end, 500)
                    # Evaluate the solution at dense time points and transpose for easier plotting
                    # T_b = array of (x1, x2) pairs along the trajectory
                    T_b = T_b.sol(t_dense).T
                    print("T_b solved")
                    # Line 9: Extract the part of trajectory within the interval
                    T_I = HelperFunctions.slice(T_b, interval)
                    print("T_b sliced")
                    # Initialise array to hold intersection points
                    intersection_pts = np.array([])
                    # For each state in the trajectory T_I
                    for state in T_I:
                        # If state is on the boundary
                        if self.is_on_upper_boundary(state) or self.is_on_lower_boundary(state):
                            # Save intersection point
                            intersection_pts = np.vstack(intersection_pts, state)
                        
                    # Line 10: If T_I intersets with the boundary V
                    if intersection_pts.size != 0:
                        print("T_I intersects with V")
                        # Line 11: Find the left most intersection point
                        x_int = T_I[np.argmin(T_I[:, 0])]
                        # Line 12: Extract the trajectory from the intersection point to the end of the trajectory. End of trajectory = end of interval (Line 9)
                        T_int = HelperFunctions.slice(T_I, [interval[0], x_int])
                        # Line 13: Extract the boundary within interval, up to the intersection point
                        V_int = HelperFunctions.slice(V, [x_int, interval[1]])
                        # Line 14: Z = Z and T_int and V_int
                        Z = Z.union(T_int).union(V_int)
                    # Line 15: If T_I does not intersect with the boundary V
                    else:
                        print("T_I didn't intersect with V")
                        # Line 16: Z = Z and T_I
                        Z = Z.union(T_I)  
            # Line 17: If interval is in I_out
            elif interval in I_out:
                print("In I_out")
                # Line 18: If y is on the upper or lower boundary
                if self.is_on_upper_boundary(y) or self.is_on_lower_boundary(y):
                    # Line 19: y = (y1, y2 + delta)
                    y = np.array([y[0], y[1] + delta])
                    
                # Integrate backwards in time from y with control u
                # until crossing a boundary or reaching the interval end
                # Initial state
                x0 = np.array([y[0], y[1]])
                # Solve ODE
                T_b = solve_ivp(dynamics_func, t_span, x0, method='RK45', events=[cross_x_axis, leave_interval, cross_Cu, cross_Cl], dense_output=True, rtol=1e-6, atol=1e-8)
                # Set the last valid time before the event
                t_end = T_b.t[-1]
                # Create a dense time array for smooth plotting
                t_dense = np.linspace(T_b.t[0], t_end, 500)
                # Evaluate the solution at dense time points and transpose for easier plotting
                # T_b = array of (x1, x2) pairs along the trajectory
                T_b = T_b.sol(t_dense).T
                # Line 20: Extract the part of trajectory within the interval
                T_b_slice = HelperFunctions.slice(T_b, interval)
                # Convert from array to set
                T_b_slice = set(tuple(row) for row in T_b_slice)
                
                # Line 21: Z = Z and T_b_slice
                Z = Z.union(T_b_slice)
            
            print("update y")
            # Line 23:
            # Update y to the left most point of Z
            y = min(Z, key=lambda point: point[0])
            print("next iteration")
        
        # Line 24
        return Z