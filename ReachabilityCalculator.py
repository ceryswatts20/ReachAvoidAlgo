from typing import Callable

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
    Initialises the class

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
    
    def find_S_roots(self, x_points: np.ndarray) -> list:
        """
        Finds the roots of S(x)=0 on the upper and lower boundaries.
        
        Args:
            x_points (np.ndarray): The discrete x1-points to check for sign changes.

        Returns:
            tuple[list, list, list]: A tuple containing:
                - lowerRoots (list): Sorted roots found on lower boundary.
                - upperRoots (list): Sorted roots found on upper boundary.
                - roots (list): Sorted roots found on both boundaries.
        """ 
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
        
        # Convert roots to sorted lists
        lowerRoots = sorted(list(roots_S_lower) + [0, 1])
        upperRoots = sorted(list(roots_S_upper) + [0, 1])
        roots = sorted(list(set(lowerRoots + upperRoots)))
        
        # Return the roots found on the lower and upper boundary
        return lowerRoots, upperRoots, roots
    
    """
    Generates the partition I and the interval sets I_in, I_out based on the roots of S(x).
    
    Args:
        roots (list): List of roots of S(x).
        boundary (Callable): The boundary function the roots are on.
        tolerance (float): Small value to check the sign of S(x) just before the root.
        
    Returns:
        A tuple containing three lists: (I_in, I_out, I).
    """
    def generate_partition_I(self, roots: list, boundary: Callable, tolerance=1e-6) -> tuple[list, list, list]:
        # Initialise lists for intervals
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
                # s_before = self.calculate_S(np.array([x1_before_end, self.C_u(x1_before_end)]))
                s_before = self.calculate_S(np.array([x1_before_end, boundary(x1_before_end)]))
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
        if x2 > cu_val + tol:
            raise ValueError(f"is_on_upper_boundary: State (x1={x1:.6f}, x2={x2:.6f}) is above the upper boundary C_u(x1)={cu_val:.6f}.")
        # If state is within tolerance below or at the boundary, return True
        elif x2 <= cu_val + tol and x2 >= cu_val - tol:
            return True
        else:
            return False
        
    
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
        if x2 < cl_val - tol:
            raise ValueError(f"is_on_lower_boundary: State (x1={x1:.6f}, x2={x2:.6f}) is below the lower boundary C_l(x1)={cl_val:.6f}.")
        # If state is within tolerance above or at the boundary, return True
        elif x2 >= cl_val - tol and x2 <= cl_val + tol:
            return True
        else:
            return False
        
    
    
    @staticmethod
    def event_x1_cross(t, x, x1_target):
        """
        Event function to stop integration when crossing a targeted x1 point.
        
        Args:
            t (float): Current time.
            x (np.ndarray): Current state vector [x1, x2].
            
        Returns:
            float: Value indicating when to stop (x1).
        """
        return x[0] - x1_target
    
    @staticmethod
    def event_x2_zero(t, x):
        """
        Event function to stop integration when crossing x-axis.
        
        Args:
            t (float): Current time.
            x (np.ndarray): Current state vector [x1, x2].
            
        Returns:
            float: Value indicating when to stop (x2).
        """
        return x[1]
    
    @staticmethod
    def event_cross_boundary(t, x, boundary: Callable):
        """
        Event function to stop integration when crossing the upper boundary.
        
        Args:
            t (float): Current time.
            x (np.ndarray): Current state vector [x1, x2].
            
        Returns:
            float: Value indicating when to stop (x2 - boundary(x1)).
        """
        # Stop when x2 - boundary(x1) = 0
        x1 = float(x[0])
        x2 = float(x[1])
        return float(x2 - boundary(x1))
    
    def _make_events(self, x1_target: float, upper_boundary: Callable, lower_boundary: Callable):
        """
        Creates and configures ODE stop event functions.
        
        Args:
            x1_target: The x1 value at which to stop integration.
            upper_boundary (Callable):
            lower_boundary (Callable): 
        """
        
        # Stop when crossing the x-axis (x2=0)
        cross_x_axis = self.event_x2_zero
        # Stop when crossing x1_target
        cross_x1_target = lambda t, x, xt=x1_target: self.event_x1_cross(t, x, xt)
        # Stop when crossing the boundary functions
        cross_upper = lambda t, x, upper=upper_boundary: self.event_cross_boundary(t, x, upper)
        cross_lower = lambda t, x, lower=lower_boundary: self.event_cross_boundary(t, x, lower)
        
        # Set all events to be terminal and to trigger on both directions of crossing
        for event in [cross_x_axis, cross_x1_target, cross_upper, cross_lower]:
            event.terminal = True
        
        cross_x1_target.direction = 0
        # Triggers when crossing from negative to positive
        cross_upper.direction = 1
        # Triggers when crossing from positive to negative
        cross_x_axis.direction = -1
        cross_lower.direction = -1

        return [cross_x_axis, cross_x1_target, cross_upper, cross_lower]
    
    def integrate(self, x0: np.ndarray, u: int | float | Callable, events: int | float | list[float | Callable], direction: str) -> np.ndarray:
        """
        Integrates the system dynamics from x0 until an event occurs.

        Args:
            x0: Initial state [x1, x2].
            u: Control input (0=max decel L, 1=max accel U).
            events: 
            direction: 'forward' or 'backward'.

        Returns:
            Array of shape (N, 2) containing the trajectory.
        """
        
        # Define the dynamics function for integration based on the control input and direction
        dynamics = lambda t, x: self.boundarySim.get_double_integrator_dynamics(t, x, direction, u)
        # If events is a float then it is x1_target, default to passing Cu and Cl as boundaries
        if isinstance(events, float) or isinstance(events, int):
            x1_target = events
            upper_boundary = self.C_u
            lower_boundary = self.C_l
        else:
            x1_target, upper_boundary, lower_boundary = events
        # Create event functions for stopping conditions
        events = self._make_events(x1_target, upper_boundary, lower_boundary)
        # Large time span to ensure we integrate until an event occurs
        tspan = (0.0, 10.0)
        # Integrate the dynamics using solve_ivp with the defined events
        sol = solve_ivp(
            dynamics,
            tspan,
            x0,
            method="RK45",
            events=events,
            dense_output=True,
            rtol=1e-6,
            atol=1e-8,
        )
        # Create a dense time array for smooth trajectories up to the event time
        t_dense = np.linspace(sol.t[0], sol.t[-1], 500)
        
        # Return the trajectory as an array of shape (N, 2)
        return sol.sol(t_dense).T
    
    
    """
        TODO: Write method description
        
        Args:
            V (Callable): The boundary function (C_u or C_l).
            roots (list): S(x) roots found on the boundaries.
            x_end (float): The end of the interval.
            x_start (float): The start of the interval.
            u (int): Control input, 0 for max decceleration L, 1 for max acceleration U.
            e (float): Small value to perturb y when in I_out intervals.
            debug (bool): If True, print debug information during the computation.
            
        Returns:
            set: 
    """
    def extend(self, V: Callable, roots: list, x_end: float, x_start: float, u: int, e=25e-2, debug=True) -> set:
        # Line 1: Inputs not defined in the method signature
        # Generate partition
        I_in, I_out, I = self.generate_partition_I(roots, V)
        # Initialise delta
        delta = (-1)**(u+1) * e
        
        # Line 3: Initialise Z and y
        # Initialise y
        y = x_start
        # Initialise output set
        Z = set()
        # Variable to track if y updates to prevent infinite loops
        prev_y = None
        # Loop counter
        loop_counter = 0
        
        # Find intervals x_start and x_end are in
        I_start = I_end = i = 0
        for interval in I:
            # If x_start is in interval save interval position in I
            if x_start[0] >= interval[1] and x_start[0] <= interval[0]:
                I_start = i
                
            # If x_end is in interval, save interval position in I
            if x_end[0] >= interval[1] and x_end[0] <= interval[0]:
                I_end = i
            
            # Increment i
            i = i + 1
        
        if debug:
            print("Intervals from x_start to x_end")
            for interval in I[I_start:I_end+1]:
                print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
            print("I_out intervals:")
            for interval in I_out:
                print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
            print("I_in intervals:")
            for interval in I_in:
                print(f"  [{interval[0]:.6f}, {interval[1]:.6f}]")
            print(f"1: Delta: {delta:.6f}")
            print(f"3: y: {y[0]:.6f}, {y[1]:.6f}")
        
        # Line 4:
        # Loop though each interval from x_start and x_end inclusive
        for interval in I[I_start:I_end+1]:
            if debug:
                print(f"4: Interval: {interval[0]:.6f}, {interval[1]:.6f}")
            # While y is still in the interval i.e y's x1 is less than the interval's x_end and less than the interval's x_start
            # TODO: Why is there + 1e-5? 
            while y[0] <= interval[0] + 1e-3 and y[0] > interval[1] + 1e-3:
                # Line 5: If interval is in I_in
                if interval in I_in:
                    if debug:
                        print("5: In I_in")
                    # Line 6: If y is on the upper or lower boundary
                    if self.is_on_upper_boundary(y) or self.is_on_lower_boundary(y):
                        if debug:
                            if self.is_on_upper_boundary(y):
                                print("6: y on upper boundary")
                            else:
                                print("6: y on lower boundary")
                                
                        # Extract the part of boundary within the interval starting from y.
                        V_slice = HelperFunctions.slice(V, [y[0], interval[1]])
                        if V_slice.size == 0:
                            raise ValueError(f"No boundary points found in V_slice i.e no boundary points were found between the interval {interval[0]:.6f} and {interval[1]:.6f}.")
                        # Convert from array to set
                        V_slice = set(tuple(row) for row in V_slice)
                        if debug:
                            print(f"V_slice extracted with {len(V_slice)} pts")
                        
                        if debug:
                            print("7: Z union with V_slice attempted")
                        # Line 7: Z = Z and the slice of V over the interval
                        Z = Z.union(V_slice)
                        if debug:
                            print(f"7: Z union with V_slice completed with {len(Z)} pts")
                    # Line 8: y is not on with boundary
                    else:
                        if debug:
                            print("y not on boundary")
                        # Integrate backwards in time from y with control u
                        # until crossing a boundary (Cu/Cl) or reaching the interval end
                        T_b = self.integrate(y, u, interval[1], direction='backward')
                        
                        # If no trajectory points found, raise error
                        if len(T_b) == 0:
                            raise ValueError("No trajectory points found.")
                        if debug:
                            print(f"8: Trajectory integrated with {len(T_b)} pts")
                        
                        # Line 9: Extract the part of trajectory within the interval
                        T_I = HelperFunctions.slice(T_b, interval)
                        if debug:
                            print(f"9: T_I extracted with {len(T_I)} pts")
                        # If no trajectory points found in the interval, raise error
                        if len(T_I) == 0:
                            raise ValueError(f"9: No trajectory points found in the interval [{interval[0]}, {interval[1]}].")
                            
                        # Initialise array to hold intersection points
                        intersection_pts = np.array([])
                        # For each state in the trajectory T_I
                        for state in T_I:
                            # If state is on the boundary
                            if self.is_on_upper_boundary(state) or self.is_on_lower_boundary(state):
                                # If intersection_pts is empty
                                if intersection_pts.size == 0:
                                    # Initialise array
                                    intersection_pts = np.atleast_2d(state)
                                # Save intersection point
                                intersection_pts = np.vstack((intersection_pts, state))
                            
                        # Line 10: If T_I intersets with the boundary V
                        if intersection_pts.size != 0:
                            if debug:
                                print("10: T_I intersects with V")
                            # Line 11: Find the left most intersection point
                            x_int = T_I[np.argmin(T_I[:, 0])]
                            if debug:
                                print(f"11: Left most intersection point x_int: {x_int}")
                                
                            # Line 12: Extract the trajectory from the intersection point to the end of the trajectory. End of trajectory = end of interval (Line 9)
                            T_int = HelperFunctions.slice(T_I, [interval[0], x_int[0]])
                            # Convert from np.array to set
                            T_int = set(tuple(row) for row in T_int)
                            if debug:
                                print(f"12: T_int extracted with {len(T_int)} pts")
                            # If no trajectory points found in T_int, raise error
                            if len(T_int) == 0:
                                raise ValueError(f"No trajectory points found in T_int i.e no trajectory points were found between the intersection point {x_int[0]:.6f} and the end of the interval {interval[0]:.6f}.")
                            
                            # Line 13: Extract the boundary within interval, up to the intersection point
                            V_int = HelperFunctions.slice(V, [x_int[0], interval[1]])
                            # Convert from np.array to set
                            V_int = set(tuple(row) for row in V_int)
                            
                            # If no boundary points found in V_int, raise error
                            if len(V_int) == 0:
                                raise ValueError(f"No boundary points found in V_int i.e no boundary points were found between the intersection point {x_int[0]:.6f} and the start of the interval {interval[1]:.6f}.")
                            if debug:
                                print(f"13: V_int extracted with {len(V_int)} pts")
                            
                            # Line 14: Z = Z and T_int and V_int
                            if debug:
                                print("14: Z union with T_int and V_int attempted")
                            Z = Z.union(T_int).union(V_int)
                            if debug:
                                print(f"14: Z union completed with {len(Z)} pts")
                        # Line 15: If T_I does not intersect with the boundary V
                        else:
                            if debug:
                                print("15: T_I didn't intersect with V")
                                print("16:Z union with T_I attempted")
                            # Line 16: Z = Z and T_I
                            Z = Z.union(T_I)  
                            if debug:
                                print(f"16: Z union completed with {len(Z)} pts")
                # Line 17: If interval is in I_out
                elif interval in I_out:
                    if debug:
                        print("17: In I_out")
                    # Line 18: If y is on the upper or lower boundary
                    if self.is_on_upper_boundary(y) or self.is_on_lower_boundary(y):
                        if debug:
                            if self.is_on_upper_boundary(y):
                                print("18: y on upper boundary")
                            else:
                                print("18: y on lower boundary")
                        
                        # Line 19: y = (y1, y2 + delta)
                        y = np.array([y[0], y[1] + delta])
                        
                        if debug:
                            print(f"19: y updated to: {y[0]:.6f}, {y[1]:.6f}")
                    
                    # Integrate backwards in time from y with control u
                    # until crossing a boundary (Cu/Cl) or reaching the interval end
                    T_b = self.integrate(y, u, interval[1], direction='backward')
                    
                    # If no trajectory points found, raise error
                    if len(T_b) == 0:
                        raise ValueError("No trajectory points found.")
                    if debug:
                        print(f"20: Trajectory integrated with {len(T_b)} pts")
                        minT_b = min(T_b, key=lambda point: point[0])
                        print(f"Left most point of trajectory T_b: {minT_b[0]:.6f}, {minT_b[1]:.6f}")
                    
                    # Line 20: Extract the part of trajectory within the interval
                    T_b_slice = HelperFunctions.slice(T_b, interval)
                    # If no trajectory points found in the interval, raise error
                    if len(T_b_slice) == 0:
                        raise ValueError(f"20: No trajectory points found in the interval [{interval[0]}, {interval[1]}].")
                    # Convert from array to set
                    T_b_slice = set(tuple(row) for row in T_b_slice)
                    if debug:
                        print(f"20: T_b_slice extracted with {len(T_b_slice)} pts")
                        minT_b_slice = min(T_b_slice, key=lambda point: point[0])
                        print(f"Left most point of trajectory T_b_slice: {minT_b_slice[0]:.6f}, {minT_b_slice[1]:.6f}")
                    
                    # Line 21: Z = Z and T_b_slice
                    if debug:
                        print("21: Z union with T_b_slice attempted")
                    Z = Z.union(T_b_slice)
                    if debug:
                        print(f"21: Z union completed with {len(Z)} pts")
                        minZ = min(Z, key=lambda point: point[0])
                        print(f"Left most point of Z: {minZ[0]:.6f}, {minZ[1]:.6f}")
                    
                # Line 23:
                # Update y to the left most point of Z
                y = min(Z, key=lambda point: point[0])
                
                # If y is the same as the previous y, update counter, else reset counter
                if y == prev_y:
                    loop_counter += 1
                else:
                    loop_counter = 0
                prev_y = y
                # If y has not updated for 2 iterations, break to prevent infinite loop
                if loop_counter >= 2:
                    if debug:
                        print("y has not updated for 2 iterations, breaking to prevent infinite loop.")
                        return Z
                    
                    raise RuntimeError("y has not updated for 2 iterations, breaking to prevent infinite loop.")
                
                if debug:
                    print(f"23: Updated y to the left most point of Z: y: {y[0]:.6f}, {y[1]:.6f}")
        # Line 24
        return Z