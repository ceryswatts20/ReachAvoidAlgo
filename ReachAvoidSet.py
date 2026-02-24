import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from ManipulatorDynamics import ManipulatorDynamics
from Simulator import Simulator
from ReachabilityCalculator import ReachabilityCalculator
import HelperFunctions


class ReachAvoidSet:
    """
    Computes the reach-avoid set R(X_T) for a robotic manipulator
    given a parameters file and a target set X_T.
    """

    def __init__(self, params_file, debug=False):
        """
        Initialises the class by loading the system parameters and initilising the dynamics, simulator and reachability classes.

        Args:
            params_file: Path to the parameters text file.
        """

        # Initialise variables to store results
        self._X_T = None
        self._R_X_T = None
        self._Z_u = None
        self._Z_l = None
        self._T_star_u_arr = None
        self._T_star_l_arr = None
        self._x_d = None
        self._x_a = None
        self._debug = debug
        (
            self._robot_type,
            self._m,
            self._L,
            self._q_start,
            self._q_end,
            self._min_tau,
            self._max_tau,
        ) = HelperFunctions.load_parameters_from_file(params_file)
        
        if self._debug:
            print("\n--- Loaded Parameters ---")
            print("Robot Type:", self._robot_type)
            print("Masses (m):", self._m)
            print("Lengths (L):", self._L)
            print("Path Start (q_start_rad):", self._q_start)
            print("Path End (q_end_rad):", self._q_end)
            print("Min Torques (min_tau):", self._min_tau)
            print("Max Torques (max_tau):", self._max_tau)

        # Check Lipschitz continuity
        # The path equation
        q_s_dot_path = self._q_end - self._q_start
        q_s = lambda s: self._q_start + s * q_s_dot_path
        # Check Lipschitz continuity of the path
        lipschitz = HelperFunctions.is_lipschitz_continuous(q_s)
        # If the path is not Lipschitz continuous, raise an error as the algorithm relies on this property
        if not lipschitz[0]:
            raise ValueError("Path is not Lipschitz continuous.")
        # Store the Lipschitz constant for later use in boundary adjustments
        self._lipschitz_const = lipschitz[1]

        # Robot Dynamics and Simulator
        self._robot_dynamics = ManipulatorDynamics(self._m, self._L, self._q_start, self._q_end, self._robot_type)
        self._simulator = Simulator(self._min_tau, self._max_tau, self._robot_dynamics)

        # Compute the VLC, V_u and V_l and the boundary functions C_u(x1) and C_l(x1)
        self._x1_star = np.linspace(0, 1, 101)
        self._setup_boundaries()

        # Reachability calculator
        self._reach_calc = ReachabilityCalculator(
            self._C_u,
            self._C_l,
            self._simulator,
            self._C_u_coeffs,
            self._C_l_coeffs,
        )

    def _setup_boundaries(self, poly_degree: int = 10):
        """
        Computes the upper and lower boundary sets V_u and V_l for the reach-avoid set, and fits polynomial functions to define the upper and lower boundary functions C_u(x1) and C_l(x1) for the reach-avoid set. C_u is adjusted to ensure safety by subtracting a margin based on the Lipschitz constant.
            
        Args:
            poly_degree: Degree of the polynomial to fit for the upper boundary C_u.
        """
        
        x1_star = self._x1_star
        # Calculate the upper boundary set
        V_u = self._simulator.calculate_upper_boundary(x1_star, 30.0)
        # Set lower boundary set to zero
        V_l = np.zeros_like(V_u)
        # Fit polynomial to upper boundary set to get C_u(x1)
        _, C_u_coeffs = Simulator.create_constrained_polynomial(x1_star, V_u, degree=poly_degree)

        # Safety margin based on Lipschitz constant
        safety_diff = self._lipschitz_const * (x1_star[1] - x1_star[0]) / 2
        # Subtract safety margin from the constant term of the polynomial to ensure C_u is below the upper boundary by at least the safety margin
        C_u_coeffs_adjusted = C_u_coeffs.copy()
        C_u_coeffs_adjusted[-1] -= safety_diff
        # Create the adjusted C_u function with the safety margin
        self._C_u = lambda x1: np.polyval(C_u_coeffs_adjusted, x1)
        self._C_l = lambda x1: np.zeros_like(np.asarray(x1, dtype=float))
        # Store the coefficients for later use in reachability calculations
        self._C_u_coeffs = C_u_coeffs_adjusted
        self._C_l_coeffs = np.zeros(poly_degree + 1)
        self._V_u = V_u
        self._V_l = V_l
        self._safety_diff = safety_diff
        
        if self._debug:
            print(f"Lipschitz constant: {self._lipschitz_const:.4f}")
            print(f"Safety margin applied to C_u: {safety_diff:.6f}")

    def _make_events(self, x1_target: float):
        """
        Creates and configures ODE stop event functions.
        
        Args:
            x1_target: The x1 value at which to stop integration.
        """
        
        # Stop when crossing the x-axis (x2=0)
        cross_x_axis = ReachabilityCalculator.event_x2_zero
        # Stop when crossing x1_target
        cross_x1_target = lambda t, x, xt=x1_target: ReachabilityCalculator.event_x1_cross(t, x, xt)
        # Stop when crossing the boundary functions C_u and C_l
        cross_Cu = lambda t, x, Cu=self._C_u: ReachabilityCalculator.event_Cu_cross(t, x, Cu)
        cross_Cl = lambda t, x, Cl=self._C_l: ReachabilityCalculator.event_Cl_cross(t, x, Cl)
        
        # Set all events to be terminal and to trigger on both directions of crossing
        for event in [cross_x_axis, cross_x1_target, cross_Cu, cross_Cl]:
            event.terminal = True
            event.direction = 0

        return [cross_x_axis, cross_x1_target, cross_Cu, cross_Cl]

    def _integrate(self, x0: np.ndarray, u: int, x1_target: float, direction: str) -> np.ndarray:
        """
        Integrates the system dynamics from x0 until an event occurs.

        Args:
            x0: Initial state [x1, x2].
            u: Control input (0=max decel L, 1=max accel U).
            x1_target: x1 value at which to stop integration.
            direction: 'forward' or 'backward'.
            

        Returns:
            Array of shape (N, 2) containing the trajectory.
        """
        
        # Define the dynamics function for integration based on the control input and direction
        dynamics = lambda t, x: self._simulator.get_double_integrator_dynamics(t, x, direction, u)
        # Create event functions for stopping conditions
        events = self._make_events(x1_target)
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

    def compute(self, X_T: list) -> set:
        """
        Computes the reach-avoid set for a given target set i.e R(X_T).

        Args:
            X_T (list): Target set [x1_target, x2_min, x2_max].

        Returns:
            set: The reach-avoid set R(X_T) as a set of (x1, x2) tuples. i.e (Z_l,Z_u).
        """
        
        self._X_T = X_T
        _, x2_min, x2_max = X_T
        # Initial conditions - top of the target set
        x0_u = np.array([X_T[0], x2_max])
        # Initial conditions - bottom of the target set
        x0_l = np.array([X_T[0], x2_min])

        # Backward trajectories from top and bottom of target set. Array of (x1, x2) pairs.
        T_star_u_arr = self._integrate(x0_u, u=0, x1_target=0.0, direction="backward")
        T_star_l_arr = self._integrate(x0_l, u=1, x1_target=0.0, direction="backward")
        # Store the backward trajectories for later use in plotting
        self._T_star_u_arr = T_star_u_arr
        self._T_star_l_arr = T_star_l_arr
        # Convert trajectories to sets of tuples for easier set operations
        T_star_u = set(map(tuple, T_star_u_arr))
        T_star_l = set(map(tuple, T_star_l_arr))

        # Find the leftmost points in the upper and lower trajectories to determine x_d and x_a
        x_d = min(T_star_u, key=lambda point: point[0])
        x_a = min(T_star_l, key=lambda point: point[0])
        self._x_d = x_d
        self._x_a = x_a
        
        if self._debug:
            print(f"Target set X_T: {self._X_T}")
            print(f"x_d: {x_d[0]:.6f}, {x_d[1]:.6f}")
            print(f"x_a: {x_a[0]:.6f}, {x_a[1]:.6f}")

        # Initialise the reachability calculator
        reach_calc = self._reach_calc
        # Initialise sets Z_u and Z_l
        Z_u = set()
        Z_l = set()
        # Check which boundaries x_a and x_d are on to determine how to construct Z_u and Z_l
        on_lower_a = reach_calc.is_on_lower_boundary(x_a)
        on_lower_d = reach_calc.is_on_lower_boundary(x_d)
        on_upper_a = reach_calc.is_on_upper_boundary(x_a)
        on_upper_d = reach_calc.is_on_upper_boundary(x_d)

        # If x_a and x_d is on the lower boundary
        if on_lower_a and on_lower_d:
            if self._debug:
                print("Both x_a and x_d are on the lower boundary.")
            # Z_l = T_star_l and extended trajectory
            Z_l = T_star_l.union(reach_calc.extend(self._C_l, x_d, x_a, 1))
            Z_u = T_star_u
        # If x_a and x_d are on the upper boundary
        elif on_upper_a and on_upper_d:
            if self._debug:
                print("Both x_a and x_d are on the upper boundary.")
            # Z_u = T_star_u and extended trajectory
            Z_u = T_star_u.union(reach_calc.extend(self._C_u, x_d, x_a, 0))
            Z_l = T_star_l
        else:
            if self._debug:
                print("x_a and x_d are on different boundaries.")
            
            # If x_a is on the lower boundary
            if on_lower_a:
                if self._debug:
                    print("x_a is on the lower boundary.")
                # Z_l = T_star_l and extended trajectory
                Z_l = T_star_l.union(reach_calc.extend(self._C_l, [0, 0], x_a, 1))
            else:
                if self._debug:
                    print("x_a is not on the lower boundary.")
                Z_l = T_star_l
                
            # If x_d is on the upper boundary
            if on_upper_d:
                if self._debug:
                    print("x_d is on the upper boundary.")
                # Z_u = T_star_u and extended trajectory
                Z_u = T_star_u.union(reach_calc.extend(self._C_u, [0, 0], x_d, 0))
            else:
                if self._debug:
                    print("x_d is not on the upper boundary.")
                Z_u = T_star_u

        # Save the computed sets for later use in plotting
        self._Z_u = Z_u
        self._Z_l = Z_l
        self._R_X_T = Z_l.intersection(Z_u)

        # Return the reach-avoid set as a set of (x1, x2) tuples
        return self._R_X_T

    def plot(self, show_boundaries: bool, show_intervals: bool, show_trajectories: bool):
        """
        Plots the reach-avoid set and associated curves.

        Args:
            show_boundaries (bool): Whether to plot C_u and C_l.
            show_intervals (bool): Whether to shade S(x) sign intervals.

        Raises:
            RuntimeError: If compute() has not been called yet.
        """
        
        if self._R_X_T is None:
            raise RuntimeError("Call compute() before plot().")

        X_T = self._X_T
        x1_fine = np.linspace(0, 1, 5000)

        # Create the plot and set the labels and title
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$")
        ax.grid(True)
        
        # Plot the boundary functions C_u and C_l if requested
        if show_boundaries:
            ax.plot(x1_fine, self._C_u(x1_fine), "r-", label="$C_u(x_1)$")
            ax.plot(x1_fine, self._C_l(x1_fine), "m-", label="$C_l(x_1)$")
            
        # Plot the backward trajectories from the top and bottom of the target set if requested
        if show_trajectories:
            ax.plot(self._T_star_u_arr[:, 0], self._T_star_u_arr[:, 1], "c-", label="$T^*_u$")
            ax.plot(self._T_star_l_arr[:, 0], self._T_star_l_arr[:, 1], "y-", label="$T^*_l$")
            ax.plot(*self._x_d, "ks", label="$x_d$")
            ax.plot(*self._x_a, "k^", label="$x_a$")

        # Plot shaded intervals of S(x) sign if requested
        if show_intervals:
            
            # TODO: If there are no roots skip with a debug comment 
            x1_star = self._x1_star
            # Find the roots of S(x)
            roots = self._reach_calc.find_S_roots(x1_star)
            # Generate intervals of x1 where S(x) is positive or negative based on the roots
            I_in, I_out, _ = self._reach_calc.generate_partition_I(roots)
            
            if self._debug:
                print(f"Found roots: {roots}")
                print(f"I_in: {I_in}")
                print(f"I_out: {I_out}")
            
            # Draw the vertical lines from y=0 up to the C_u curve
            ax.vlines(x=roots, ymin=0, ymax=self._C_u(np.array(roots)), colors='green', linestyles='solid', label='Roots of $S(x)$')
            
            # Shade intervals where S(x) <= 0 in light green
            # For each interval in I_in, fill between C_l and C_u
            for i, interval in enumerate(I_in):
                mask = (x1_fine >= interval[1]) & (x1_fine <= interval[0])
                ax.fill_between(
                    x1_fine,
                    self._C_l(x1_fine),
                    self._C_u(x1_fine),
                    where=mask,
                    color="lightgreen",
                    alpha=0.3,
                    # Only label the first interval to avoid duplicate labels in the legend
                    label="$S(x)\\leq 0$" if i == 0 else "",
                )
            # Shade intervals where S(x) > 0 in light coral
            for i, interval in enumerate(I_out):
                mask = (x1_fine >= interval[1]) & (x1_fine <= interval[0])
                ax.fill_between(
                    x1_fine,
                    self._C_l(x1_fine),
                    self._C_u(x1_fine),
                    where=mask,
                    color="lightcoral",
                    alpha=0.3,
                    # Only label the first interval to avoid duplicate labels in the legend
                    label="$S(x)>0$" if i == 0 else "",
                )
                
        # Target set
        ax.vlines(X_T[0], X_T[1], X_T[2], colors="orange", label="$\\mathcal{X}_T$")
        ax.plot(X_T[0], X_T[2], "ko")
        ax.plot(X_T[0], X_T[1], "ko")
        
        # Z boundaries
        Z_u_arr = np.array(sorted(self._Z_u))
        Z_l_arr = np.array(sorted(self._Z_l))
        # Extract x and y coordinates
        x1_Z_u = Z_u_arr[:, 0]
        x2_Z_u = Z_u_arr[:, 1]
        x1_Z_l = Z_l_arr[:, 0]
        x2_Z_l = Z_l_arr[:, 1]
        
        ax.plot(x1_Z_u, x2_Z_u, "m--", label="$Z_u$")
        ax.plot(x1_Z_l, x2_Z_l, "g--", label="$Z_l$")

        # Shaded reach-avoid set
        x_vals = np.unique(np.concatenate((x1_Z_u, x1_Z_l)))
        # Filter x_vals to be within 0 and the target set
        x_vals_filtered = x_vals[(x_vals >= 0) & (x_vals <= X_T[0])]
        # Create interpolation functions for the upper and lower boundaries to fill between them. Needed for ax.fill_between which requires functions or arrays defined on the same x values.
        f_u = interp1d(x1_Z_u, x2_Z_u, kind="linear", fill_value="extrapolate")
        f_l = interp1d(x1_Z_l, x2_Z_l, kind="linear", fill_value="extrapolate")
        
        # Plot the interpolated upper and lower boundaries of the reach-avoid set
        ax.plot(x_vals_filtered, f_u(x_vals_filtered), "b--", label="$Z_u$ (interpolated)")
        ax.plot(x_vals_filtered, f_l(x_vals_filtered), "r--", label="$Z_l$ (interpolated)")
        # Fill between the upper and lower boundaries to show the reach-avoid set
        ax.fill_between(
            x_vals_filtered,
            f_l(x_vals_filtered),
            f_u(x_vals_filtered),
            color="gray",
            alpha=0.75,
            label="$\\mathcal{R}(\\mathcal{X}_T)$",
        )

        # Set the legend and show the plot
        ax.legend()
        plt.tight_layout()
        plt.show()
        


if __name__ == "__main__":
    reachAvoidSet = ReachAvoidSet("parameters.txt", debug=True)
    
    X_T = [0.8, 0.05, 4]
    R_X_T = reachAvoidSet.compute(X_T)
    reachAvoidSet.plot(True, False, True)
    
    X_T_2 = [1, 0, 3]
    R_X_T_2 = reachAvoidSet.compute(X_T_2)
    reachAvoidSet.plot(True, False, True)
    