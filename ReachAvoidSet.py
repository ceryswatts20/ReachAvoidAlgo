from typing import Dict

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

    def __init__(self, params_file: str, q_start: np.ndarray, q_end: np.ndarray, debug: bool = False):
        """
        Initialises the class by loading the system parameters and initilising the dynamics, simulator and reachability classes.

        Args:
            params_file: Path to the parameters text file.
        """

        # Initialise variables to store results
        self._X_T = None
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
            _,
            _,
            self._min_tau,
            self._max_tau,
        ) = HelperFunctions.load_parameters_from_file(params_file)
        # Convert degrees to radians for joint angles
        self._q_start = np.deg2rad(q_start)
        self._q_end = np.deg2rad(q_end)
        
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
        self.lipschitz_const = lipschitz[1]

        # Robot Dynamics and Simulator
        self._robot_dynamics = ManipulatorDynamics(self._m, self._L, self._q_start, self._q_end, self._robot_type)
        self.simulator = Simulator(self._min_tau, self._max_tau, self._robot_dynamics)

        # Compute the VLC, V_u and V_l and the boundary functions C_u(x1) and C_l(x1)
        self._x1_star = np.linspace(0, 1, 101)
        self._setup_boundaries()

        # Reachability calculator
        self.reach_calc = ReachabilityCalculator(
            self._C_u,
            self._C_l,
            self.simulator,
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
        V_u = self.simulator.calculate_upper_boundary(x1_star, 30.0)
        # Set lower boundary set to zero
        V_l = np.zeros_like(V_u)
        # Fit polynomial to upper boundary set to get C_u(x1) with a safety margin based on the Lipschitz constant
        self._C_u, self._C_u_coeffs = self.simulator.create_boundary_function(V_u, self.lipschitz_const, x1_star)
        
        self._C_l = lambda x1: np.zeros_like(np.asarray(x1, dtype=float))
        self._C_l_coeffs = np.zeros(poly_degree + 1)
        self._V_u = V_u
        self._V_l = V_l
        
        if self._debug:
            print(f"Lipschitz constant: {self.lipschitz_const:.4f}")

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
        T_star_u_arr = self.reach_calc.integrate(x0_u, u=0, events=0.0, direction="backward")
        T_star_l_arr = self.reach_calc.integrate(x0_l, u=1, events=0.0, direction="backward")
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
        
        if False:
            print(f"Target set X_T: {self._X_T}")
            print(f"x_d: {x_d[0]:.6f}, {x_d[1]:.6f}")
            print(f"x_a: {x_a[0]:.6f}, {x_a[1]:.6f}")
            
            # Create the plot and set the labels and title
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_title("Backward Trajectories from Target Set")
            ax.grid(True)
            
            # Target set
            ax.vlines(X_T[0], X_T[1], X_T[2], colors="orange", label="$\\mathcal{X}_T$")
            ax.plot(X_T[0], X_T[2], "ko")
            ax.plot(X_T[0], X_T[1], "ko")
                
            # Plot the backward trajectories from the top and bottom of the target set
            ax.plot(self._T_star_u_arr[:, 0], self._T_star_u_arr[:, 1], "c-", label="$T^*_u$")
            ax.plot(self._T_star_l_arr[:, 0], self._T_star_l_arr[:, 1], "y-", label="$T^*_l$")
            ax.plot(*self._x_d, "ks", label="$x_d$")
            ax.plot(*self._x_a, "k^", label="$x_a$")
            
            # Set the legend and show the plot
            ax.legend()
            plt.tight_layout()
            plt.show()

        # Initialise the reachability calculator
        reach_calc = self.reach_calc
        # Generate partition
        _, _, roots = reach_calc.find_S_roots(self._x1_star)
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
            Z_l = T_star_l.union(reach_calc.extend(self._C_l, roots, x_d, x_a, u=1, debug=self._debug))
            Z_u = T_star_u
        # If x_a and x_d are on the upper boundary
        elif on_upper_a and on_upper_d:
            if self._debug:
                print("Both x_a and x_d are on the upper boundary.")
            # Z_u = T_star_u and extended trajectory
            Z_u = T_star_u.union(reach_calc.extend(self._C_u, roots, x_d, x_a, u=0, debug=self._debug))
            Z_l = T_star_l
        else:
            if self._debug:
                print("x_a and x_d are on different boundaries.")
            
            # If x_a is on the lower boundary
            if on_lower_a:
                if self._debug:
                    print("x_a is on the lower boundary.")
                # Z_l = T_star_l and extended trajectory
                Z_l = T_star_l.union(reach_calc.extend(self._C_l, roots, [0, 0], x_a, u=1, debug=self._debug))
            else:
                if self._debug:
                    print("x_a is not on the lower boundary.")
                Z_l = T_star_l
            
            # If x_d is on the upper boundary
            if on_upper_d:
                if self._debug:
                    print("x_d is on the upper boundary.")
                # Z_u = T_star_u and extended trajectory
                Z_u = T_star_u.union(reach_calc.extend(self._C_u, roots, [0, 0], x_d, u=0, debug=self._debug))
            else:
                if self._debug:
                    print("x_d is not on the upper boundary.")
                Z_u = T_star_u

        # Save the computed sets for later use in plotting
        self._Z_u = Z_u
        self._Z_l = Z_l

        # Return the reach-avoid set
        return {
            'Z_u': Z_u,
            'Z_l': Z_l
        }
        
    def getTargetSet(self, x1, tol=1e-2):
        """Creates the maxmimum feasible target set at a given x1 point."""
        
        # Get the minimum x2 value at x1
        min_x2 = float(self._C_l(x1)) + tol
        # Get the maximum x2 value at x1
        max_x2 = float(self._C_u(x1)) - tol
        
        return [x1, min_x2, max_x2]

    def plot(self, show_boundaries: bool, show_intervals: bool, show_trajectories: bool, X_T: list | None = None, ras : Dict[str, tuple] | None = None, trajectory: np.ndarray | None = None, title: str = "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$", ax: plt.Axes | None = None):
        """
        Plots the reach-avoid set and associated curves.

        Args:
            show_targetset (bool): Whether to plot the target set.
            show_reach_avoid (bool): Whether to plot the reach-avoid set.
            show_boundaries (bool): Whether to plot C_u and C_l.
            show_intervals (bool): Whether to shade S(x) sign intervals.

        Raises:
            RuntimeError: If compute() has not been called yet.
        """
        
        # Create a new figure if no ax is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            standalone = True
        else:
            standalone = False
        
        # if show_reach_avoid and (self._Z_l is None or self._Z_u is None):
        #     raise RuntimeError("To plot the reach-avoid set and/or the target set, call compute() before plot().")

        x1_fine = np.linspace(0, 1, 5000)

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(title)
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
            x1_star = self._x1_star
            # Find the roots of S(x)
            lower_roots, upper_roots, roots = self.reach_calc.find_S_roots(x1_star)
            # Generate intervals of x1 where S(x) is positive or negative based on the roots
            I_in_lower, I_out_lower, _ = self.reach_calc.generate_partition_I(lower_roots, self._C_l)
            I_in_upper, I_out_upper, _ = self.reach_calc.generate_partition_I(upper_roots, self._C_u)
            
            if True:
                print(f"{roots}")
                print(f"Found lower roots: {lower_roots}")
                print(f"I_in lower boundary: {I_in_lower}")
                print(f"I_out lower boundary: {I_out_lower}")
                print(f"Found upper roots: {upper_roots}")
                print(f"I_in upper boundary: {I_in_upper}")
                print(f"I_out upper boundary: {I_out_upper}")
            
            # Draw the vertical lines from y=0 to halfway to C_u curve for lower roots
            ax.vlines(x=lower_roots, ymin=0, ymax=(self._C_u(np.array(lower_roots))/2), colors='green', linestyles='solid', label='Roots of $S(x) on lower boudnary$')
            # Draw the vertical lines from halfway to C_u curve to the C_u curve for upper roots
            ax.vlines(x=upper_roots, ymax=self._C_u(np.array(upper_roots)), ymin=(self._C_u(np.array(roots))/2), colors='red', linestyles='solid', label='Roots of $S(x) on upper boudnary$')
            
            # Shade intervals where S(x) <= 0 in light green
            # For each interval in I_in_lower, fill between C_l and halfway to C_u
            for i, interval in enumerate(I_in_lower):
                mask = (x1_fine >= interval[1]) & (x1_fine <= interval[0])
                ax.fill_between(
                    x1_fine,
                    self._C_l(x1_fine),
                    self._C_u(x1_fine)/2,
                    where=mask,
                    color="lightgreen",
                    alpha=0.3,
                    # Only label the first interval to avoid duplicate labels in the legend
                    label="$S(x)\\leq 0$" if i == 0 else "",
                )
            # For each interval in I_in_upper, fill between halfway to C_u and C_u
            for i, interval in enumerate(I_in_upper):
                mask = (x1_fine >= interval[1]) & (x1_fine <= interval[0])
                ax.fill_between(
                    x1_fine,
                    self._C_u(x1_fine)/2,
                    self._C_u(x1_fine),
                    where=mask,
                    color="lightgreen",
                    alpha=0.3,
                    # Only label the first interval to avoid duplicate labels in the legend
                    label="$S(x)\\leq 0$" if i == 0 else "",
                )
            # Shade intervals where S(x) > 0 in light coral
            # For each interval in I_out_lower, fill between C_l and halfway to C_u
            for i, interval in enumerate(I_out_lower):
                mask = (x1_fine >= interval[1]) & (x1_fine <= interval[0])
                ax.fill_between(
                    x1_fine,
                    self._C_l(x1_fine),
                    self._C_u(x1_fine)/2,
                    where=mask,
                    color="lightcoral",
                    alpha=0.3,
                    # Only label the first interval to avoid duplicate labels in the legend
                    label="$S(x)>0$" if i == 0 else "",
                )
            # For each interval in I_out_upper, fill between halfway to C_u and C_u
            for i, interval in enumerate(I_out_upper):
                mask = (x1_fine >= interval[1]) & (x1_fine <= interval[0])
                ax.fill_between(
                    x1_fine,
                    self._C_u(x1_fine)/2,
                    self._C_u(x1_fine),
                    where=mask,
                    color="lightcoral",
                    alpha=0.3,
                    # Only label the first interval to avoid duplicate labels in the legend
                    label="$S(x)>0$" if i == 0 else "",
                )
        
        if X_T is not None:
            # Target set
            ax.vlines(X_T[0], X_T[1], X_T[2], colors="orange", label="$\\mathcal{X}_T$")
            ax.plot(X_T[0], X_T[2], "ko")
            ax.plot(X_T[0], X_T[1], "ko")
        
        if ras is not None:
            Z_u = ras['Z_u']
            Z_l = ras['Z_l']
            # Z boundaries as functions, to allow shading
            z_u, _ = self.simulator.create_boundary_function(Z_u, self.lipschitz_const)
            z_l, _ = self.simulator.create_boundary_function(Z_l, self.lipschitz_const)
            
            # Fill between the upper and lower boundaries to show the reach-avoid set
            mask = (x1_fine >= 0) & (x1_fine <= X_T[0])
            ax.fill_between(
                x1_fine,
                z_l(x1_fine),
                z_u(x1_fine),
                where=mask,
                color="gray",
                alpha=0.75,
                label="$\\mathcal{R}(\\mathcal{X}_T)$",
            )
        
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory", color="blue")

        # Set the legend and show the plot
        ax.legend()
        if standalone:
            plt.tight_layout()
            plt.show()
        
    

if __name__ == "__main__":
    q_start = np.array([0, 0])
    q_end = np.array([108, 108])
    reachAvoidSet = ReachAvoidSet("parameters.txt", q_start, q_end, debug=True)
    
    X_T = [1, 0, 4]
    R_X_T = reachAvoidSet.compute(X_T)
    reachAvoidSet.plot(True, True, True, True, True)

    
    qB_start = np.array([0, 108])
    qB_end = np.array([108, 0])
    reachAvoidSetB = ReachAvoidSet("parameters.txt", qB_start, qB_end, debug=True)
    X_Tb = [1, 0.05, 80]
    #R_X_Tb = reachAvoidSetB.compute(X_Tb)
    #reachAvoidSetB.plot(True, True, True, True, True)
    