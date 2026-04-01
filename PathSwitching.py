import re
from typing import Callable, Dict

from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt

from ReachAvoidSet import ReachAvoidSet
import Simulator

velocity_tol = 1e-2

def findX1(qStart, qEnd, p, theta=np.array([0, 0])):
    """Finds the value of x1 for a point p on the path defined by qStart and qEnd, with an optional theta offset.
    """
    direction = qEnd - qStart
    x1 = np.dot(p - qStart - theta, direction) / np.dot(direction, direction)
    return x1
        
def pathsIntersect(qA: Callable, qB: Callable, point: np.ndarray=None):
    """Checks if two paths defined by qA and qB intersect in the joint space, assuming they can be translated along q1 and qB = -qA."""
    
    # Get the minimum q2 value from the largest minimum q2 value of each path
    min_q2 = max(min(qA(0)[1], qA(1)[1]), min(qB(0)[1], qB(1)[1]))
    # Get the maximum q2 value from the smallest maximum q2 value of each path
    max_q2 = min(max(qA(0)[1], qA(1)[1]), max(qB(0)[1], qB(1)[1]))
    
    # If a point is provided, check if the point's q2 value is within the range of the paths' q2 values
    if point is not None:
        point_q2 = point[1]
        return min_q2 <= point_q2 <= max_q2
    
    # If the minimum q2 value is less than or equal to the maximum q2 value, then the paths intersect 
    return min_q2 <= max_q2


def validSwichedPath(qStart: np.ndarray, qEnd: np.ndarray, qA: Callable, qB: Callable):
    """Checks if a switch path start and end points are reachable from the given paths.

    Args:
        qStart (np.ndarray): Proposed switched path start point (joint space).
        qEnd (np.ndarray): Proposed switched path end point (joint space).

    Returns:
        bool: True/False if the switched path start and end points are feasible.
    """
    
    # Check if the paths intersect
    if pathsIntersect(qA, qB):
        # Get the minimum q2 value for each given paths
        min_q2A = min(qA(0)[1], qA(1)[1])
        min_q2B = min(qB(0)[1], qB(1)[1])
        max_q2A = max(qA(0)[1], qA(1)[1])
        max_q2B = max(qB(0)[1], qB(1)[1])
        
        # Get the minimum q2 value of all paths
        min_q2 = min(min_q2A, min_q2B)
        # Get the maximum q2 value of all paths
        max_q2 = max(max_q2A, max_q2B)
        
        # If qA and qB start and end at the same q2
        if min_q2A == min_q2B and max_q2A == max_q2B:
            # Check if qStart and qEnd are within that range
            return pathsIntersect(qA, qB, qStart) and pathsIntersect(qA, qB, qEnd) 
        # If qStart and qEnd are within the minimum and maximum q2 of the given path
        elif min_q2 <= qStart <= max_q2 and min_q2 <= qEnd <= max_q2:
            # For each point
            for pt, value in {'qStart': qStart, 'qEnd': qEnd}:
                # Check if pt is within the area only accessible by qA
                if min_q2A <= pt < min_q2B or max_q2B < pt <= max_q2A:
                    print(f"{value} is on qA")
                # Check if pt is within the area only accessible by qA
                elif min_q2B <= pt < min_q2A or max_q2A < pt <= max_q2B:
                    print(f"{value} is on qB")
                else:
                    print(f"validSwichedPath error: max_q2A: {max_q2A}, min_q2B: {max_q2B}")
                
            return True
        else:
            print("qStart and/or qEnd not within range of given paths:")
            print(f"\t min_q2: {min_q2}, \t max_q2: {max_q2}")
            print(f"\t qStart: {qStart}, \t qEnd: {qEnd}")
            return False
    else:
        print("Paths don't intersect.")
        return False
    
def getTargetSetIntersection(x1: float | int, XTa: list[float], XTb: list[float]) -> list[float]:
    """
    Creates a new target set from the intersection of 2 given target sets.
    
    Args:
        x1 (float | int): 
        XTa (list[float]): 1st target set.
        XTb (list[float]): 2nd target set.

    Returns:
        list[float]: New target set of [x1, minx2, maxx2].
    """
    
    # Find the maximum minx2 value of the given target sets
    minx2 = max(XTa[1], XTb[1])
    # Find the minimum maxx2 value of the given target sets
    maxx2 = min(XTa[2], XTb[2])
    
    # If there is no intersection
    if minx2 > maxx2:
        raise ValueError("getTargetSetIntersection: Target sets don't intersect.")
    
    return [x1, minx2, maxx2]

def find_overlapping_velocities(qA: Callable, qB: Callable, switchingPoint: Dict, reachAvoidSetA: ReachAvoidSet, reachAvoidSetB: ReachAvoidSet, boundaries: list, tol: float = 1e-10, epsilon: float = 1e-8) -> list[float]:
    """

    Args:
        qA (Callable): Path A function that takes a scalar input and returns a point in the joint space.
        qB (Callable): Path B function that takes a scalar input and returns a point in the joint space.
        switchingPoint (Dict): The x1 values of the switching point on each path.
        reachAvoidSetA (ReachAvoidSet): Reach-avoid set for path A.
        reachAvoidSetB (ReachAvoidSet): Reach-avoid set for path B.
        tol (float): Tolerance for zero-length vectors and parallelism.
        epsilon (float): Step size for numerical differentiation.

    Returns:
        list[float]: [[x1, min_x2, max_x2], [y1, min_y2, max_y2]] for overlapping velocities.
    """
    
    x1 = switchingPoint["A"]
    y1 = switchingPoint["B"]
    z_A = boundaries[0]
    z_B = boundaries[1]
    
    # Numerical differentiation to get tangent vectors
    direction_A = (qA(x1 + epsilon) - qA(x1 - epsilon)) / (2 * epsilon)
    direction_B = (qB(y1 + epsilon) - qB(y1 - epsilon)) / (2 * epsilon)
    
    # Compute norms
    norm_A = float(np.linalg.norm(direction_A))
    norm_B = float(np.linalg.norm(direction_B))
    
    # Handle zero-length vectors: Trivially parallel.
    if norm_A < tol or norm_B < tol:
        print("One or both paths have zero-length tangent vectors at the switching point. Treating as parallel.")
        return [[x1, 0, velocity_tol], [y1, 0, velocity_tol]]
    
    # Parallel check using cosine similarity
    cos_angle = float(np.dot(direction_A, direction_B) / (norm_A * norm_B))
    
    # Check if the paths are parallel (cos_angle close to 1)
    if np.isclose(cos_angle, 1.0):
        print("Paths are parallel at this point.")
        # Get the range of x2 values at the switching point
        _, min_x2, max_x2 = reachAvoidSetA.getTargetSet(x1, z_A)
        # Get the range of y2 values at the switching point
        _, min_y2, max_y2 = reachAvoidSetB.getTargetSet(y1, z_B)

        # Compute scaling ratio k where direction_A = k * direction_B
        # Use largest component of direction_B to avoid division by near-zero values
        idx = np.argmax(np.abs(direction_B))
        k = float(direction_A[idx] / direction_B[idx])

        # Scale path A's x2 range to path B's y2 scale for comparison
        scaled_min_x2 = min_x2 * k
        scaled_max_x2 = max_x2 * k
        
        # Find overlap in unified (path B) scale
        overlap_min = max(scaled_min_x2, min_y2)
        overlap_max = min(scaled_max_x2, max_y2)

        # Check if ranges overlap
        if overlap_min > overlap_max:
            raise RuntimeError("No overlapping velocities between path A and path B at the switching point {switchingPoint}. Paths are parallel but have disjoint velocity ranges.")

        # Convert overlap back to original path coordinates
        X_Ta = [x1, overlap_min/k, overlap_max/k]
        X_Tb = [y1, overlap_min, overlap_max]
        
        return [X_Ta, X_Tb]
            
    # If cos_angle is close to -1, the paths are anti-parallel
    elif np.isclose(cos_angle, -1.0):
        print("Paths are anti-parallel at this point.")
    # If cos_angle is not close to 1 or -1, the paths are not parallel
    else:
        # Paths are not parallel, so the velocity at the switching point must be 0.
        print("Paths are not parallel at this point.")
        return [[x1, 0, velocity_tol], [y1, 0, velocity_tol]]
    
    
def basicController(t, x, Zu: Callable, Zl: Callable) -> float:
    """Compute blending factor y(x) ∈ [0, 1].

    y=1 drives the system toward the lower boundary (max accel),
    y=0 drives it toward the upper boundary (max decel).

    Args:
        x: Current state [x1, x2].
        Zu (Callable): Upper boundary function.
        Zl (Callable): Lower boundary function.
        tol (float): Tolerance band near each boundary.

    Returns:
        float: Blending factor y(x) ∈ [0, 1].
    """
    
    x1, x2 = x
    # If x2 is approaching the lower boundary
    if x2 <= Zl(x1):
        return 1
    # If x2 is approaching the upper boundary
    elif x2 >= Zu(x1):
        return 0
    else:
        y = (x2 - Zu(x1))/(Zl(x1) - Zu(x1))
        
    # Clamp to [0, 1] to guarantee boundaries are never crossed
    return float(np.clip(y, 0.0, 1.0))

def controller(t, x, x_target, Zu: Callable, Zl: Callable, sim: Simulator, tol: float = 0.5) -> float:
    """
    Blending controller that stays within boundaries and reaches x_target with x2=0.
    
    Args:
        x: Current state [x1, x2].
        x_target: Target state [x1_target, x2_target=0].
        Zu, Zl: Upper/lower boundary functions.
        L, U: Min/max acceleration bounds.
        tol: Tolerance band near boundaries.
    
    Returns:
        float: Blending factor y(x) in [0, 1].
    """
    
    # Current State
    x1, x2 = x
    # Target State
    x1_target, x2_target = x_target
    # Distance to target
    dx = x1_target - x1
    # Get acceleration bounds
    L, U = sim.get_accel_bounds(x1, x2)
    
    # Compute stopping distance needed to reach target velocity
    if L < 0:  # valid deceleration
        stopping_distance = (x2**2 - x2_target**2) / (2 * abs(L))
    else:
        stopping_distance = float('inf')
    
    # If overshooting target, decelerate
    if dx < 0:
        return 0
    
    # Priority 2: Adaptive deceleration based on boundary proximity
    zu = Zu(x1)
    zl = Zl(x1)
    
    # Distance to boundaries
    dist_to_upper = zu - x2
    dist_to_lower = x2 - zl
    
    # Compute achievable deceleration margin
    # If close to boundaries, reduce margin since controller will blend
    boundary_range = zu - zl
    if boundary_range > 0:
        normalized_dist = min(dist_to_upper, dist_to_lower) / boundary_range
        # Margin ranges from 1.3 (far from boundaries) to 1.05 (near boundaries)
        safety_margin = 1.3 - 0.25 * (1 - normalized_dist)
    else:
        safety_margin = 1.2
    
    if 0 < dx <= safety_margin * stopping_distance:
        return 0
    
    # Far from target: use boundary-based blending
    return basicController(t, x, Zu, Zl)

if __name__ == "__main__":
    try:
        # Given paths
        qA_start = np.array([0, 0])
        qA_end = np.array([108, 108])
        qAs = lambda s: qA_start + s * (qA_end - qA_start)
        qB_start = np.array([108, 108])
        qB_end = np.array([216, 0])
        # qB(s) = -qA(s) + theta (216 degrees)
        qBs = lambda s: qB_start + s * (qB_end - qB_start)
        # Dummy path used to test parallel paths method
        qCs = lambda s : qA_start + s * (qA_end/2 - qA_start)
        
        # Swithing path points in joint space
        start = np.array([27, 27])
        end = np.array([256.5, 13.5])
        
        # TODO: Find switching points automatically.
        # Desired switching points (on path A, on path B)
        # 1st switching point
        p1 = np.array([108, 108])
        # 2nd switching point
        p2 = np.array([162, 54])
        # 3rd switching point
        p3 = np.array([189, 81])
        # Switching path
        qSwitch = [start, p1, p2, p3, end]
        
        # 3rd segment: qA(s) + thetaA
        thetaA = np.array([108, 0])
        # 4th segment: qB(s) + thetaB
        thetaB = np.array([54, 0])
        
        names = ["start", "p1", "p2", "p3", "end"]
        switching_points = [{"point": name, "A": None, "B": None} for name in names]
        # Convert path switching points to psuedo-state space
        for i, p in enumerate(qSwitch):
            if i == 0 or i == 1:
                switching_points[i]["A"] = float(findX1(qA_start, qA_end, p))
            elif i == 2 or i == 3:
                switching_points[i]["A"] = float(findX1(qA_start, qA_end, p, thetaA))

            if i == 1 or i == 2:
                switching_points[i]["B"] = float(findX1(qB_start, qB_end, p))
            elif i == 3 or i == 4:
                switching_points[i]["B"] = float(findX1(qB_start, qB_end, p, thetaB))
            
        print("Switching points on paths A and B:")
        for sp in switching_points:
            print(f"{sp['point']}: A: {sp['A']}, B: {sp['B']}")
        
        ################################################################
        ### Check points geometrically intersect in the joint space. ###
        ################################################################
        # Check if qA and qB intersect in the joint space
        if pathsIntersect(qAs, qBs):
            print("Paths qA and qB intersect in the joint space.")
        # Check if proposed switched path start and end points are feasible
        if not validSwichedPath(start, end, qAs, qBs):
            exit()
        # Check if the path segments geometrically interset (the switching point).
        for sp in switching_points:
            if sp["A"] is not None:
                point_on_A = qAs(sp["A"])
                if pathsIntersect(qAs, qBs, point_on_A):
                    print(f"Paths qA and qB intersect at {sp['point']} on path A.")
            
            if sp["B"] is not None:
                point_on_B = qBs(sp["B"])
                if pathsIntersect(qAs, qBs, point_on_B):
                    print(f"Paths qA and qB intersect at {sp['point']} on path B.")
                    
        # Define path segments
        # P1: start -> p1 on A
        P1 = [switching_points[0]["A"], switching_points[1]["A"], "A"]
        # P2: p1 -> p2 on B
        P2 = [switching_points[1]["B"], switching_points[2]["B"], "B"]
        # P3: p2 -> p3 on A
        P3 = [switching_points[2]["A"], switching_points[3]["A"], "A"]
        # P4: p3 -> end on B
        P4 = [switching_points[3]["B"], switching_points[4]["B"], "B"]
        # Switched path - tuple of lists
        switched_path = [P1, P2, P3, P4]
        print(f"switched path: {switched_path}")

        # Calculate the reach-avoid set for given paths - used to create path segment target sets
        reachAvoidSetA = ReachAvoidSet("parameters.txt", qA_start, qA_end)
        reachAvoidSetB = ReachAvoidSet("parameters.txt", qB_start, qB_end)
        reachAvoidSetC = ReachAvoidSet("parameters.txt", qCs(0), qCs(1))
        lipschitz_A = reachAvoidSetA.lipschitz_const
        lipschitz_B = reachAvoidSetB.lipschitz_const
        lipschitz_C = reachAvoidSetC.lipschitz_const
        simA = reachAvoidSetA.simulator
        simB = reachAvoidSetB.simulator
        simC = reachAvoidSetC.simulator
        # Create target sets
        X_Ta = reachAvoidSetA.getTargetSet(1)
        X_Tb = reachAvoidSetB.getTargetSet(1)
        X_Tc = reachAvoidSetC.getTargetSet(1)
        print(f"X_Ta: {X_Ta} \nX_Tb: {X_Tb}")
        # Compute the reach avoid set
        R_A = reachAvoidSetA.compute(X_Ta)
        R_B = reachAvoidSetB.compute(X_Tb)
        R_C = reachAvoidSetC.compute(X_Tc)
        # Create boundary functions for reach-avoid set A
        z_u_A, _ = simA.create_boundary_function(R_A['Z_u'], True, lipschitz_A)
        z_l_A, _ = simA.create_boundary_function(R_A['Z_l'], False, lipschitz_A)
        z_A = [z_u_A, z_l_A]
        # Create boundary functions for reach-avoid set B
        z_u_B, _ = simB.create_boundary_function(R_B['Z_u'], True, lipschitz_B)
        z_l_B, _ = simB.create_boundary_function(R_B['Z_l'], False, lipschitz_B)
        z_B = [z_u_B, z_l_B]
        # Create boundary functions for reach-avoid set C
        z_u_C, _ = simC.create_boundary_function(R_C['Z_u'], True, lipschitz_C)
        z_l_C, _ = simC.create_boundary_function(R_C['Z_l'], False, lipschitz_C)
        z_C = [z_u_C, z_l_C]
        
        # Create target sets for each path segment
        X_T = {"Start": [[0,0,0]], "End": []}
        # Loop through the switching points excluding START and END
        # range(1, len(switching_points)-1) -> 1 to len(switching_points)-2
        for i in range(1, len(switching_points) - 1):
            # For each switching point, find the overlapping velocity range for the paths that intersect at that point to create the target set for the path segment.
            x_Ta, x_Tb = find_overlapping_velocities(qAs, qBs, switching_points[i], reachAvoidSetA, reachAvoidSetB, [z_A, z_B])
            # If the path segment is on path A, use X_Ta as the target set
            if switched_path[i-1][2] == "A":
                # Target set for the start of the next path segment
                X_T["Start"].append(x_Tb)
                # Target set for the end of path segment
                X_T["End"].append(x_Ta)
            # If the path segment is on path B, use X_Tb as the target set
            elif switched_path[i-1][2] == "B":
                # Target set for the start of the next path segment
                X_T["Start"].append(x_Ta)
                # Target set for the end of path segment
                X_T["End"].append(x_Tb)
            else:
                raise ValueError(f"Invalid path segment {i} path type: {switched_path[i][2]}")
            print(f"X_T{i}a: {x_Ta} \nX_T{i}b: {x_Tb}")
            
            # For testing parallel paths method
            # xta1, xtc = find_overlapping_velocities(qAs, qCs, switching_points[i], reachAvoidSetA, reachAvoidSetC, [z_A, z_C])
            # print(f"X_T{i}a1: {xta1} \nX_T{i}c: {xtc}")
            
        # Create target set for final path segment, which is always x2=0 as we want to end at rest.
        if switched_path[-1][2] == "A":
            X_T["End"].append([switching_points[4]["A"], 0, velocity_tol])
        elif switched_path[-1][2] == "B":
            X_T["End"].append([switching_points[4]["B"], 0, velocity_tol])
        else:
            raise ValueError(f"Invalid path segment {i} path type: {switched_path[i][2]}")
        print(f"X_T = {X_T}")
        
        # Plot all path-switching trajectories on 1 figure.
        fig, axes = plt.subplots(3, 2)
        # Reach Avoid Set A
        reachAvoidSetA.plot(True, False, False, X_Ta, R_A, title= "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T^A)$",ax=axes[0, 0])
        # Reach Avoid Set B
        reachAvoidSetB.plot(True, False, False, X_Tb, R_B, title="Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T^B)$", ax=axes[0, 1])
        
        # Path segments
        R_P = []
        # Path segment initial states, end states, and boundary functions
        x0 = []
        x_end = []
        # TODO: Dictionary or lists to store boundary functions???
        z_u = []
        z_l = []
        z = {'z_u': [], 'z_l': []}
        # Trajectories for each path segment
        trajectories = []
        # Compute the trajectory for each path segment
        for i in range(0, len(switched_path)):
            # Compute the initial state for each path segment, starting at minimum velocity
            x0.append([switched_path[i][0], 0])
            # Compute the end state for each path segment, at rest (x2=0)
            x_end.append([switched_path[i][1], 0])
            print(f"x0_{i+1}: {x0[i]} \tx_end_{i+1}: {x_end[i]}")
            
            # Check if the path segment is on path A or B
            # Compute the path segment reach-avoid set
            # Create boundary functions for the path segment's reach-avoid set
            if switched_path[i][2] == "A":
                R_P.append(reachAvoidSetA.compute(X_T["End"][i]))
                z_u.append(simA.create_boundary_function(R_P[i]['Z_u'], True, lipschitz_A)[0])
                z_l.append(simA.create_boundary_function(R_P[i]['Z_l'], False, lipschitz_A)[0])
            elif switched_path[i][2] == "B":
                R_P.append(reachAvoidSetB.compute(X_T["End"][i]))
                z_u.append(simB.create_boundary_function(R_P[i]['Z_u'], True, lipschitz_B)[0])
                z_l.append(simB.create_boundary_function(R_P[i]['Z_l'], False, lipschitz_B)[0])
            else:
                raise ValueError(f"Invalid path segment {i+1} path type: {switched_path[i][2]}")
            # Add to boundary function to dictionary
            z['z_u'].append(z_u[i])
            z['z_l'].append(z_l[i])
            
            # Create controller for each path segment's trajectory
            u = lambda t, x: basicController(t, x, z_u[i], z_l[i])
            # Define the stop events for each path segment's trajectory
            events = [x_end[i][0], z_u[i], z_l[i]]
            # Compute the trajectory using the computed control input
            if switched_path[i][2] == "A":
                trajectories.append(reachAvoidSetA.reach_calc.integrate(x0[i], u, events=events, direction='forward'))
            elif switched_path[i][2] == "B":
                trajectories.append(reachAvoidSetB.reach_calc.integrate(x0[i], u, events=events, direction='forward'))
            else:
                raise ValueError(f"Invalid path segment {i+1} path type: {switched_path[i][2]}")
            print(f"Trajectory {i+1} completed.")
            
            # Plot the path segment
            if switched_path[i][2] == "A":
                reachAvoidSetA.plot(True, False, False, X_T["End"][i], R_P[i], trajectories[i], f"Trajectory for Path Segment {i+1}", axes[(i//2)+1, i%2])
                # reachAvoidSetA.plot(True, False, False, X_T["End"][i], R_P[i], title=f"Trajectory for Path Segment {i+1}", ax=axes[(i//2)+1, i%2])
            elif switched_path[i][2] == "B":
                reachAvoidSetB.plot(True, False, False, X_T["End"][i], R_P[i], trajectories[i], f"Trajectory for Path Segment {i+1}", axes[(i//2)+1, i%2])
                # reachAvoidSetB.plot(True, False, False, X_T["End"][i], R_P[i], title=f"Trajectory for Path Segment {i+1}", ax=axes[(i//2)+1, i%2])
        
        plt.tight_layout()
        plt.show()
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")