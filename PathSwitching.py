from typing import Callable

from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt

from ReachAvoidSet import ReachAvoidSet

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

def arePathsParallel(qA: Callable, qB: Callable) -> bool:
    
    qA_start = qA(0)
    qA_end = qA(1)
    qB_start = qB(0)
    qB_end = qB(1)
    direction_A = qA_end - qA_start
    direction_B = qB_end - qB_start
    
    # Normalize directions
    dir_A_norm = direction_A / np.linalg.norm(direction_A)
    dir_B_norm = direction_B / np.linalg.norm(direction_B)

    # Check if parallel (dot product ≈ ±1)
    dot_product = np.dot(dir_A_norm, dir_B_norm)
    tolerance = 1e-6

    if np.abs(np.abs(dot_product) - 1) < tolerance:
        print("Paths are parallel")
        return True
    else:
        print("Paths are not parallel")
        return False
    
    
def controller(t, x, Zu: Callable, Zl: Callable, tol: float = 0.5) -> float:
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
    if x2 <= Zl(x1) + tol:
        return 1
    # If x2 is approaching the upper boundary
    elif x2 >= Zu(x1) - tol:
        return 0
    else:
        y = (x2 - Zu(x1))/(Zl(x1) - Zu(x1))
        
    # Clamp to [0, 1] to guarantee boundaries are never crossed
    return float(np.clip(y, 0.0, 1.0))

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
        
        arePathsParallel(qAs, qBs)
        arePathsParallel(qAs, qCs)
        
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
        lipschitz_A = reachAvoidSetA.lipschitz_const
        lipschitz_B = reachAvoidSetB.lipschitz_const
        # Create target sets
        X_Ta = reachAvoidSetA.getTargetSet(1)
        X_Tb = reachAvoidSetB.getTargetSet(1)
        print(f"X_Ta: {X_Ta} \nX_Tb: {X_Tb}")
        # Compute the reach avoid set
        R_A = reachAvoidSetA.compute(X_Ta)
        R_B = reachAvoidSetB.compute(X_Tb)
        # Create boundary functions for reach-avoid set A
        z_u_A, _ = reachAvoidSetA.simulator.create_boundary_function(R_A['Z_u'], lipschitz_A)
        z_l_A, _ = reachAvoidSetA.simulator.create_boundary_function(R_A['Z_l'], lipschitz_A)
        # Create boundary functions for reach-avoid set B
        z_u_B, _ = reachAvoidSetB.simulator.create_boundary_function(R_B['Z_u'], lipschitz_B)
        z_l_B, _ = reachAvoidSetB.simulator.create_boundary_function(R_B['Z_l'], lipschitz_B)
        
        reachAvoidSetA.plot(True, False, True, X_Ta, R_A, title="A")
        reachAvoidSetB.plot(True, False, True, X_Tb, R_B, title="B")
        
        
        X_T = []
        # Loop through the switching points excluding START and END
        # range(1, len(switching_points)-1) -> 1 to len(switching_points)-2
        for i in range(1, len(switching_points) - 1):
            X_Ta = reachAvoidSetA.getTargetSet(switching_points[i]["A"], [z_u_A, z_l_A])
            X_Tb = reachAvoidSetB.getTargetSet(switching_points[i]["B"], [z_u_B, z_l_B])
            print(f"X_T1a: {X_Ta} \nX_T1b: {X_Tb}")
            X_T.append(getTargetSetIntersection(switching_points[i]["A"], X_Ta, X_Tb))
        
        # X_T1a = reachAvoidSetA.getTargetSet(switching_points[1]["A"])
        # X_T1b = reachAvoidSetB.getTargetSet(switching_points[1]["B"])
        # print(f"X_T1a: {X_T1a} \nX_T1b: {X_T1b}")
        # X_T1 = getTargetSetIntersection(switching_points[1]["A"], X_T1a, X_T1b)
        # print(f"Intersecting Target Set, X_T1: {X_T1}")
        
        # X_T2a = reachAvoidSetA.getTargetSet(switching_points[2]["A"])
        # X_T2b = reachAvoidSetB.getTargetSet(switching_points[2]["B"])
        # print(f"X_T2a: {X_T2a} \nX_T2b: {X_T2b}")
        # X_T2 = getTargetSetIntersection(switching_points[2]["B"], X_T2a, X_T2b)
        # print(f"Intersecting Target Set, X_T2: {X_T2}")
        
        # X_T3a = reachAvoidSetA.getTargetSet(switching_points[3]["A"])
        # X_T3b = reachAvoidSetB.getTargetSet(switching_points[3]["B"])
        # print(f"X_T3a: {X_T3a} \nX_T3b: {X_T3b}")
        # X_T3 = getTargetSetIntersection(switching_points[3]["A"], X_T3a, X_T3b)
        # print(f"Intersecting Target Set, X_T3: {X_T3}")
        
        X_T.append(reachAvoidSetB.getTargetSet(switching_points[4]["B"]))
        print(f"X_T = {X_T}")
        
        # Path Segment 1 RAS
        R_P1 = reachAvoidSetA.compute(X_T[0])
        # Path Segment 2 RAS
        R_P2 = reachAvoidSetB.compute(X_T[1])
        # Path Segment 3 RAS
        R_P3 = reachAvoidSetA.compute(X_T[2])
        # Path Segment 4 RAS
        R_P4 = reachAvoidSetB.compute(X_T[3])
        
        # Path segments starting points, at rest
        x0_1 = [switched_path[0][0], 0]
        xEnd_1 = switched_path[0][1]
        print(f"x0_1: {x0_1} \txEnd_1: {xEnd_1}")
        x0_2 = [switched_path[1][0], 0]
        xEnd_2 = switched_path[1][1]
        print(f"x0_2: {x0_2} \txEnd_2: {xEnd_2}")
        x0_3 = [switched_path[2][0], 0]
        xEnd_3 = switched_path[2][1]
        print(f"x0_3: {x0_3} \txEnd_3: {xEnd_3}")
        x0_4 = [switched_path[3][0], 0]
        xEnd_4 = switched_path[3][1]
        print(f"x0_4: {x0_4} \txEnd_4: {xEnd_4}")
        
        
        # Implement a controller to go from the start of the path segment to the end
        # Create boundary functions for reach-avoid set for path segment 1
        z_u_1, _ = reachAvoidSetA.simulator.create_boundary_function(R_P1['Z_u'], lipschitz_A)
        z_l_1, _ = reachAvoidSetA.simulator.create_boundary_function(R_P1['Z_l'], lipschitz_A)
        # Compute control input for the trajectory
        u = lambda t, x: controller(t, x, z_u_1, z_l_1)
        events = [xEnd_1, z_u_1, z_l_1]
        # Compute the trajectory using the computed control input
        trajectory1 = reachAvoidSetA.reach_calc.integrate(x0_1, u=u, direction="forward", events=events)
        
        # Create boundary functions for reach-avoid set for path segment 2
        z_u_2, _ = reachAvoidSetB.simulator.create_boundary_function(R_P2['Z_u'], lipschitz_B)
        z_l_2, _ = reachAvoidSetB.simulator.create_boundary_function(R_P2['Z_l'], lipschitz_B)
        # Compute control input for the trajectory
        u = lambda t, x: controller(t, x, z_u_2, z_l_2)
        events = [xEnd_2, z_u_2, z_l_2]
        # Compute the trajectory using the computed control input
        trajectory2 = reachAvoidSetB.reach_calc.integrate(x0_2, u=u, direction="forward", events=events)
        
        # Create boundary functions for reach-avoid set for path segment 3
        z_u_3, _ = reachAvoidSetA.simulator.create_boundary_function(R_P3['Z_u'], lipschitz_A)
        z_l_3, _ = reachAvoidSetA.simulator.create_boundary_function(R_P3['Z_l'], lipschitz_A)
        # Compute control input for the trajectory
        u = lambda t, x: controller(t, x, z_u_3, z_l_3)
        events = [xEnd_3, z_u_3, z_l_3]
        # Compute the trajectory using the computed control input
        trajectory3 = reachAvoidSetA.reach_calc.integrate(x0_3, u=u, direction="forward", events=events)
        
        # Create boundary functions for reach-avoid set for path segment 4
        z_u_4, _ = reachAvoidSetB.simulator.create_boundary_function(R_P4['Z_u'], lipschitz_B)
        z_l_4, _ = reachAvoidSetB.simulator.create_boundary_function(R_P4['Z_l'], lipschitz_B)
        # Compute control input for the trajectory
        u = lambda t, x: controller(t, x, z_u_4, z_l_4)
        events = [xEnd_4, z_u_4, z_l_4]
        # Compute the trajectory using the computed control input
        trajectory4 = reachAvoidSetB.reach_calc.integrate(x0_4, u=u, direction="forward", events=events)
        
        # Plot all path-switching trajectories on 1 figure.
        fig, axes = plt.subplots(3, 2)
        # Reach Avoid Set A
        reachAvoidSetA.plot(True, False, False, X_Ta, R_A, title= "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T^A)$",ax=axes[0, 0])
        # Reach Avoid Set B
        reachAvoidSetB.plot(True, False, False, X_Tb, R_B, title="Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T^B)$", ax=axes[0, 1])
        # Path segment 1
        reachAvoidSetA.plot(True, False, False, X_T[0], R_P1, trajectory1, "Trajectory for Path Segment 1", axes[1, 0])
        reachAvoidSetB.plot(True, False, False, X_T[1], R_P2, trajectory2, "Trajectory for Path Segment 2", axes[1, 1])
        reachAvoidSetA.plot(True, False, False, X_T[2], R_P3, trajectory3, "Trajectory for Path Segment 3", axes[2, 0])
        reachAvoidSetB.plot(True, False, False, X_T[3], R_P4, trajectory4, "Trajectory for Path Segment 4", axes[2, 1])
        
        plt.tight_layout()
        plt.show()
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")