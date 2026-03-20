from typing import Callable

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
    
    # Get the minimum x2 value from the largest minimum x2 value of each path
    min_x2 = max(min(qA(0)[1], qA(1)[1]), min(qB(0)[1], qB(1)[1]))
    # Get the maximum x2 value from the smallest maximum x2 value of each path
    max_x2 = min(max(qA(0)[1], qA(1)[1]), max(qB(0)[1], qB(1)[1]))
    
    # If a point is provided, check if the point's x2 value is within the range of the paths' x2 values
    if point is not None:
        point_x2 = point[1]
        return min_x2 <= point_x2 <= max_x2
    
    # If the minimum x2 value is less than the maximum x2 value, then the paths intersect 
    return min_x2 < max_x2

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
        # Get the minimum x2 value for each given paths
        min_x2A = min(qA(0)[1], qA(1)[1])
        min_x2B = min(qB(0)[1], qB(1)[1])
        max_x2A = max(qA(0)[1], qA(1)[1])
        max_x2B = max(qB(0)[1], qB(1)[1])
        
        # Get the minimum x2 value of all paths
        min_x2 = min(min_x2A, min_x2B)
        # Get the maximum x2 value of all paths
        max_x2 = max(max_x2A, max_x2B)
        
        # If qA and qB start and end at the same x2
        if min_x2A == min_x2B and max_x2A == max_x2B:
            # Check if qStart and qEnd are within that range
            return pathsIntersect(qA, qB, qStart) and pathsIntersect(qA, qB, qEnd) 
        # If qStart and qEnd are within the minimum and maximum x2 of the given path
        elif min_x2 <= qStart <= max_x2 and min_x2 <= qEnd <= max_x2:
            # For each point
            for pt, value in {'qStart': qStart, 'qEnd': qEnd}:
                # Check if pt is within the area only accessible by qA
                if min_x2A <= pt < min_x2B or max_x2B < pt <= max_x2A:
                    print(f"{value} is on qA")
                # Check if pt is within the area only accessible by qA
                elif min_x2B <= pt < min_x2A or max_x2A < pt <= max_x2B:
                    print(f"{value} is on qB")
                else:
                    print(f"validSwichedPath error: max_x2A: {max_x2A}, min_x2B: {max_x2B}")
                
            return True
        else:
            print("qStart and/or qEnd not within range of given paths:")
            print(f"\t min_x2: {min_x2}, \t max_x2: {max_x2}")
            print(f"\t qStart: {qStart}, \t qEnd: {qEnd}")
            return False
    else:
        print("Paths don't intersect.")
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
                    
        
        # Calculate the reach-avoid set for given paths - used to create path segment target sets
        reachAvoidSetA = ReachAvoidSet("parameters.txt", qA_start, qA_end)
        reachAvoidSetB = ReachAvoidSet("parameters.txt", qB_start, qB_end)
        reachAvoidSet1 = ReachAvoidSet("parameters.txt", qA_start, qA_end)
        reachAvoidSet2 = ReachAvoidSet("parameters.txt", qB_start, qB_end)
        reachAvoidSet3 = ReachAvoidSet("parameters.txt", qA_start, qA_end)
        reachAvoidSet4 = ReachAvoidSet("parameters.txt", qB_start, qB_end)
        
        # Create target sets
        X_Ta = reachAvoidSetA.getTargetSet(1)
        X_Tb = reachAvoidSetB.getTargetSet(1)
        print(f"X_Ta: {X_Ta} \nX_Tb: {X_Tb}")
        
        X_T1a = reachAvoidSetA.getTargetSet(switching_points[1]["A"])
        X_T1b = reachAvoidSetB.getTargetSet(switching_points[1]["B"])
        print(f"X_T1a: {X_T1a} \nX_T1b: {X_T1b}")
        X_T1 = getTargetSetIntersection(switching_points[1]["A"], X_T1a, X_T1b)
        print(f"Intersecting Target Set, X_T1: {X_T1}")
        
        X_T2a = reachAvoidSetA.getTargetSet(switching_points[2]["A"])
        X_T2b = reachAvoidSetB.getTargetSet(switching_points[2]["B"])
        print(f"X_T2a: {X_T2a} \nX_T2b: {X_T2b}")
        X_T2 = getTargetSetIntersection(switching_points[2]["B"], X_T2a, X_T2b)
        print(f"Intersecting Target Set, X_T2: {X_T2}")
        
        X_T3a = reachAvoidSetA.getTargetSet(switching_points[3]["A"])
        X_T3b = reachAvoidSetB.getTargetSet(switching_points[3]["B"])
        print(f"X_T3a: {X_T3a} \nX_T3b: {X_T3b}")
        X_T3 = getTargetSetIntersection(switching_points[3]["A"], X_T3a, X_T3b)
        print(f"Intersecting Target Set, X_T3: {X_T3}")
        
        X_T4 = reachAvoidSetB.getTargetSet(switching_points[4]["B"])
        print(f"X_T4: {X_T4}")

        
        # Path Segment 1 RAS
        R_P1 = reachAvoidSet1.compute(X_T1)
        # Path Segment 2 RAS
        R_P2 = reachAvoidSet2.compute(X_T2)
        # Path Segment 3 RAS
        R_P2 = reachAvoidSet3.compute(X_T3)
        # Path Segment 4 RAS
        R_P2 = reachAvoidSet4.compute(X_T4)
        
        # Path segments starting points, at rest
        # P1: start -> p1 on A
        x0_1 = [switching_points[0]["A"], 0]
        xEnd_1 = switching_points[1]["A"]
        print(f"x0_1: {x0_1} \txEnd_1: {xEnd_1}")
        # P2: p1 -> p2 on B
        x0_2 = [switching_points[1]["B"], 0]
        xEnd_2 = switching_points[2]["B"]
        print(f"x0_2: {x0_2} \txEnd_2: {xEnd_2}")
        # P3: p2 -> p3 on A
        x0_3 = [switching_points[2]["A"], 0]
        xEnd_3 = switching_points[3]["A"]
        print(f"x0_3: {x0_3} \txEnd_3: {xEnd_3}")
        # P4: p3 -> end on B
        x0_4 = [switching_points[3]["B"], 0]
        xEnd_4 = switching_points[4]["B"]
        print(f"x0_4: {x0_4} \txEnd_4: {xEnd_4}")
        
        lipschitz_1 = reachAvoidSet1._lipschitz_const
        lipschitz_2 = reachAvoidSet2._lipschitz_const
        lipschitz_3 = reachAvoidSet3._lipschitz_const
        lipschitz_4 = reachAvoidSet4._lipschitz_const
        
        # 12. Implement a controller to go from the start of the path switching path to the 1st switching point, p1 -> qA
        # Create boundary functions for reach-avoid set for path segment 1
        Z_u_func, _ = reachAvoidSet1.simulator.create_boundary_function(R_P1['Z_u'], lipschitz_1)
        Z_l_func, _ = reachAvoidSet1.simulator.create_boundary_function(R_P1['Z_l'], lipschitz_1)
        # Compute control input for the trajectory from start to p1
        u = lambda t, x: controller(t, x, Z_u_func, Z_l_func)
        
        events = [xEnd_1, Z_u_func, Z_l_func]
        # Compute the trajectory from start to p1 using the computed control input
        trajectory = reachAvoidSet1.reach_calc.integrate(x0_1, u=u, direction="forward", events=events)
        reachAvoidSet1.plot(True, True, True, False, False, trajectory=trajectory, title="Trajectory for Path Segment 1")
        
        # # 13. Implement a controller to go from the 1st switching point, p(1), i.e qB(0), to the 2nd switching point, p(2) -> qB
        # Create boundary functions for reach-avoid set for path segment 1
        Z_u_func, _ = reachAvoidSet2.simulator.create_boundary_function(R_P2['Z_u'], lipschitz_2)
        Z_l_func, _ = reachAvoidSet2.simulator.create_boundary_function(R_P2['Z_l'], lipschitz_2)
        # Compute control input for the trajectory from start to p1
        u = lambda t, x: controller(t, x, Z_u_func, Z_l_func)
        
        events = [xEnd_2, Z_u_func, Z_l_func]
        # Compute the trajectory from start to p1 using the computed control input
        trajectory = reachAvoidSet2.reach_calc.integrate(x0_2, u=u, direction="forward", events=events)
        reachAvoidSet2.plot(True, True, True, False, False, trajectory=trajectory, title="Trajectory for Path Segment 2")
        
        # 14. Implement a controller to go from the 2nd switching point, p(2), to the 3rd switching point, p(3) -> qA
        Z_u_func, _ = reachAvoidSet3.simulator.create_boundary_function(R_P2['Z_u'], lipschitz_3)
        Z_l_func, _ = reachAvoidSet3.simulator.create_boundary_function(R_P2['Z_l'], lipschitz_3)
        # Compute control input for the trajectory from start to p1
        u = lambda t, x: controller(t, x, Z_u_func, Z_l_func)
        
        events = [xEnd_3, Z_u_func, Z_l_func]
        # Compute the trajectory from start to p1 using the computed control input
        trajectory = reachAvoidSet3.reach_calc.integrate(x0_3, u=u, direction="forward", events=events)
        reachAvoidSet3.plot(True, True, True, False, False, trajectory=trajectory, title="Trajectory for Path Segment 3")
        
        # 15. Implement a controller to go from the 3rd switching point, p(3), to the end of the path switching path -> qB
        Z_u_func, _ = reachAvoidSet4.simulator.create_boundary_function(R_P2['Z_u'], lipschitz_4)
        Z_l_func, _ = reachAvoidSet4.simulator.create_boundary_function(R_P2['Z_l'], lipschitz_4)
        # Compute control input for the trajectory from start to p1
        u = lambda t, x: controller(t, x, Z_u_func, Z_l_func)
        
        events = [xEnd_4, Z_u_func, Z_l_func]
        # Compute the trajectory from start to p1 using the computed control input
        trajectory = reachAvoidSet4.reach_calc.integrate(x0_4, u=u, direction="forward", events=events)
        reachAvoidSet4.plot(True, True, True, False, False, trajectory=trajectory, title="Trajectory for Path Segment 4")
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")