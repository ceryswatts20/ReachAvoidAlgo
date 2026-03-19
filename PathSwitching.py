from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from ReachAvoidSet import ReachAvoidSet
import Simulator

def find_x1(qStart, qEnd, p, theta=np.array([0, 0])):
    """Finds the value of x1 for a point p on the path defined by qStart and qEnd, with an optional theta offset.
    """
    direction = qEnd - qStart
    x1 = np.dot(p - qStart - theta, direction) / np.dot(direction, direction)
    return x1
        
def paths_intersect(qA: Callable, qB: Callable, point: np.ndarray=None):
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

def readchAvoidSetsIntersect():
    return

def valid_new_path(qStart: np.ndarray, qEnd: np.ndarray, qA: Callable, qB: Callable):
    """Checks if a switch path start and end points are reachable from the given paths.

    Args:
        qStart (np.ndarray): Proposed switched path start point (joint space).
        qEnd (np.ndarray): Proposed switched path end point (joint space).

    Returns:
        bool: True/False if the switched path start and end points are feasible.
    """
    
    # Check if the paths intersect
    if paths_intersect(qA, qB):
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
            return paths_intersect(qA, qB, qStart) and paths_intersect(qA, qB, qEnd) 
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
                    print(f"valid_new_path error: max_x2A: {max_x2A}, min_x2B: {max_x2B}")
                
            return True
        else:
            print("qStart and/or qEnd not within range of given paths:")
            print(f"\t min_x2: {min_x2}, \t max_x2: {max_x2}")
            print(f"\t qStart: {qStart}, \t qEnd: {qEnd}")
            return False
    else:
        print("Paths don't intersect.")
        return False

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
                switching_points[i]["A"] = float(find_x1(qA_start, qA_end, p))
            elif i == 2 or i == 3:
                switching_points[i]["A"] = float(find_x1(qA_start, qA_end, p, thetaA))

            if i == 1 or i == 2:
                switching_points[i]["B"] = float(find_x1(qB_start, qB_end, p))
            elif i == 3 or i == 4:
                switching_points[i]["B"] = float(find_x1(qB_start, qB_end, p, thetaB))
            
        print("Switching points on paths A and B:")
        for sp in switching_points:
            print(f"{sp['point']}: A: {sp['A']}, B: {sp['B']}")
        
        ################################################################
        ### Check points geometrically intersect in the joint space. ###
        ################################################################
        # Check if qA and qB intersect in the joint space
        if paths_intersect(qAs, qBs):
            print("Paths qA and qB intersect in the joint space.")
        # Check if proposed switched path start and end points are feasible
        if not valid_new_path(start, end, qAs, qBs):
            exit()
        # Check if the path segments geometrically interset (the switching point).
        for sp in switching_points:
            if sp["A"] is not None:
                point_on_A = qAs(sp["A"])
                if paths_intersect(qAs, qBs, point_on_A):
                    print(f"Paths qA and qB intersect at {sp['point']} on path A.")
            
            if sp["B"] is not None:
                point_on_B = qBs(sp["B"])
                if paths_intersect(qAs, qBs, point_on_B):
                    print(f"Paths qA and qB intersect at {sp['point']} on path B.")
                    
        
        
        # Calculate the reach-avoid set for given paths - used to create path segment target sets
        reachAvoidSetA = ReachAvoidSet("parameters.txt", qA_start, qA_end)
        reachAvoidSetB = ReachAvoidSet("parameters.txt", qB_start, qB_end)
        
        # Create target sets
        X_Ta = reachAvoidSetA.getTargetSet(1)
        X_Tb = reachAvoidSetB.getTargetSet(1)
        print(f"X_Ta: {X_Ta} \nX_Tb: {X_Tb}")
        X_T2a = reachAvoidSetA.getTargetSet(switching_points[2]["A"])
        X_T2b = reachAvoidSetB.getTargetSet(switching_points[2]["B"])
        print(f"X_T2a: {X_T2a} \nX_T2b: {X_T2b}")
        X_T3a = reachAvoidSetA.getTargetSet(switching_points[3]["A"])
        X_T3b = reachAvoidSetB.getTargetSet(switching_points[3]["B"])
        print(f"X_T3a: {X_T3a} \nX_T3b: {X_T3b}")
        X_T4 = [switching_points[4]["B"], 0, 22]
        X_T4b = reachAvoidSetB.getTargetSet(switching_points[4]["B"])
        print(f"X_T4b: {X_T4b}")

        
        # Path Segment 1 RAS
        R_A = reachAvoidSetA.compute(X_Ta)
        # reachAvoidSetA.plot(False, False, True, False, False, "$q_A(s)$ Velocity Limit Curves")
        reachAvoidSetA.plot(True, True, True, False, True, "$q_A(x1)$ Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$")
        
        # R_B = reachAvoidSetB.compute(X_Tb)
        # reachAvoidSetB.plot(True, True, True, True, True, "$q_B(x1)$ Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$")
        # reachAvoidSetB.plot(False, False, True, False, False, "$q_B(s)$ Velocity Limit Curves")
        
        
        # 3. geometrically intersect
        
        # 4. Calculate the reach-avoid set for the 2nd path segment
        R_2 = reachAvoidSetB.compute(X_T2b)
        # reachAvoidSetB.plot(True, True, True, False, True, "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$ for 2nd path segment")
        
        
        
        # 7. Calculate the reach-avoid set for the 3rd path segment
        # 3rd path segment RAS is path A's RAS
        # R_3 = reachAvoidSetA.compute(X_T3)
        # reachAvoidSetA.plot(True, True, True, False, True, "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$ for 3rd path segment")
        
        # 6. Do 2nd and 3rd path segments target sets interset
        
        # 7. new reach avoid sets for 2nd and 3rd path segments 
        
        
        # 10. Calculate the reach-avoid set for the 4th path segement
        # 4th path segment RAS is path B's RAS
        # R_4 = reachAvoidSetB.compute(X_T4)
        # reachAvoidSetB.plot(True, True, True, False, True, "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$ for 4th path segment")
        
        
        def control_law(x, Zu, Zl):
            x1, x2 = x
            # If x2 is approaching the lower boundary
            if x2 < Zl(x1) + 0.5:
                return 1
            # If x2 is approaching the upper boundary
            elif x2 > Zu(x1) - 0.5:
                return 0
            else:
                return (x2 - Zu(x1))/(Zl(x1) - Zu(x1))
            
        def compute_acceleration(t, x, Zu: Callable, Zl: Callable, simulator: Simulator):
            """Compute control input u(x, y(x)) = L(x) + y(x)(U(x) - L(x))."""
            
            # Get acceleration bounds
            L, U = simulator.get_accel_bounds(x[0], x[1])
            
            # Compute blending factor
            y = control_law(x, Zu, Zl)
            
            # Apply control law
            u = L + y * (U - L)
            
            return u
            
        # 12. Implement a controller to go from the start of the path switching path to the 1st switching point, p1 -> qA
        # Starting point
        x0 = qAs(switching_points[0]["A"])
        x1_pts = reachAvoidSetA._x1_star
        lipschitz_A = reachAvoidSetA._lipschitz_const
        # Create boundary functions for the upper and lower boundaries of the reach-avoid set for path A
        Z_u_func = reachAvoidSetA.simulator.create_boundary_function(R_A['Z_u'], lipschitz_A)
        Z_l_func = reachAvoidSetA.simulator.create_boundary_function(R_A['Z_l'], lipschitz_A)
        # Compute control input for the trajectory from start to p1
        u = lambda t, x: compute_acceleration(t, x, Z_u_func, Z_l_func, reachAvoidSetA.simulator)
        # Compute the trajectory from start to p1 using the computed control input
        trajectory = reachAvoidSetA._reach_calc.integrate(x0, u=u, direction="forward", x1_target=switching_points[1]["A"])
        # Plot
        plt.figure()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory from start to p1", color="blue")
        plt.legend()
        plt.grid()
        plt.show()
        
        # # 13. Implement a controller to go from the 1st switching point, p(1), i.e qB(0), to the 2nd switching point, p(2) -> qB
        # bZu, bZl = reachAvoidSetB.get_boundary_functions()
        # bL, bU = lambda x: reachAvoidSetB.simulator.get_accel_bounds(x)
        # uB = lambda x: bL(x) + control_law(x, bZu, bZl) * (bU(x) - bL(x))
        
        # 14. Implement a controller to go from the 2nd switching point, p(2), to the 3rd switching point, p(3) -> qA
        
        
        # 15. Implement a controller to go from the 3rd switching point, p(3), to the end of the path switching path -> qB
        
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")