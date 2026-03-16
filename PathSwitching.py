from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from ReachAvoidSet import ReachAvoidSet

def find_x1(q_start, q_end, p, theta=np.array([0, 0])):
    """Finds the value of x1 for a point p on the path defined by q_start and q_end, with an optional theta offset.
    """
    direction = q_end - q_start
    x1 = np.dot(p - q_start - theta, direction) / np.dot(direction, direction)
    return x1
        
def paths_intersect(qA: Callable, qB: Callable, point: np.ndarray=None):
    """Checks if two paths defined by qA and qB intersect in the joint space, assuming they can be translated along q1."""
    
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

if __name__ == "__main__":
    try:
        # Given paths
        qA_start = np.array([0, 0])
        qA_end = np.array([108, 108])
        qA_s = lambda s: qA_start + s * (qA_end - qA_start)
        qB_start = np.array([108, 108])
        qB_end = np.array([216, 0])
        # qB(s) = -qA(s) + theta (216 degrees)
        qB_s = lambda s: qB_start + s * (qB_end - qB_start)
        
        # Swithing path points in joint space
        start = np.array([27, 27])
        end = np.array([256.5, 13.5])
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
            
        # Check if qA and qB intersect in the joint space
        if paths_intersect(qA_s, qB_s):
            print("Paths qA and qB intersect in the joint space.")
            
        # Check if the path segments geometrically interset (the switching point) i.e share a point in the joint space.
        for sp in switching_points:
            if sp["A"] is not None:
                point_on_A = qA_s(sp["A"])
                if paths_intersect(qA_s, qB_s, point_on_A):
                    print(f"Paths qA and qB intersect at {sp['point']} on path A.")
            
            if sp["B"] is not None:
                point_on_B = qB_s(sp["B"])
                if paths_intersect(qA_s, qB_s, point_on_B):
                    print(f"Paths qA and qB intersect at {sp['point']} on path B.")
                    
        
        # TODO: Target sets
        X_Ta = [1, 0, 4]
        X_Tb = [1, 0.05, 80]
        X_T2 = [switching_points[2]["B"], 0, 15]
        X_T3 = [switching_points[3]["A"], 0, 4]
        X_T4 = [switching_points[4]["B"], 0, 22]

        # 1. Calculate the reach-avoid set for qA(s) (used for 1st path segment)
        reachAvoidSetA = ReachAvoidSet("parameters.txt", qA_start, qA_end)
        R_A = reachAvoidSetA.compute(X_Ta)
        reachAvoidSetA.plot(True, True, True, False, True, "$q_A(x1)$ Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$")
        #reachAvoidSetA.plot(False, False, True, False, False, "$q_A(s)$ Velocity Limit Curves")
        
        # 2. Calculate the reach-avoid set for qB(s)
        # Not needed in this example
        # reachAvoidSetB = ReachAvoidSet("parameters.txt", qB_start, qB_end, debug=False)
        # R_B = reachAvoidSetB.compute(X_Tb)
        # # Plot RAS for qB(s)
        # reachAvoidSetB.plot(True, True, True, True, True, "$q_B(x1)$ Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$")
        #reachAvoidSetB.plot(False, False, True, False, False, "$q_B(s)$ Velocity Limit Curves")
        
        
        
        # 4. Calculate the reach-avoid set for the 2nd path segment
        # 2nd path segment RAS is path B's RAS
        # R_2 = reachAvoidSetB.compute(X_T2)
        # reachAvoidSetB.plot(True, True, True, False, True, "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$ for 2nd path segment")
        
        
        
        
        # 7. Calculate the reach-avoid set for the 3rd path segment
        # 3rd path segment RAS is path A's RAS
        # R_3 = reachAvoidSetA.compute(X_T3)
        # reachAvoidSetA.plot(True, True, True, False, True, "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$ for 3rd path segment")
        
        
        # 10. Calculate the reach-avoid set for the 4th path segement
        # 4th path segment RAS is path B's RAS
        # R_4 = reachAvoidSetB.compute(X_T4)
        # reachAvoidSetB.plot(True, True, True, False, True, "Reach-Avoid Set $\\mathcal{R}(\\mathcal{X}_T)$ for 4th path segment")
        
        
        def control_law(x, Zu, Zl):
            # If x is approaching the lower boundary
            if x[1] < Zl + 0.5:
                return 1
            # If x is approaching the upper boundary
            elif x[1] > Zu - 0.5:
                return 0
            
        def compute_acceleration(x, Zu, Zl, simulator):
            """Compute control input u(x, y(x)) = L(x) + y(x)(U(x) - L(x))."""
            
            print(f"x0: {x}")
            # Get acceleration bounds
            L, U = simulator.get_accel_bounds(x[0], x[1])
            
            # Compute blending factor
            y = control_law(x, Zu, Zl)
            
            # Apply control law
            u = L + y * (U - L)
            
            return u
            
        # 12. Implement a controller to go from the start of the path switching path to the 1st switching point, p1 -> qA
        # Starting point
        x0 = qA_s(switching_points[0]["A"])
        x1_pts = reachAvoidSetA._x1_star
        # Create boundary functions for the upper and lower boundaries of the reach-avoid set for path A
        Z_u_func = reachAvoidSetA.simulator.create_boundary_function(x1_pts, R_A['Z_u'], reachAvoidSetA._lipschitz_const)
        Z_l_func = reachAvoidSetA.simulator.create_boundary_function(x1_pts, R_A['Z_l'], reachAvoidSetA._lipschitz_const)
        # Compute control input for the trajectory from start to p1
        u = compute_acceleration(x0, Z_u_func, Z_l_func, reachAvoidSetA.simulator)
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