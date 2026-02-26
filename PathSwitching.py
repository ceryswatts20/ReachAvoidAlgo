from re import X

import numpy as np

from ReachAvoidSet import ReachAvoidSet

if __name__ == "__main__":
    try:
        # Given paths
        qA_start = np.array([0, 0])
        qA_end = np.array([108, 108])
        qA_s = lambda s: qA_start + s * (qA_end - qA_start)
        qB_start = np.array([0, 108])
        qB_end = np.array([108, 0])
        qB_s = lambda s: qB_start + s * (qB_end - qB_start)
        
        # Desired switching start and end points
        # TODO: Convert these points to x1 value. One for each path
        start = np.array([27, 27])
        end = np.array([256.5, 13.5])
        # 1st switching point
        p1 = qA_end
        # 2nd switching point
        p2 = np.array([162, 54])
        # 3rd switching point
        p3 = np.array([189, 81])
        
        # Switching path
        qSwitch = [start, p1, p2, p3, end]
        
        # TODO: Target sets
        X_Ta = [1]
        X_Tb = [1]
        X_T2 = []
        X_T3 = []
        X_T4 = []
        
        # 1. Calculate the reach-avoid set for qA(s) (used for 1st path segment)
        reachAvoidSetA = ReachAvoidSet("parameters.txt", qA_start, qA_end, debug=False)
        R_A = reachAvoidSetA.compute(X_Ta)
        reachAvoidSetA.plot(False, False, True, True, False)
        
        # 2. Calculate the reach-avoid set for qB(s)
        # Not needed in this example
        reachAvoidSetB = ReachAvoidSet("parameters.txt", qB_start, qB_end, debug=False)
        R_B = reachAvoidSetB.compute(X_Tb)
        reachAvoidSetB.plot(False, False, True, True, False)
        
        
        # 3. Check if the 1st and 2nd path segments geometrically interset (the switching point) i.e share a point in the joint space.
        
        
        # 4. Calculate the reach-avoid set for the 2nd path segment
        reachAvoidSet2 = ReachAvoidSet("parameters.txt", qSwitch[1], qSwitch[2], debug=False)
        R_2 = reachAvoidSet2.compute(X_T2)
        reachAvoidSet2.plot(False, False, True, True, False)
        
        # 5. Check if the 1st and 2nd path segements have intersecting velocities in their reach-avoid sets at the switching point (where they geometrically intersect)
        
        
        # 6. Check if the 2nd and 3rd path segments geometrically interset (the switching point) i.e share a point in the joint space.
        
        
        # 7. Calculate the reach-avoid set for the 3rd path segment
        reachAvoidSet3 = ReachAvoidSet("parameters.txt", qSwitch[2], qSwitch[3], debug=False)
        R_3 = reachAvoidSet3.compute(X_T3)
        reachAvoidSet3.plot(False, False, True, True, False)
        
        # 8. Check if the 2nd and 3rd path segements have intersecting velocities in their reach-avoid sets at the switching point (where they geometrically intersect)
        
        
        # 9. Check if the 3rd and 4th path segments geometrically interset (the switching point) i.e share a point in the joint space.
        
        
        # 10. Calculate the reach-avoid set for the 4th path segement
        reachAvoidSet4 = ReachAvoidSet("parameters.txt", qSwitch[3], qSwitch[4], debug=False)
        R_4 = reachAvoidSet4.compute(X_T4)
        reachAvoidSet4.plot(False, False, True, True, False)
        
        # 11. Check if the 3rd and 4th path segements have intersecting velocities in their reach-avoid sets at the switching point (where they geometrically intersect)
        
        
        # 12. Implement a controller to go from the start of the path switching path to the 1st switching point, p(1), i.e qA(1)
        
        
        # 13. Implement a controller to go from the 1st switching point, p(1), i.e qB(0), to the 2nd switching point, p(2)
        
        
        # 14. Implement a controller to go from the 2nd switching point, p(2), to the 3rd switching point, p(3)
        
        
        # 15. Implement a controller to go from the 3rd switching point, p(3), to the end of the path switching path
        
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")