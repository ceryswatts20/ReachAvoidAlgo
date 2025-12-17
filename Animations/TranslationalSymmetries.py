import pybullet as p
import pybullet_data
import time
import numpy as np

# connect GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# environment setup
p.setGravity(0, 0, -9.81)  # gravity along -z, perpendicular to plane
plane = p.loadURDF("plane.urdf")

# load the 2-link planar arm
robot = p.loadURDF("2LinkRevoluteNoGravity.urdf", basePosition=[0, 0, 0], useFixedBase=True)

n_joints = p.getNumJoints(robot)
print("Number of joints:", n_joints)  # expect 2

# make camera view planar
p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=45, cameraPitch=-45,
                             cameraTargetPosition=[0.5, 0, 0])

# simple joint-space trajectory: q1(t), q2(t)
t_max = 5.0
fps = 240
steps = int(t_max * fps)
t = np.linspace(0, t_max, steps)
q1 = 0.5 * np.sin(2 * np.pi * t / t_max)
q2 = 0.8 * np.sin(4 * np.pi * t / t_max)

# run animation
for i in range(steps):
    q = [q1[i], q2[i]]
    p.setJointMotorControlArray(robot,
                                jointIndices=[0, 1],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=q)
    p.stepSimulation()
    time.sleep(1 / fps)

p.disconnect()