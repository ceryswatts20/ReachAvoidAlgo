import math
import numpy as np

class ManipulatorDynamics:
    """
    Generates the parameterised Lagrangian dynamics for robotic manipulators.
    """

    GRAVITATIONAL_CONSTANT = 9.81

    def __init__(self, m, L, q_start_rad, q_end_rad):
        """
        Initializes the ManipulatorDynamics class with link masses, lengths,
        and start/end joint angles in radians.
        """
        self.m = m
        self.L = L
        # Ensure q_start and q_end are column vectors
        self.q_start = q_start_rad.reshape(-1, 1)
        self.q_end = q_end_rad.reshape(-1, 1)

    def _mass_matrix_2_rev(self, q: np.ndarray) -> np.ndarray:
        """
        Creates the mass matrix for a 2 Link, revolute joint robotic manipulator.
        """
        m1, m2 = self.m
        L1, L2 = self.L
        q1, q2 = q[0, 0], q[1, 0]

        M11 = m1 * L1**2 + m2 * (L1**2 + 2 * L1 * L2 * math.cos(q2) + L2**2)
        M12 = m2 * (L1 * L2 * math.cos(q2) + L2**2)
        M21 = M12
        M22 = m2 * L2**2

        return np.array([[M11, M12], [M21, M22]])

    def _c_vector_2_rev(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """
        Create the c vector containing the Coriolis and centripetal torques
        of a 2 Link, revolute joint robotic manipulator.
        """
        _, m2 = self.m
        L1, L2 = self.L
        q1, q2 = q[0, 0], q[1, 0]
        qdot1, qdot2 = qdot[0, 0], qdot[1, 0]

        c1 = -m2 * L1 * L2 * math.sin(q2) * (2 * qdot1 * qdot2 + qdot2**2)
        c2 = m2 * L1 * L2 * qdot1**2 * math.sin(q2)

        return np.array([[c1], [c2]])

    def _gravitational_vector_2_rev(self, q: np.ndarray) -> np.ndarray:
        """
        Creates the gravitational torque vector for a 2 Link, revolute joint robotic manipulator.
        """
        m1, m2 = self.m
        L1, L2 = self.L
        q1, q2 = q[0, 0], q[1, 0]
        g = self.GRAVITATIONAL_CONSTANT

        g1 = (m1 + m2) * L1 * g * math.cos(q1) + m2 * g * L2 * math.cos(q1 + q2)
        g2 = m2 * g * L2 * math.cos(q1 + q2)

        return np.array([[g1], [g2]])

    def get_2_rev_dynamics(self, s: float, sdot: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the path-parameterized Lagrangian dynamics for the 2 Link manipulator with revolute joints.
        """
        q_s_dot_path = self.q_end - self.q_start
        q_s = self.q_start + s * q_s_dot_path
        qdot = q_s_dot_path * sdot

        M = self._mass_matrix_2_rev(q_s)
        c = self._c_vector_2_rev(q_s, qdot)
        g = self._gravitational_vector_2_rev(q_s)

        m_s = M @ q_s_dot_path
        c_s = c
        g_s = g

        return m_s, c_s, g_s