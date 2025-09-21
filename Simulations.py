import numpy as np
from ManipulatorDynamics import ManipulatorDynamics

class Simulations:
    """
    Simulates the system to calculate various system boundaries
    """
    def __init__(self, min_tau: np.ndarray, max_tau: np.ndarray, manipulator_dynamics_instance: ManipulatorDynamics):
        """
        Initializes the VLCSimulator.
        """
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.robot = manipulator_dynamics_instance

    def calculate_upper_boundary(self, s: float, sdot: float) -> float:
        """
        Calculates the upper boundary value for given path parameter s
        """
        l = np.zeros(2)
        u = np.zeros(2)

        m_s, c_s, g_s = self.robot.get_2_rev_dynamics(s, sdot)

        # Calculate lower and upper bounds for each joint
        for i in range(2):
            m_val = m_s[i, 0]
            c_val = c_s[i, 0]
            g_val = g_s[i, 0]

            if m_val > 0:
                l[i] = (self.min_tau[i] - c_val - g_val) / m_val
                u[i] = (self.max_tau[i] - c_val - g_val) / m_val
            elif m_val < 0:
                l[i] = (self.max_tau[i] - c_val - g_val) / m_val
                u[i] = (self.min_tau[i] - c_val - g_val) / m_val
            else:
                l[i] = -np.inf
                u[i] = np.inf

        L = np.max(l)
        U = np.min(u)

        #print(f"Upper boundary calculations at s={s}, sdot={sdot}: u={u}")
        vlc = L - U
        return vlc
    
    def calculate_lower_boundary(self, s: float, sdot: float) -> float:
        # L < U -> L - U = 0 for upper boundary
        # L < sddot < u -> does that mean L - sddot = 0 for lower boundary?
        return 0.0  # Placeholder implementation
