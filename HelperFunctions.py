import numpy as np
from typing import Callable

def load_parameters_from_file(filepath='parameters.txt'):
    # Dictionary to hold extracted parameters
    parameters = {}
    current_key = None
    # Map comment headers to parameter keys
    key_map = {
        "Min joint torques": "min_torques",
        "Max joint torques": "max_torques",
        "Path start point": "q_start_deg",
        "Path end point": "q_end_deg",
        "2 DOF link masses": "m",
        "2 DOF link lengths": "L",
    }
    try:
        with open(filepath, 'r') as f:
            # For each line in the file
            for line in f:
                # Strip whitespace
                line = line.strip()
                # If the line is not empty or is a comment
                if not line or line.startswith('%'):
                    # Removes the '%' and any leading/trailing whitespace
                    header_text = line.lstrip('%').strip()
                    # Check if this header is one we care about i.e., in key_map
                    if header_text in key_map:
                        # Set the current key to the corresponding parameter name
                        current_key = key_map[header_text]
                    continue
                # If we have a current key
                if current_key:
                    try:
                        # Convert the line into a numpy array of floats
                        values = np.array([float(val.strip()) for val in line.split(',')])
                        # Store the values in the parameters dictionary
                        parameters[current_key] = values
                        # Reset current_key
                        current_key = None
                    except ValueError:
                        print(f"Warning: Could not parse values for '{current_key}' from line: '{line}'")
                        current_key = None
                else:
                    print(f"Warning: Data line '{line}' found without a preceding header comment. Skipping.")
        
        # Check if all required parameters are present
        required_keys = ["m", "L", "q_start_deg", "q_end_deg", "min_torques", "max_torques"]
        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Missing required parameter: '{key}' in '{filepath}'")

        m = parameters["m"]
        L = parameters["L"]
        # Convert degrees to radians for joint angles
        q_start_rad = np.deg2rad(parameters["q_start_deg"])
        q_end_rad = np.deg2rad(parameters["q_end_deg"])
        min_tau = parameters["min_torques"]
        max_tau = parameters["max_torques"]

        return m, L, q_start_rad, q_end_rad, min_tau, max_tau
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure it exists.")
        raise
    except Exception as e:
        print(f"An error occurred while loading parameters: {e}")
        raise


"""
    Extract a set of values from either a list of states or a function over a specified interval.
    
    Args:
        states (2D array or function): Either a 2D numpy array of states (each row is a state with two components) or a function that takes a numpy array of x1 values and returns corresponding x2 values.
        interval (list): A list with two elements [x_end, x_start] defining the interval to slice.
    
    Returns:  
"""
def slice(states, interval: list):
    # Extract interval bounds
    x_end, x_start = interval
    
    # If states is a 2D array
    if isinstance(states, np.ndarray):
        # Extract the first column
        x1 = states[:, 0]
        # Create a boolean mask:
        # Condition 1: values in x1 are greater than or equal to x_start
        # Condition 2: values in x1 are less than or equal to x_end
        mask = (x1 >= x_start) & (x1 <= x_end)
        # Use the mask to select the desired rows from the original data_array and append to Z
        return states[mask]
    # If states is a callable function
    elif isinstance(states, Callable):
        # Create a range of x1 values within the specified interval
        x1 = np.linspace(x_start, x_end, 50)
        # Compute corresponding x2 values using the provided function
        x2 = states(x1)
        # Combine x1 and x2 into a 2D array
        x = np.vstack((x1, x2)).T
        # Return points within the interval
        return x
    # If states is a set
    elif isinstance(states, set):
        pts = []
        for x in states:
            # Is x is list/tuple like with 2 items
            if isinstance(x, (list, tuple)) and len(x) == 2:
                try:
                    # Convert to float
                    x1 = float(x[0])
                    x2 = float(x[1])
                    # Add to list
                    pts.append([x1, x2])
                # Is conversion fails
                except (TypeError, ValueError):
                    # Skip
                    continue
        # If pts is empty
        if not pts:
            # Return empty array
            return np.empty((0, 2))
        # Convert pts list to an array
        arr = np.array(pts)
        # Extract the first column
        x1 = states[:, 0]
        # Create a boolean mask:
        # Condition 1: values in x1 are greater than or equal to x_start
        # Condition 2: values in x1 are less than or equal to x_end
        mask = (x1 >= x_start) & (x1 <= x_end)
        # Return the subset of points whose x1 is within the interval as a set
        return arr[mask]
    else:
        raise ValueError("The 'states' parameter must be either a list or a function.")
    
    
"""
    Numerically checks if a path q(s) is Lipschitz continuous for s = [0, 1] and returns the Lipschitz constant if found.
    Assumes the path itself is continuously differentiable.
    The method finds the maximum of the norm of the path derivative over the interval by sampling. If this supremum is finite, the path is Lipschitz continuous.

    Args:
        q_s (callable): A function `q(s)` that takes a scalar `s` and returns a NumPy array (the joint configuration vector)
        num_samples (int): The number of samples to take within the interval to approximate the maximum of the derivative's norm.
        diff_h (float): Step size for numerical derivative approximation.
        tolerance (float): Tolerance for comparing values (e.g., if a derivative appears unbounded due to numerical instability).

    Returns:
        tuple: (is_lipschitz_continuous, lipschitz_constant)
            - is_lipschitz_continuous (bool): True if the path derivative's norm is bounded (and thus Lipschitz), False otherwise.
            - lipschitz_constant (float or None): The estimated Lipschitz constant (maximum of ||q'(s)||) if Lipschitz, else None.
"""
def is_lipschitz_continuous(q_s, num_samples=1000, diff_h=1e-6, tol=1e-9) -> tuple[bool, float | None]:
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    # Numerical Derivative Function for q(s)
    # Computes q'(s) using central difference
    def dq_ds(s_val):
        q_plus_h = q_s(s_val + diff_h)
        q_minus_h = q_s(s_val - diff_h)
        return (q_plus_h - q_minus_h) / (2 * diff_h)
    

    # 3. Sample the interval to find the maximum derivative norm
    s_values = np.linspace(0, 1, num_samples)
    norms = []

    for s in s_values:
        try:
            # Calculate the Euclidean (L2) norm of the derivative vector
            norm_val = np.linalg.norm(dq_ds(s))
            norms.append(norm_val)

            # Early exit if a very large value is encountered, suggesting unboundedness
            if norm_val > 1e12: # Arbitrary large number to catch potential "infinity"
                return False, None
        except Exception as e:
            print(f"Error evaluating path at s={s}: {e}")
            # If the derivative function fails, we can't guarantee Lipschitz
            return False, None

    # Check if all norms are finite (not NaN or Inf)
    if not np.all(np.isfinite(norms)):
        return False, None

    # The Lipschitz constant is the max derivative norm
    lipschitz_constant = np.max(norms)

    # If the max norm is extremely large, it's numerically effectively unbounded
    if lipschitz_constant > 1e10: # Another arbitrary threshold for practical "unboundedness"
        return False, None

    return True, lipschitz_constant