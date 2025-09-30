import numpy as np

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
