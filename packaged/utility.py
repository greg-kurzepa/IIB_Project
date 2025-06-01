import json
import numpy as np

def numpy_dict_to_json(np_dict, filename):
    """
    Save a dictionary with numpy arrays to a JSON file.
    """
    # Convert numpy arrays to lists
    json_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in np_dict.items()}
    
    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=4)