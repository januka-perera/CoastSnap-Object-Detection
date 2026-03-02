import json
import utm
import numpy as np



def load_camera_pos_utm(keypoints_path: str) -> np.ndarray:
    """Read camera_pos from the reference keypoints JSON.

    Returns
    -------
    (3,) float64 array: [Easting, Northing, Elevation] in UTM/MGA94 metres.
    """
    with open(keypoints_path) as f:
        data = json.load(f)
    if "camera_pos" not in data:
        raise ValueError(
            f"'camera_pos' key not found in {keypoints_path}.\n"
            "Add the camera UTM position as:  \"camera_pos\": [X, Y, Z]"
        )
    return np.array(data["camera_pos"], dtype=np.float64)


camera_pos = load_camera_pos_utm("../reference/reference.json")
easting, northing, _ = camera_pos

lat, lon = utm.to_latlon(easting, northing, 56, northern=False)
