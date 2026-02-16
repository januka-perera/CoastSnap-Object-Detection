import scipy.io
import numpy as np 
from pathlib import Path

mat_path = list(Path(".").glob("*mat"))[0]
print(f"File {mat_path}")

data = scipy.io.loadmat(str(mat_path))

uv_coordinates = data["metadata"]["gcps"][0,0]["UVpicked"][0,0]
print(uv_coordinates)
