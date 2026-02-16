import scipy.io
import numpy as np 
from pathlib import Path

mat_path = list(Path(".").glob("*mat"))[0]
print(f"File {mat_path}")

data = scipy.io.loadmat(str(mat_path))

meta = data["metadata"]
print("metadata fields:", meta.dtype.names)

geom = data["metadata"]["geom"][0,0]
print("geom fields:", geom.dtype.names)

gcps = data["metadata"]["gcps"][0,0]
print("gcp fields:", gcps.dtype.names)

lcp = data["metadata"]["geom"][0, 0]["lcp"][0, 0]                                                                              
print("lcp fields:", lcp.dtype.names)                                                                                          
print("NU:", lcp["NU"][0, 0].item())                                                                                           
print("NV:", lcp["NV"][0, 0].item())