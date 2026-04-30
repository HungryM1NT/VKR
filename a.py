
import json
import numpy as np

with open("training/PCD/001/poses.json", 'r') as f:
    poses = json.load(f)
    
x = 0
y = 0
z = 0 
for pose in poses:
    p = pose['position']
    x += p['x']
    y += p['y']
    z += p['z']
    
print(x / 80)
print(y / 80)
print(z / 80)

print(poses[4]['position'])