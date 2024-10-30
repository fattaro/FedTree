import numpy as np

sv1 = np.array([0.152962, 0.151399, 0.159207, 0.137379, 0.158787, 0.142518])
sv = np.array([0.147462, 0.144061, 0.158306, 0.13191, 0.155429, 0.165083])

avg = np.mean(np.abs((sv1 - sv) / sv))

print("The average error ratio is:", avg)