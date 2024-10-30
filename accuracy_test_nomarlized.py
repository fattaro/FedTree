import numpy as np

sv1 = np.array([38.7331, 50.5339, 40.4975, 45.5077, 45.3792, 44.7348])
sv = np.array([0.147462, 0.144061, 0.158306, 0.13191, 0.155429, 0.165083])

sum_sv1 = np.sum(sv1)
sum_sv = np.sum(sv)
scaler = sum_sv / sum_sv1
sv1 *= scaler
print(sv1)

avg = np.mean(np.abs((sv1 - sv) / sv))

print("The average error ratio is:", avg)