import numpy as np

sv1 = np.array([45.8444, 59.2319, 47.7185, 53.393, 54.3035, 52.1324])
sv = np.array([0.152866, 0.151207, 0.158789, 0.136996, 0.158528, 0.142256])

sum_sv1 = np.sum(sv1)
sum_sv = np.sum(sv)
scaler = sum_sv / sum_sv1
sv1 *= scaler
print(sv1)

avg = np.mean(np.abs((sv1 - sv) / sv))

print("The average error ratio is:", avg)