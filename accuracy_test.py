import numpy as np

# base_path = "./exp_result/MC_retrain_error_a9a/MC_retrain_inner_n_6"
base_path = "./exp_result/CC_retrain_error_a9a/CC_retrain_inner_n_6"
m_values = [10, 20, 30, 40, 50]
# m_values = [30, 60, 90, 120, 150]
file_paths = [f"{base_path}_m_{m}" for m in m_values]

# 标准SV值
sv = np.array([0.150375, 0.150375, 0.150375, 0.150375, 0.150375, 0.150375])
# sv = np.array([0.147462, 0.144061, 0.158306, 0.13191, 0.155429, 0.165083])

def read_sv_file(file_path):
    sv_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if parts[0] == 'sv':
                if parts[1].endswith(','):
                    parts[1] = parts[1][:-1]
                sv_list = [float(x) for x in parts[1].split(', ')]
                sv_values = np.array(sv_list)
    return sv_values

def calculate_aer(sv_values, sv):
    return np.mean(np.abs((sv_values - sv) / sv))

# 计算每个文件的AER并存储在列表中
aer_values = []
for file_path in file_paths:
    sv_values = read_sv_file(file_path)
    aer = calculate_aer(sv_values, sv)
    aer_values.append(aer)

# 打印所有AER值
print("The average error ratios are:", aer_values)