import numpy as np
import os

data_path = os.path.join("test_result","malware-per_token.npy")

data = np.load(data_path)
print(data.shape)
print(data)

for i in range(np.size(data,axis=0)):
    print(data[i])