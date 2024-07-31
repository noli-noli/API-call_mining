import numpy as np
import matplotlib.pyplot as plot
from scipy.spatial import distance


cleanware = np.load("cleanware.npy")
malware = np.load("malware.npy")

cleanware_average = np.mean(cleanware[0:48],axis=0)
malware_average = np.mean(malware,axis=0)

cov = np.cov(cleanware.T)
cov_i = np.linalg.pinv(cov)

for i in range(cleanware.shape[0]-1):
    d = distance.mahalanobis(cleanware_average,cleanware[i+48],cov_i)
    print("マハラノビス距離の計算結果: %1.2f" % d)




exit()

# 散布図の描画
plot.figure()

# x軸を0から99、y軸を各テンソルの値とする
#for i in range(cleanware.shape[0]):
    #plot.plot(cleanware[i], '.', label='Cleanware', color='blue')
#    plot.plot(malware[i], '.', label='Malware' , color='red')

plot.plot(cleanware_average,color='blue')
plot.plot(malware_average,color="red")
plot.plot(malware[1],color="green")

plot.ylim(0, 1050)  # y軸の範囲
plot.xlim(0, 100)   # x軸の範囲
plot.xlabel("Index")
plot.ylabel("Value")
plot.title("Scatter Plot of Cleanware and Malware")
plot.grid(True)

# グラフを保存
plot.savefig("sample.png")