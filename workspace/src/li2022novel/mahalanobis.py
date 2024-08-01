import numpy as np
import matplotlib.pyplot as plot
from scipy.spatial import distance


cleanware = np.load("cleanware-per_token.npy")
malware = np.load("malware-per_token.npy")

cleanware_average = np.mean(cleanware[0:49],axis=0)
malware_average = np.mean(malware,axis=0)

cov = np.cov(cleanware.T)
cov_i = np.linalg.pinv(cov)


distances_cleanware = []
distances_malware = []
for i in range(cleanware.shape[0]-50):
    d = distance.mahalanobis(cleanware_average,cleanware[50+i],cov_i)
    print("マハラノビス距離の計算結果: %1.2f" % d)
    distances_cleanware.append(d)

print("--------------------------------------")

for i in range(cleanware.shape[0]-50):
    d = distance.mahalanobis(cleanware_average,malware[50+i],cov_i)
    print("マハラノビス距離の計算結果: %1.2f" % d)
    distances_malware.append(d)

# 箱ひげ図をプロット
plot.figure(figsize=(8, 6))
box = plot.boxplot([distances_cleanware, distances_malware], labels=['Cleanware', 'Malware'],
                  patch_artist=True, widths=0.6)# 各ボックスの色を設定

colors = ['#1f77b4', '#ff7f0e']  # 色の指定
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# 中央値の線を黒色に設定
for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)  # 中央値の線の太さを調整

plot.title('Mahalanobis Distance Comparison')
plot.ylabel('Mahalanobis Distance')
plot.grid(True)
# グラフを保存
plot.savefig("Boxplot.png")
plot.clf()


# 散布図の描画
plot.figure()

# x軸を0から99、y軸を各テンソルの値とする
for i in range(cleanware.shape[0]):
    y = cleanware[i]
    X = np.arange(cleanware.shape[1])
    plot.scatter(X,y,color='blue')

for i in range(malware.shape[0]):
    y = malware[i]
    X = np.arange(malware.shape[1])
    plot.scatter(X,y,color='red')


#plot.plot(malware_average,color="red")
#plot.plot(malware[1],color="green")

plot.ylim(-1, 23)  # y軸の範囲
plot.xlim(-10, 110)   # x軸の範囲
plot.xlabel("Index")
plot.ylabel("Value")
plot.title("Scatter Plot of Cleanware and Malware")
plot.grid(True)
plot.savefig("call-error.png")

