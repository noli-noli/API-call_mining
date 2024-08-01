import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow 
from sklearn.metrics import precision_score, recall_score, f1_score,precision_recall_curve,roc_curve, auc
from tensorflow.keras.initializers import RandomUniform
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl


def create_dataset(sample: np.ndarray , DEBUG: bool) -> np.ndarray:
    """
    引数として受け取る「sample」は必ずnumpy.ndarray形式のリストにする事。
    本関数は、受け取ったリストをローリングウィンドウで分割しつつパディングを行い、Xとyを返す。

    例としては、以下のようなデータがあるとする。
    sample = [[1,2,3,4,5],[6,7,8,9,10]]

    このデータをローリングウィンドウし、パディングを行うと以下の出力が得られる。
    X:[0,0,0,0,1] y:2
    X:[0,0,0,1,2] y:3
    X:[0,0,1,2,3] y:4
    X:[0,1,2,3,4] y:5

    X:[0,0,0,0,6] y:7
    X:[0,0,0,6,7] y:8
    X:[0,0,6,7,8] y:9
    X:[0,6,7,8,9] y:10
    """

    new_data_X = []
    new_data_y = []

    for i in sample:
        data_length = len(i)                #リストの長さを取得
        num_zero = (i==0).sum()             #リスト内の0の数を取得
        roop = data_length - (num_zero+1)   #ループ回数を計算

        if(DEBUG == True):
            print("-----------------")
            print("data_length:"+str(data_length))
            print("num_zero:"+str(num_zero))
            print("roop:"+str(roop))
            print(i)

        for j in range(roop):
            zero_padding = data_length - (j+1)                                      #ゼロパディングの数を計算
            new_data_X.append([0]*zero_padding + list(i[num_zero:num_zero+j+1]))    #Xのデータを作成
            new_data_y.append(i[num_zero+j+1])                                      #yのデータを作成
            if(DEBUG == True):
                print("[RESULT]zero_padding:"+str(zero_padding))
                print("[RESULT]train:",([0]*zero_padding + list(i[num_zero:num_zero+j+1])))
                print("[RESULT]test:",i[num_zero+j+1])

    #new_data_X = pd.Series(new_data_X , name="new_data_X",dtype=object)
    #new_data_y = pd.Series(new_data_y , name="new_data_y",dtype=object)
    new_data_X = np.array(new_data_X)
    new_data_y = np.array(new_data_y)
    
    return new_data_X,new_data_y

def dataset_preprocess(sample: np.ndarray) -> np.ndarray:
    """
    ゼロパディングとデータの配置を交換する関数。

    例として、以下のようなデータがあるとする。
    sample = [1,2,3,4,5,0,0,0,0]

    出力として、以下のようなデータが得られる。
    return = [0,0,0,0,1,2,3,4,5]
    """
    new_data = []

    for i in sample:
        zero_count = (i==0).sum()
        new_sample = [0]*zero_count + [a for a in i if a !=0]
        new_data.append(new_sample)

    new_data = np.array(new_data, dtype=object)

    return new_data


DATASET_PATH = os.path.join("/workspace","datasets","li2022novel","data_demo.npz")
DATASET_LOAD = np.load(DATASET_PATH)

x_name = DATASET_LOAD["x_name"]
x_semantic = DATASET_LOAD["x_semantic"]
y = DATASET_LOAD["y"]

DATASET_PANDAS = pd.DataFrame({
    "x_name":list(x_name),
    "x_semantic":list(x_semantic),
    "y":list(y)
})

"""
def sequence_length(sample):
    new_data = []

    for i in sample:
        if len(i)==1000:
            new_data.append(i)
    return new_data

def sequence_padding_removal(cleanware_sequences):
    cleanware_sequence = []
    for a in cleanware_sequences:
            cleanware_sequence.append([i for i in a if i != 0])
    return cleanware_sequence
"""

cleanware_sequences = DATASET_PANDAS[DATASET_PANDAS["y"] == 0]["x_name"]
#リストサイズを縮小するには、以下のコメントアウトを外す。縮小サイズは x[0:10] にて指定可能
cleanware_sequences = cleanware_sequences.apply(lambda x: x[0:100])

#特定のシーケンス長のデータのみを取り出す。
#cleanware_sequences = sequence_padding_removal(cleanware_sequences)
#cleanware_sequences = sequence_length(cleanware_sequences)
#cleanware_sequences = np.array(cleanware_sequences, dtype=object)

new_data = dataset_preprocess(cleanware_sequences)

print(new_data.max())

X,y = create_dataset(new_data[0:999],False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# ランダムシードを固定
#tensorflow.random.set_seed(42)

model = Sequential()
model.add(Embedding(302, 200, input_length=1000,embeddings_initializer=RandomUniform(seed=0)))
model.add(LSTM(200))
model.add(Dense(303, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=1,restore_best_weights=True)
model.fit(X_train, y_train, batch_size=256,epochs=100, verbose=1,validation_data=(X_test, y_test),callbacks=[early_stopping])
model.save("100sequence.h5")


# 予測
predicted = model.predict(X_test)
pred = np.argmax(predicted, axis=1)  # 予測されたクラスラベルを取得



# グラフのプロット
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [1, 1]})


# 図全体の枠線の設定
fig.patch.set_edgecolor('black')  # 枠線の色を黒に設定
fig.patch.set_linewidth(3)  # 枠線の幅を1に設定
mpl.rcParams['font.size'] = 25

# 上のグラフ（ECG）
ax1.plot(y_test, label='Actual',color='blue' , linewidth=0.5)
ax1.plot(pred, label='Predicted',color='red', linewidth=0.5)
ax1.set_xlabel('API call sequence')
ax1.set_ylabel('Correct answer and predicted value')
ax1.set_facecolor(color='#F0F0F0')
ax1.grid()
ax1.legend(loc='upper right')


# 2乗誤差の計算
mse_values = []
for i in range(len(X_test)):
    true_vector = np.zeros(303)
    true_vector[y_test[i]] = 1
    pred_vector = predicted[i]
    mse = mean_squared_error(true_vector, pred_vector)
    mse_values.append(mse)

# 下のグラフ（Mahalanobis Distance）
ax2.plot(range(1, len(mse_values) + 1), mse_values, color='red', linewidth=0.5)
#ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=1, label='Threshold')
ax2.set_xlabel('API call sequence')
ax2.set_ylabel('Mean Squared Error for API Call Predictions')
ax2.set_facecolor(color='#F0F0F0')
ax2.grid()
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"二乗誤差mal_reports.png", dpi=200)
plt.clf()



# 予測結果の評価
precision = precision_score(y_test, pred, average='weighted')
recall = recall_score(y_test, pred, average='weighted')
f1 = f1_score(y_test, pred, average='weighted')

print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)

"""
# ROC曲線を計算
fpr, tpr, thresholds = roc_curve(y_test, pred)
# AUCを計算
roc_auc = auc(fpr, tpr)
# ROC曲線をプロット
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(f"ROC.png",dpi=200)
plt.clf()
"""

mse = mean_squared_error(y_test, pred)
print("MSE: ", mse)