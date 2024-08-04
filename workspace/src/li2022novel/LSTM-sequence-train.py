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
from modules import create_dataset,dataset_preprocess,sequence_padding_removal



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


cleanware_sequences = DATASET_PANDAS[DATASET_PANDAS["y"] == 0]["x_name"]
#リストサイズを縮小するには、以下のコメントアウトを外す。縮小サイズは x[0:10] にて指定可能
cleanware_sequences = cleanware_sequences.apply(lambda x: x[0:130])


new_data = dataset_preprocess(cleanware_sequences)

print(new_data.max())

_,X,y = create_dataset(new_data[0:1499],False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False)


model = Sequential()
model.add(Embedding(302, 200, input_length=100,embeddings_initializer=RandomUniform(seed=0)))
model.add(LSTM(200))
model.add(Dense(303, activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])



early_stopping = EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)
model.fit(X_train, y_train, batch_size=128,epochs=100, verbose=1,validation_data=(X_test, y_test),callbacks=[early_stopping])
model.save(os.path.join("train_data","LSTM_100sequence_128batch.h5"))


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