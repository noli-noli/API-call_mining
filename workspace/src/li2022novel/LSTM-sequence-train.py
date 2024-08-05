import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow 
from sklearn.metrics import precision_score, recall_score, f1_score,precision_recall_curve,roc_curve, auc
from tensorflow.keras.initializers import RandomUniform
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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
cleanware_sequences = cleanware_sequences.apply(lambda x: x[0:100])


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



# 予測結果の評価
precision = precision_score(y_test, pred, average='weighted')
recall = recall_score(y_test, pred, average='weighted')
f1 = f1_score(y_test, pred, average='weighted')

print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)


mse = mean_squared_error(y_test, pred)
print("MSE: ", mse)



lstm_layer = model.layers[-2]  # LSTMレイヤー（必要に応じて変更）
_, h, c = lstm_layer.output  # 隠れ状態とセル状態

# 新しいモデルを作成（入力からLSTMレイヤーの出力まで）
internal_model = Model(inputs=model.input, outputs=[h, c])

# データセットの内部状態を取得
hidden_states = internal_model.predict(X)  # 隠れ状態とセル状態の取得

# 隠れ状態をt-SNEで2次元に圧縮して可視化
tsne = TSNE(n_components=2)
hidden_states_2d = tsne.fit_transform(hidden_states[0])  # 隠れ状態のみ使用

# 可視化
plt.scatter(hidden_states_2d[:, 0], hidden_states_2d[:, 1])
plt.title("t-SNE Visualization of LSTM Hidden States")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("sample.png")