import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tensorflow import keras
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
_,X,y = create_dataset(new_data[0:1499],False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False)





model_path = os.path.join("train_data","100sequence.h5")
model = keras.models.load_model(model_path)



# 名前を使用して中間層のアウトプットを取得するモデルを作成
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('lstm').output)

# 中間層のアウトプットを計算
intermediate_output = intermediate_layer_model.predict(X_test)

# t-SNEを実行
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(intermediate_output)

# 可視化
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.title('t-SNE visualization of LSTM hidden states')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()