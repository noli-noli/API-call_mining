import os
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.initializers import RandomUniform
from tensorflow import keras

import tensorflow 
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
malware_sequences = DATASET_PANDAS[DATASET_PANDAS["y"] == 1]["x_name"]
#リストサイズを縮小するには、以下のコメントアウトを外す。縮小サイズは x[0:10] にて指定可能
cleanware_sequences = cleanware_sequences[1500:2000].apply(lambda x: x[0:100])
malware_sequences = malware_sequences[0:500].apply(lambda x: x[0:100])

embedding_model = Sequential()
embedding_model.add(Embedding(302, 200, input_length=100,embeddings_initializer=RandomUniform(seed=0)))
model = keras.models.load_model("100sequence.h5")


#sample = np.array([[1,2,3,4,5,0,0,0,0,0]],dtype=int)


new_data = dataset_preprocess(cleanware_sequences)
order,X,y = create_dataset(new_data,False)

predicted = model.predict(X)

sum = []
tmp = []

print("target:",new_data)
print("axis0=",np.size(X,axis=0))
print("axis1=",np.size(X,axis=1))



for a in range(np.size(X,axis=0)):
    #predicted = model.predict(X[0:a+1])
    sample = X[a][X[a] != 0]
    #print("\n(入力値)",sample,f"--> [予測値:{np.argmax(predicted, axis=1)[a]}] [正解値:{y[a]}]")
    #print(order[a],"の予測確率:",predicted[a][y[a]])
    sum.append([order[a],predicted[a][y[a]]])
    #print("予測値(生):",np.argmax(predicted, axis=1)[a])
    #print("予測値(確率):",f"{predicted[0][17]:.8f}")
    #print("マックス:",npargmax(predicted,axis=0))
    #print("---------------")
    #if a==5:
    #    break

print(sum)
np.save(os.path.join("train_result","malware-per_token.npy"),np.array(sum))
