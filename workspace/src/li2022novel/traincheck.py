import os
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.initializers import RandomUniform
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


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
READ_DATASET = np.load(DATASET_PATH)

#word2id = np.load(os.path.join("/workspace","datasets","word2id.npz"),allow_pickle=True)
#word2id = word2id['word2id'][()]
#print(word2id)


feature_name = READ_DATASET.files
print("特徴量:",feature_name)
print("type:",type(READ_DATASET))


x_name = READ_DATASET["x_name"]
x_semantic = READ_DATASET["x_semantic"]
y = READ_DATASET["y"]


DATASET_PANDAS = pd.DataFrame({
    "x_name":list(x_name),
    "x_semantic":list(x_semantic),
    "y":list(y)
})


cleanware_sequences = DATASET_PANDAS[DATASET_PANDAS["y"] == 0]["x_name"]
malware_sequences = DATASET_PANDAS[DATASET_PANDAS["y"] == 1]["x_name"]
#リストサイズを縮小するには、以下のコメントアウトを外す。縮小サイズは x[0:10] にて指定可能
cleanware_sequences = cleanware_sequences[1500:2000].apply(lambda x: x[0:100])
malware_sequences = malware_sequences.apply(lambda x: x[0:100])

embedding_model = Sequential()
embedding_model.add(Embedding(302, 200, input_length=100,embeddings_initializer=RandomUniform(seed=0)))
model = keras.models.load_model("100sequence.h5")

new_data = dataset_preprocess(cleanware_sequences)

X,y = create_dataset(new_data[0:999],False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False)

# 予測
predicted = model.predict(np.array([0,1,5,7]))
#print(type(predicted))
#print(predicted)
pred = np.argmax(predicted, axis=1)  # 予測されたクラスラベルを取得
print(X_test.shape)


exit()

print("--------------------")

print(type(pred))
print(pred)
print(y_test)

print(pred.shape)


# 予測結果の評価
precision = precision_score(y_test, pred, average='weighted')
recall = recall_score(y_test, pred, average='weighted')
f1 = f1_score(y_test, pred, average='weighted')

print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)