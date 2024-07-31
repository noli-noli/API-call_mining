import os
import pandas as pd
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.initializers import RandomUniform
from tensorflow import keras


from scipy.special import rel_entr
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import jensenshannon
import tensorflow 


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

cleanware_sequences = DATASET_PANDAS[DATASET_PANDAS["y"] == 0]["x_name"]
malware_sequences = DATASET_PANDAS[DATASET_PANDAS["y"] == 1]["x_name"]
#リストサイズを縮小するには、以下のコメントアウトを外す。縮小サイズは x[0:10] にて指定可能
cleanware_sequences = cleanware_sequences[1800:2000].apply(lambda x: x[0:100])
malware_sequences = malware_sequences.apply(lambda x: x[0:100])

#print(cleanware_sequences[10])
# print(malware_sequences[2000])



embedding_model = Sequential()
embedding_model.add(Embedding(302, 200, input_length=1000,embeddings_initializer=RandomUniform(seed=0)))


print(cleanware_sequences.size)

sum = []

for b in range(malware_sequences.size-1900):
    embedding_data = dataset_preprocess(malware_sequences)
    embedding_data = embedding_model(np.array(embedding_data[b],dtype=int))
    embedding_data_one_hot = dataset_preprocess(malware_sequences)
    embedding_data_one_hot = embedding_data_one_hot[b]



    #sample_data = [59,2,3,25,10,11,10,15,25,15,25,20,25,40,20,32,20,25,24,15,3,27,19,24,27,19,62,62,81,20,9,15,25,15,25,15,3,2,3,3,3,3,15,25,15,25,15,25,10,10,10,2,3,3,3,78,78,21,22,18,9,9,15,25,15,25,1,24,27,19,24,27,19,24,27,19,15,2,25,15,3,2,25,15,3,2,3,72,15,3,15,3,3,29,18,24,27,19,24,27]
    #sample_data = [2,1,53,62,1,12,131,39,126,126,126,25,2,15,25,2,12,13,9,2,3,3,3,3,12,13,12,13,9,12,127,12,13,13,13,128,128,128,128,128,13,9,127,9,12,12,9,12,12,12,128,9,9,12,9,12,12,12,128,9,9,12,9,12,12,12,13,9,9,12,9,12,12,13,13,9,9,9,9,131,13,9,15,15,15,15,15,3,3,51,89,2,15,3,3,126,126,126,126,126]
    #sample_data = [59,2,3,25,10,11,10,15]
    #sample_data = [0,1,5,0,3,1,3,6]
    #sample_data = np.array(sample_data, dtype=int)
    #embedding_sample_data = embedding_model(sample_data)


    model = keras.models.load_model("100sequence.h5")
    #data = model.predict(cleanware_sequences[0])




    test = to_categorical(embedding_data_one_hot,303)
    # print(test[0])
    # print(test[1])
    # print(test[2])

    tmp = []
    for i in range(embedding_data_one_hot.size):
        data = model.predict(embedding_data[0:i+1])
        print("data:",data[0:i+1].shape)
        print("test:",test[0:i+1].shape)
        print("data-result:",data[i].sum())
        print("test-result:",test[i].sum())

        #各トークンの予測誤差
        print(np.sum(rel_entr(test[i],data[i])))
        tmp.append(np.sum(rel_entr(test[i],data[i])))

        #シーケンス長の累積誤差
        #print(np.sum(rel_entr(test[0:i+1],data[0:i+1])))
        #tmp.append(np.sum(rel_entr(test[0:i+1],data[0:i+1])))

        #if i==1:
        #    break

        #以下は試行錯誤の残骸
        #print(jensenshannon(data[0:i+1].flatten(),test[0:i+1].flatten())**2)
        #print(data[0:i+1])
        #print(test[0:i+1])
        #sum.append(jensenshannon(data[0:i+1].flatten(),test[0:i+1].flatten())**2)
        #print(most_probable_token_id)
        #print(f"{i}配列目")
        #print("リザルト:",sample_data[0:i+2])
        # print("正解値",sample_data[i+2])
        # print("最も高い分布:",np.max(data))
        # print("実際の分布  :",np.max(data[0,sample_data[0:i+2]]))
        #print(np.sum(rel_entr(data,data[0,sample_data[0:i+1]])))
    sum.append(tmp)

#print(sum)
np.save("malware.npy",np.array(sum,dtype=int))
