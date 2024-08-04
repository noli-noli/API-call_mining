import numpy as np

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

    new_data_order = []
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
            new_data_order.append(j)                                                #処理対象の文字位置
            new_data_X.append([0]*zero_padding + list(i[num_zero:num_zero+j+1]))    #Xのデータを作成
            new_data_y.append(i[num_zero+j+1])                                      #yのデータを作成
            if(DEBUG == True):
                print("[RESULT]zero_padding:"+str(zero_padding))
                print("[RESULT]train:",([0]*zero_padding + list(i[num_zero:num_zero+j+1])))
                print("[RESULT]test:",i[num_zero+j+1])

    #new_data_X = pd.Series(new_data_X , name="new_data_X",dtype=object)
    #new_data_y = pd.Series(new_data_y , name="new_data_y",dtype=object)
    new_data_order = np.array(new_data_order)
    new_data_X = np.array(new_data_X)
    new_data_y = np.array(new_data_y)
    
    return new_data_order,new_data_X,new_data_y

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

def sequence_padding_removal(cleanware_sequences):
    """
    リストから"0"の要素だけを削除する
    """
    cleanware_sequence = []
    for a in cleanware_sequences:
            cleanware_sequence.append([i for i in a if i != 0])
    return cleanware_sequence