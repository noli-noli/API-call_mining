import numpy as np
import os
import matplotlib.pyplot as plot

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


def sequence_padding_removal(x_name,y):
    cleanware_sequence_len = []
    cleanware_sequence = []
    for x,y in zip(x_name,y):
        if(y==1):
            #print([i for i in x if i!= 0])
            cleanware_sequence.append([i for i in x if i != 0])
            cleanware_sequence_len.append(len([i for i in x if i != 0]))
    return cleanware_sequence_len,cleanware_sequence


def sequence_length(sample):
    new_data = []

    for i in sample:
        if len(i)==1000:
            new_data.append(i)
    return new_data

cleanware_sequence_len,cleanware_sequence = sequence_padding_removal(x_name,y)
cleanware_sequence_1000 = sequence_length(cleanware_sequence)




"""
plot.hist(cleanware_sequence_len,bins=30)
# タイトルとラベルを追加
plot.title('cleanware sequence results')
plot.xlabel('sequence sample')
plot.ylabel('sequence length')

plot.savefig("cleanware_sequence.png")
"""