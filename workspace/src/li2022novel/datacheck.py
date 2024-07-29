import numpy as np
import os

DATASET_PATH = os.path.join("/workspace","datasets","li2022novel","data_demo.npz")
READ_DATASET = np.load(DATASET_PATH)

#word2id = np.load(os.path.join("/workspace","datasets","word2id.npz"),allow_pickle=True)
#word2id = word2id['word2id'][()]
#print(word2id)


counter = 0

for feature in READ_DATASET["y"]:
    if feature == 1:
        counter = counter + 1


feature_name = READ_DATASET.files
print(feature_name)
print(counter)