import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def vectorized(i):
    e = np.zeros((2,1))
    e[i] = 1
    return e

def data_loader():
    # 读取数据并预处理
    data = pd.read_csv('../datasets/mushrooms.csv')
    encoder = preprocessing.LabelEncoder()
    for col in data.columns:
        data[col] = encoder.fit_transform(data[col])
    data = np.array(data)
    train, test = train_test_split(data, test_size = 0.4)
    # 处理数据为输入输出/特征和标签
    train_out = [x[0] for x in train]
    train_in = np.array([x[1:] for x in train]).astype('float')
    test_out = [x[0] for x in test]
    test_in = np.array([x[1:] for x in test]).astype('float')
    # 向量化
    train_outs = [vectorized(y) for y in train_out]
    train_ins = [np.reshape(x, (22,1)) for x in train_in]
    test_outs = [vectorized(y) for y in test_out]
    test_ins = [np.reshape(x, (22,1)) for x in test_in]
    train_datas = list(zip(train_ins, train_outs))
    test_datas = list(zip(test_ins, test_outs))

    return train_datas, test_datas

if __name__ == "__main__":
    train_datas, test_datas = data_loader()
    print(train_datas[0])