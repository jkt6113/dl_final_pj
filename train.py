from torch.utils.data import DataLoader
from dataset import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


path_list = ['000001.XSHE_2014_to_2019.csv', '000001.XSHE_2020.csv']

def transform_scaler(path_list):
    scaler = MinMaxScaler()
    train_data = pd.read_csv(path_list[0])
    test_data = pd.read_csv(path_list[1])
    data = pd.concat([train_data, test_data], axis = 0)
    data = np.array(data)
    scaler.fit(data[:, 2:])
    return scaler

train_set = StockDataset(file_path = path_list[0], transform = transform_scaler(path_list))
test_set = StockDataset(file_path = path_list[1], transform = transform_scaler(path_list))

for window, target in train_set:
        print(window)
        print(target)
        break

for window, target in test_set:
        print(window)
        print(target)
        break


    
