from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, file_path, transform = None, window_size = 15):
        self.path = file_path
        self.transform = transform
        data = pd.read_csv(file_path)
        self.data = np.array(data)
        if transform != None:
            self.data[:, 2:] = transform.transform(self.data[:, 2:])
        self.window_size = window_size
    def __getitem__(self, index):
        window = self.data[index:index+self.window_size, 2:]
        target = self.data[index+self.window_size, 3]   # close price
        return window, target
        
    def __len__(self):
        return len(self.data) - self.window_size


if __name__ == "__main__":
    dataset = StockDataset(file_path = '000001.XSHE_2020.csv')
    for window, target in dataset:
        print(window)
        print(target)
        break
