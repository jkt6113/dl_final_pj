import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import model
from torch.utils.data import DataLoader
from dataset import StockDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from train import transform_scaler


path_list = ['000001.XSHE_2014_to_2019_ti.csv', '000001.XSHE_2020_ti.csv']
test_set = StockDataset(file_path = path_list[1], transform = transform_scaler(path_list))
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.resnet32()
model.to(device)
model.load_state_dict(torch.load('best_cnn.pt'))
model.eval()
loss_fn = nn.MSELoss()
tar = []
predict = []
total_loss = 0.
total_len = len(test_dataloader.dataset)
tar_max = transform_scaler(path_list).data_max_[1]
tar_min = transform_scaler(path_list).data_min_[1]

def real_tar(tar):
    return tar * (tar_max - tar_min) + tar_min

with torch.no_grad():
    for idx, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.float().to(device)
        tar.append(real_tar(target.cpu().detach().numpy()))
        output = model(data)
        output = output.squeeze(-1)
        predict.append(real_tar(output.cpu().detach().numpy()))
        total_loss += loss_fn(output, target).item()
    total_loss /= total_len

tar = [t for x in tar for t in x]
predict = [p for x in predict for p in x]

x = range(len(tar))
plt.plot(x,tar)
plt.plot(x,predict)
plt.legend(['target', 'prediction'])
plt.xlabel('Date')
plt.ylabel('Close price')
plt.savefig('target and prediction_56.jpg')

print(total_loss)