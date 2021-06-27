import pandas as pd
import numpy as np

path_list = ['000001.XSHE_2014_to_2019.csv', '000001.XSHE_2020.csv']
train_data = pd.read_csv(path_list[0])
test_data = pd.read_csv(path_list[1])
train_len = train_data.shape[0]
data = pd.concat([train_data, test_data], axis = 0)

date = pd.to_datetime(data['date']).dt.date
date_diff = date - date.shift(1)

data['time_interval'] = date_diff.map(lambda x: x/np.timedelta64(1,'D'))

data['SMA_5'] = data['close'].transform(lambda x: x.rolling(window = 5).mean())
data['SMA_15'] = data['close'].transform(lambda x: x.rolling(window = 15).mean())

SD = data['close'].transform(lambda x: x.rolling(window = 15).std())
data['upperband'] = data['SMA_15'] + 2 * SD
data['lowerband'] = data['SMA_15'] - 2 * SD

data['SMA5_volume'] = data['volume'].transform(lambda x: x.rolling(window = 5).mean())


high_low = data['high'] - data['low']
high_close = np.abs(data['high'] - data['close'].shift())
low_close = np.abs(data['low'] - data['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
data['ATR_5'] = true_range.rolling(5).sum()/5
data['ATR_15'] = true_range.rolling(15).sum()/15

_5Ewm = data['close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
_15Ewm = data['close'].transform(lambda x: x.ewm(span=15, adjust=False).mean())
data['MACD'] = _15Ewm - _5Ewm

data = data.fillna(0)

train_data = data.iloc[:train_len, :]
test_data = data.iloc[train_len:, :]
train_path = path_list[0][:-4] + '_ti.csv'
test_path = path_list[1][:-4] + '_ti.csv'
train_data.to_csv(train_path, index = False)
test_data.to_csv(test_path, index = False)