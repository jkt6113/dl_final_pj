import pandas as pd
import tqdm

def main(filepath):
    df = pd.read_pickle(filepath)
    df['date'] = pd.to_datetime(df['time']).dt.date
    df['time'] = pd.to_datetime(df['time']).dt.time
    group = df.groupby(['code', 'date'])
    new_group = pd.DataFrame()
    new_group['open'] = group.first()['open']
    new_group['close'] = group.tail(1).set_index(['code', 'date'])['close']
    new_group['high'] = group['high'].max()
    new_group['low'] = group['low'].min()
    new_group['volume'] = group['volume'].sum()
    new_group['money'] = group['money'].sum()
    new_group = new_group[~new_group['volume'].isin([0])]
    new_group = new_group.reset_index()
    resultpath = filepath[:-4] + '.csv'
    new_group.to_csv(resultpath, index = False)


if __name__ == "__main__":
    for year in tqdm.tqdm(range(2014, 2021)):
        path = 'jqdata_a_stocks_5min_' + str(year) + '.pkl'
        main(path)
