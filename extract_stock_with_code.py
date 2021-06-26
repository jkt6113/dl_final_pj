import pandas as pd
import tqdm

def extract_stock(code):
    stock = pd.DataFrame(columns = ['code', 'date', 'open', 'close', 'high', 'low', 'volume', 'money'])
    for year in tqdm.tqdm(range(2014, 2021)):
        path = 'processed_data/jqdata_a_stocks_' + str(year) + '.csv'
        every_year = pd.read_csv(path)
        every_year = every_year[every_year['code'] == code]
        if year == 2020:
            save_path = code + '_2020.csv'
            every_year.to_csv(save_path, index = False)
            break
        stock = pd.concat([stock, every_year], axis = 0)
    save_path = code + '_2014_to_2019.csv'
    stock.to_csv(save_path, index = False)


if __name__ == "__main__":
    extract_stock('000001.XSHE')