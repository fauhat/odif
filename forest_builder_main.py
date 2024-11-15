from forest_builder import ForestBuilder
import pandas as pd
import numpy as np

def data_preprocessing_local(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    x = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)
    """
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x)
    x = minmax_scaler.transform(x)
    """
    return x, y

print("--------Forest Builder Main Starts-----------------")
df = pd.read_csv("data\\tabular\\cricket.csv")
x, y = data_preprocessing_local(df)
ForestBuilder(n_estimators=3,max_samples=16,random_state=49,x=x)
print("--------Forest Builder Main Ends-------------------")