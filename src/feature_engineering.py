import pandas as pd

#encoding
def encode_data(df):
    return pd.get_dummies(df, drop_first=True)