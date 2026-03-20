import pandas as pd

def prepare_input(input_dict, columns):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    return df