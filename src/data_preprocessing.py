import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Keeping only relevant columns
    df = df[["team1", "team2", "toss_winner", "toss_decision", "winner", "venue"]]

    # Droping missing values
    df = df.dropna()

    # Removing matches with no result
    df = df[df["winner"] != ""]

    return df

#Saving Cleaned Data
def save_data(df, path):
    df.to_csv(path, index=False)
