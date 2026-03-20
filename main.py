from src.data_preprocessing import load_data, clean_data, save_data
from src.feature_engineering import encode_data
from src.train_model import train, save_model
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = "data/raw/matches.csv"

#Loading data
df = load_data(DATA_PATH)

#Cleaning data
df = clean_data(df)
save_data(df, "data/processed/clean_data.csv")

#Separate features and target
y = df["winner"]
X = df.drop("winner", axis=1)

#Encode only features
X = encode_data(X)

#Encode target labels (IMPORTANT FIX)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Saving encoder and columns (used in Streamlit)
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(X.columns, "models/columns.pkl")

# Training model
model = train(X, y_encoded)

# Saving model
save_model(model, "models/model.pkl")