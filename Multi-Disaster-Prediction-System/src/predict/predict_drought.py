import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/cyclone/Historical_Tropical_Storm_Tracks.csv"
MODEL_OUT = "src/models/cyclone_model.pkl"
RANDOM_STATE = 42

# ---------------------------
# 1) Load dataset & inspect BASIN values
# ---------------------------
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded. Shape:", df.shape)

# show first rows and unique basin values to understand source labels
print("\n--- sample rows ---")
print(df.head()[['BASIN']].head(10))
print("\n--- Unique BASIN values (top 50) ---")
print(df['BASIN'].value_counts(dropna=False).head(50))

# ---------------------------