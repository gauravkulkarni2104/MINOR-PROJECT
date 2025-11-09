"""
train_and_predict_cyclone_fixed.py

Put this in: Multi-Disaster-Prediction-System/src/models/
Run from project root:
python src/models/train_and_predict_cyclone_fixed.py
"""

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
# 2) Robust mapping of BASIN -> 4 regions
# ---------------------------
def map_basin_to_region(basin):
    if pd.isna(basin):
        return None
    b = str(basin).strip().lower()
    # try common full words
    if "north" in b and "indian" in b:
        return "North_Indian"
    if "south" in b and "indian" in b:
        return "South_Indian"
    if "western" in b and "pacific" in b:
        return "Western_Pacific"
    if "eastern" in b and "pacific" in b:
        return "Eastern_Pacific"
    # check common short forms/abbreviations
    if b in ("ni", "northindian", "north_indian", "n.indian"):
        return "North_Indian"
    if b in ("si", "southindian", "south_indian", "s.indian"):
        return "South_Indian"
    if b in ("wp", "w.pacific", "westernpacific", "westpac"):
        return "Western_Pacific"
    if b in ("ep", "e.pacific", "easternpacific", "eastpac"):
        return "Eastern_Pacific"
    # fallback: check presence of keywords loosely
    if "indian" in b and "north" in b:
        return "North_Indian"
    if "indian" in b and "south" in b:
        return "South_Indian"
    if "pacific" in b and ("west" in b or "western" in b):
        return "Western_Pacific"
    if "pacific" in b and ("east" in b or "eastern" in b):
        return "Eastern_Pacific"
    # If it still doesn't match, return None (we'll drop)
    return None

df['Region'] = df['BASIN'].apply(map_basin_to_region)

print("\n--- After mapping: region value counts (including NaN) ---")
print(df['Region'].value_counts(dropna=False))

# show example rows where mapping failed (Region is null)
num_failed = df['Region'].isna().sum()
print(f"\nNumber of rows that could not be mapped to our 4 regions: {num_failed}")
if num_failed > 0:
    print("Sample of BASIN values that couldn't be mapped:")
    print(df.loc[df['Region'].isna(), 'BASIN'].value_counts().head(20))

# ---------------------------
# 3) Keep only mapped rows — but make sure we have all 4
# ---------------------------
df_mapped = df.dropna(subset=['Region']).copy()
print("\nMapped dataset shape:", df_mapped.shape)
print("Mapped region distribution:")
print(df_mapped['Region'].value_counts())

# If any of the 4 regions are missing, print a warning and stop
expected_regions = {"North_Indian", "South_Indian", "Western_Pacific", "Eastern_Pacific"}
present = set(df_mapped['Region'].unique())
missing = expected_regions - present
if missing:
    print("\n⚠️ Warning: The following expected regions are missing after mapping:", missing)
    print("Please inspect 'BASIN' values above and update mapping rules if needed.")
    # Do NOT exit forcibly; we continue, but model may have fewer classes

# ---------------------------
# 4) Prepare features & label
# ---------------------------
# Ensure numeric columns exist
for col in ['YEAR','MONTH','DAY','LAT','LONG','WIND_KTS','PRESSURE']:
    if col not in df_mapped.columns:
        raise ValueError(f"Required column missing: {col}")

data = df_mapped[['YEAR','MONTH','DAY','LAT','LONG','WIND_KTS','PRESSURE']].astype(float)
labels = df_mapped['Region'].astype(str)

# ---------------------------
# 5) Train/test split (stratify to keep class distribution) and train classifier
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)

print("\nTraining shape:", X_train.shape, "Test shape:", X_test.shape)
# Use class_weight='balanced' to help with imbalance
model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')
model.fit(X_train, y_train)

# ---------------------------
# 6) Evaluate
# ---------------------------
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 7) Save model
# ---------------------------
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to: {MODEL_OUT}")

# ---------------------------
# 8) Example prediction (use DataFrame to avoid sklearn warning)
# ---------------------------
sample = {
    "YEAR": 2025,
    "MONTH": 11,
    "DAY": 7,
    "LAT": 15.0,
    "LONG": 85.0,
    "WIND_KTS": 70.0,
    "PRESSURE": 970.0
}
sample_df = pd.DataFrame([sample])
pred = model.predict(sample_df)[0]
proba = model.predict_proba(sample_df)[0]

print(f"\n🌪 Predicted Cyclone Region: {pred}")
print("🔍 Probability Scores:")
for region, p in zip(model.classes_, proba):
    print(f"  {region}: {round(p*100, 2)}%")
