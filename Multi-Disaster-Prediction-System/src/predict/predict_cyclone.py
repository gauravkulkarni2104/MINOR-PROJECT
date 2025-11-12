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
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------
# 1) File paths & constants
# ---------------------------
DATA_PATH = "Historical_Tropical_Storm_Tracks.csv"
MODEL_OUT = "src/models/cyclone_model.pkl"
RANDOM_STATE = 42

# ---------------------------
# 2) Load dataset
# ---------------------------
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded. Shape:", df.shape)

# show first rows and unique basin values
print("\n--- sample rows ---")
print(df.head()[['BASIN']].head(10))
print("\n--- Unique BASIN values (top 50) ---")
print(df['BASIN'].value_counts(dropna=False).head(50))

# ---------------------------
# 3) Robust mapping of BASIN -> 4 regions
# ---------------------------
def map_basin_to_region(basin):
    if pd.isna(basin):
        return None
    b = str(basin).strip().lower()

    # Try common full words
    if "north" in b and "indian" in b:
        return "North_Indian"
    if "south" in b and "indian" in b:
        return "South_Indian"
    if "western" in b and "pacific" in b:
        return "Western_Pacific"
    if "eastern" in b and "pacific" in b:
        return "Eastern_Pacific"

    # Check short forms
    if b in ("ni", "northindian", "north_indian", "n.indian"):
        return "North_Indian"
    if b in ("si", "southindian", "south_indian", "s.indian"):
        return "South_Indian"
    if b in ("wp", "w.pacific", "westernpacific", "westpac"):
        return "Western_Pacific"
    if b in ("ep", "e.pacific", "easternpacific", "eastpac"):
        return "Eastern_Pacific"

    # Fallback keywords
    if "indian" in b and "north" in b:
        return "North_Indian"
    if "indian" in b and "south" in b:
        return "South_Indian"
    if "pacific" in b and ("west" in b or "western" in b):
        return "Western_Pacific"
    if "pacific" in b and ("east" in b or "eastern" in b):
        return "Eastern_Pacific"

    return None

df['Region'] = df['BASIN'].apply(map_basin_to_region)

print("\n--- After mapping: region value counts (including NaN) ---")
print(df['Region'].value_counts(dropna=False))

# show unmapped rows
num_failed = df['Region'].isna().sum()
print(f"\nNumber of rows that could not be mapped: {num_failed}")
if num_failed > 0:
    print("Sample of unmapped BASIN values:")
    print(df.loc[df['Region'].isna(), 'BASIN'].value_counts().head(20))

# ---------------------------
# 4) Keep only mapped rows
# ---------------------------
df_mapped = df.dropna(subset=['Region']).copy()
print("\nMapped dataset shape:", df_mapped.shape)
print("Mapped region distribution:")
print(df_mapped['Region'].value_counts())

# Warn if expected regions missing
expected_regions = {"North_Indian", "South_Indian", "Western_Pacific", "Eastern_Pacific"}
present = set(df_mapped['Region'].unique())
missing = expected_regions - present
if missing:
    print("\n⚠️ Warning: Missing expected regions:", missing)
    print("Please inspect BASIN values if needed.")

# ---------------------------
# 5) Prepare features & labels
# ---------------------------
required_cols = ['YEAR','MONTH','DAY','LAT','LONG','WIND_KTS','PRESSURE']
for col in required_cols:
    if col not in df_mapped.columns:
        raise ValueError(f"Required column missing: {col}")

data = df_mapped[required_cols].astype(float)
labels = df_mapped['Region'].astype(str)

# ---------------------------
# 6) Split and train model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)

print("\nTraining shape:", X_train.shape, "Test shape:", X_test.shape)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ---------------------------
# 7) Evaluate model
# ---------------------------
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 8) Save model
# ---------------------------
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
print(f"\n💾 Model saved to: {MODEL_OUT}")

# ---------------------------
# 9) Prediction using user-specified date + inputs
# ---------------------------
print("\n🌀 --- Cyclone Region Prediction ---")

# Ask user for custom date
print("\n📅 Enter the date for prediction:")
try:
    year = int(input("Enter Year (e.g., 2025): ") or datetime.now().year)
    month = int(input("Enter Month (1-12): ") or datetime.now().month)
    day = int(input("Enter Day (1-31): ") or datetime.now().day)
except ValueError:
    print("⚠️ Invalid date input! Using current date instead.")
    today = datetime.now()
    year, month, day = today.year, today.month, today.day

print(f"✅ Using date: {day}-{month}-{year}")

# Get meteorological parameters
try:
    lat = float(input("Enter Latitude (e.g., 15.0): ") or 15.0)
    lon = float(input("Enter Longitude (e.g., 85.0): ") or 85.0)
    wind = float(input("Enter Wind Speed in KTS (e.g., 70): ") or 70.0)
    pressure = float(input("Enter Pressure in hPa (e.g., 970): ") or 970.0)
except ValueError:
    print("⚠️ Invalid meteorological input! Using defaults.")
    lat, lon, wind, pressure = 15.0, 85.0, 70.0, 970.0

# Create sample DataFrame
sample = {
    "YEAR": year,
    "MONTH": month,
    "DAY": day,
    "LAT": lat,
    "LONG": lon,
    "WIND_KTS": wind,
    "PRESSURE": pressure
}

sample_df = pd.DataFrame([sample])
pred = model.predict(sample_df)[0]
proba = model.predict_proba(sample_df)[0]

# ---------------------------
# 10) Display prediction results
# ---------------------------
print(f"\n🌪 Predicted Cyclone Region: {pred}")
print("\n🔍 Probability Scores for Each Region:")
for region, p in zip(model.classes_, proba):
    print(f"  {region}: {round(p * 100, 2)}%")

print("\n✅ Prediction complete.")
