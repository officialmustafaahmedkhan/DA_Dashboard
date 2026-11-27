# ============================================================
# PHASE 4: ENCODING, TRAIN-TEST SPLIT & FEATURE SCALING
# Dataset: sindh-education-stats_2.csv
# Covers: Steps 15 to 18
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 4.1 LOAD PHASE 3 OUTPUT
# ------------------------------------------------------------

df = pd.read_csv("P3_engineered_subset.csv")

print("=" * 65)
print("PHASE 4: ENCODING, TRAIN-TEST SPLIT & FEATURE SCALING")
print("=" * 65)

# ------------------------------------------------------------
# 4.2 ENCODE CATEGORICAL FEATURES (Step 15)
# ------------------------------------------------------------

print("\n--- 4.2 LABEL ENCODING CATEGORICAL FEATURES ---")

cat_features = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
encoders = {}

for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"Encoded '{col}' -> Classes: {list(le.classes_)}")

# ------------------------------------------------------------
# 4.3 DEFINE INPUT FEATURES (X) AND TARGET (y) (Step 16)
# ------------------------------------------------------------

print("\n--- 4.3 DEFINE INPUT FEATURES & TARGET ---")

# Predict education 'Category' (ENROLMENT, INSTITUTIONS, TEACHING STAFF)
target_col = "Category"
features = [
    "Location",
    "School Type",
    "Composition",
    "Number",
    "Population",
    "No/100000",
    "Log_Number"
]
features = [f for f in features if f in df.columns]

X = df[features]
y = df[target_col]

print("Selected Features:", features)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ------------------------------------------------------------
# 4.4 STRATIFIED TRAIN-TEST SPLIT (Step 17)
# ------------------------------------------------------------

print("\n--- 4.4 TRAIN-TEST SPLIT (80/20 STRATIFIED) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("\nTrain Target Distribution:\n", y_train.value_counts(normalize=True).round(4))
print("\nTest Target Distribution:\n", y_test.value_counts(normalize=True).round(4))

# ------------------------------------------------------------
# 4.5 SCALE NUMERICAL FEATURES (Step 18)
# ------------------------------------------------------------

print("\n--- 4.5 STANDARDSCALER TRANSFORMATION ---")

scaler = StandardScaler()

# Fit ONLY on training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled Train Mean (approx 0):", X_train_scaled.mean(axis=0).round(2))
print("Scaled Train Std (approx 1):", X_train_scaled.std(axis=0).round(2))

# Save preprocessed arrays and encoded dataset for Phase 5 & Phase 6
np.save("P4_X_train_scaled.npy", X_train_scaled)
np.save("P4_X_test_scaled.npy", X_test_scaled)
np.save("P4_y_train.npy", y_train.to_numpy())
np.save("P4_y_test.npy", y_test.to_numpy())
df.to_csv("P4_encoded_clean_dataset.csv", index=False)

print("\nPHASE 4 COMPLETED SUCCESSFULLY!")