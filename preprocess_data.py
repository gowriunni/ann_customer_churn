
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Remove customer ID
df = df.drop(columns=["customerID"])

# 3. Convert TotalCharges if present
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

# 4. Select exactly 5 original features
selected_features = [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "InternetService",
    "TechSupport"
]

X = df[selected_features].copy()
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in Churn column.")

print("Selected original features:", selected_features)
print("Number of original features:", X.shape[1])

# 5. Identify numerical and categorical features
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)

# 6. Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 7. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine preprocessing
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 9. Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 10. Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrices to NumPy arrays
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)
X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

y_train = y_train.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.float32)

# 11. Save processed data
np.savez(
    "processed_data.npz",
    X_train=X_train_processed,
    X_test=X_test_processed,
    y_train=y_train,
    y_test=y_test
)

# Save the fitted preprocessor
joblib.dump(preprocessor, "preprocessor.joblib")

# 12. Display results
print("\nPreprocessing completed successfully!")
print("Original input features:", X.shape[1])
print("Processed input features:", X_train_processed.shape[1])
print("Training input shape:", X_train_processed.shape)
print("Testing input shape:", X_test_processed.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
print("Training churn rate:", round(float(y_train.mean()), 3))
print("Testing churn rate:", round(float(y_test.mean()), 3))

