import pandas as pd
df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("first fivr record:")
print(df.head())

print("\ndataset shape:")
print(df.shape)

print("\ncolumn names:")
print(df.columns.tolist())

print("\nmissing values:")
print(df.isnull().sum())

print("\nchurn distribution:")
print(df["Churn"].value_counts())