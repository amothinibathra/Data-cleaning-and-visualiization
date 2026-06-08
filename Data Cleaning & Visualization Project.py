import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. LOAD DATASET

df = pd.read_csv("dataset.csv")

# View basic info
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# 3. DATA CLEANING

# 3.1 Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3.2 Fill missing values
# Numeric columns → mean
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Categorical columns → mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 3.3 Remove duplicates
df.drop_duplicates(inplace=True)

# 4. OUTLIER HANDLING (IQR METHOD)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df[col] >= Q1 - 1.5 * IQR) &
        (df[col] <= Q3 + 1.5 * IQR)
    ]

# =========================
# 5. DATA TYPE FIX
# =========================
for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')

# =========================
# 6. EXPLORATORY DATA ANALYSIS (EDA)
# =========================

print("\nCleaned Dataset Info:")
print(df.info())

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# =========================
# 7. VISUALIZATION
# =========================

sns.set(style="whitegrid")

# -------------------------
# 7.1 Histogram
# -------------------------
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# -------------------------
# 7.2 Boxplot (Category vs Numeric)
# -------------------------
cat_cols = df.select_dtypes(include='object').columns

if len(cat_cols) > 0 and len(numeric_cols) > 0:
    plt.figure()
    sns.boxplot(x=cat_cols[0], y=numeric_cols[0], data=df)
    plt.xticks(rotation=45)
    plt.title(f"{cat_cols[0]} vs {numeric_cols[0]}")
    plt.show()

# -------------------------
# 7.3 Correlation Heatmap
# -------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# 7.4 Line Chart (if date column exists)
# -------------------------
date_cols = [col for col in df.columns if "date" in col.lower()]

if len(date_cols) > 0 and len(numeric_cols) > 0:
    df.groupby(date_cols[0])[numeric_cols[0]].sum().plot(figsize=(10,5))
    plt.title(f"{numeric_cols[0]} Trend Over Time")
    plt.ylabel(numeric_cols[0])
    plt.show()

# =========================
# 8. FINAL CLEAN DATA EXPORT
# =========================
df.to_csv("cleaned_dataset.csv", index=False)

print("\nProject Completed Successfully!")
print("Cleaned dataset saved as cleaned_dataset.csv")