# ============================================================
# WATER QUALITY ANALYSIS AND POTABILITY PREDICTION PROJECT
# ============================================================

# ---------------- IMPORTING LIBRARIES ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


# ============================================================
# LOADING DATASET
# ============================================================

df = pd.read_csv("water_potability.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)

print("\n========== MISSING VALUES BEFORE HANDLING ==========")
print(df.isnull().sum())


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

# ---------------- UNIVARIATE ANALYSIS ----------------

# Count of potable and non-potable water
plt.figure(figsize=(6, 4))
sns.countplot(x="Potability", data=df)
plt.title("Count of Potable and Non-Potable Water")
plt.xlabel("Potability")
plt.ylabel("Count")
plt.show()

# Histogram of pH
plt.figure(figsize=(6, 4))
sns.histplot(df["ph"], bins=30, kde=True)
plt.title("Distribution of pH")
plt.xlabel("pH Value")
plt.ylabel("Frequency")
plt.show()

# Histogram of Hardness
plt.figure(figsize=(6, 4))
sns.histplot(df["Hardness"], bins=30, kde=True)
plt.title("Distribution of Hardness")
plt.xlabel("Hardness")
plt.ylabel("Frequency")
plt.show()

# Histogram of Solids
plt.figure(figsize=(6, 4))
sns.histplot(df["Solids"], bins=30, kde=True)
plt.title("Distribution of Solids")
plt.xlabel("Solids")
plt.ylabel("Frequency")
plt.show()


# ---------------- BIVARIATE ANALYSIS ----------------

# pH vs Potability
plt.figure(figsize=(6, 4))
sns.boxplot(x="Potability", y="ph", data=df)
plt.title("pH vs Potability")
plt.xlabel("Potability")
plt.ylabel("pH")
plt.show()

# Hardness vs Potability
plt.figure(figsize=(6, 4))
sns.boxplot(x="Potability", y="Hardness", data=df)
plt.title("Hardness vs Potability")
plt.xlabel("Potability")
plt.ylabel("Hardness")
plt.show()

# Solids vs Potability
plt.figure(figsize=(6, 4))
sns.boxplot(x="Potability", y="Solids", data=df)
plt.title("Solids vs Potability")
plt.xlabel("Potability")
plt.ylabel("Solids")
plt.show()


# ---------------- MULTIVARIATE ANALYSIS ----------------

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Water Quality Features")
plt.show()


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

print("\n========== HANDLING MISSING VALUES ==========")

# Making a copy of original dataset
df_cleaned = df.copy()

# Filling missing numerical values using median
for column in df_cleaned.columns:
    if df_cleaned[column].isnull().sum() > 0:
        median_value = df_cleaned[column].median()
        df_cleaned[column] = df_cleaned[column].fillna(median_value)

print("\n========== MISSING VALUES AFTER HANDLING ==========")
print(df_cleaned.isnull().sum())

print("\nTotal missing values after handling:")
print(df_cleaned.isnull().sum().sum())


# ============================================================
# FEATURE SELECTION
# ============================================================

# X contains input features
X = df_cleaned.drop("Potability", axis=1)

# y contains target/output column
y = df_cleaned["Potability"]


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n========== FEATURE SCALING COMPLETED ==========")
print("Features are standardized using StandardScaler.")


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ============================================================
# MODEL TRAINING - RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\n========== MODEL TRAINING COMPLETED ==========")
print("Random Forest Classifier model has been trained.")


# ============================================================
# MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== RANDOM FOREST MODEL ACCURACY ==========")
print("Accuracy:", accuracy * 100, "%")

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))


# Confusion Matrix Graph
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Water Quality Features")
plt.show()


# ============================================================
# PCA - DIMENSIONALITY REDUCTION
# ============================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
pca_df["Potability"] = y

print("\n========== PCA EXPLAINED VARIANCE RATIO ==========")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="Potability", data=pca_df)
plt.title("PCA Visualization of Water Quality Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# ============================================================
# SAMPLE PREDICTION
# ============================================================

print("\n========== SAMPLE PREDICTION ==========")

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

if prediction[0] == 1:
    print("The sample water is predicted as: POTABLE / SAFE TO DRINK")
else:
    print("The sample water is predicted as: NOT POTABLE / NOT SAFE TO DRINK")


# ============================================================
# PROJECT CONCLUSION
# ============================================================

print("\n========== CONCLUSION ==========")
print("This project analyzed water quality data using EDA and preprocessing techniques.")
print("Missing values were handled using median imputation.")
print("Features were scaled using StandardScaler.")
print("A Random Forest Classifier model was trained to predict water potability.")
print("Feature importance was used to identify important water quality parameters.")
print("PCA was applied to reduce the dataset into two principal components for visualization.")
print("Random Forest performed better because it can learn non-linear patterns from multiple water quality features.")