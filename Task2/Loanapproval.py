import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
df = pd.read_csv(r"D:\3. Internships\ShadowFox\Task2\loan_prediction.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset Shape:", df.shape)
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)
cat_cols = [
    'Gender',
    'Married',
    'Dependents',
    'Self_Employed',
    'Credit_History'
]
for col in cat_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
num_cols = [
    'LoanAmount',
    'Loan_Amount_Term'
]
for col in num_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)
outlier_cols = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount'
]
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df = df[
        (df[col] >= lower_limit) &
        (df[col] <= upper_limit)
    ]
print("\nShape after removing outliers:", df.shape)
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Status Distribution")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Loan_Status', data=df)
plt.title("Gender vs Loan Status")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.title("Education vs Loan Status")
plt.show()
df['TotalIncome'] = (
    df['ApplicantIncome'] +
    df['CoapplicantIncome']
)
df['Income_Loan_Ratio'] = (
    df['TotalIncome'] /
    df['LoanAmount']
)
df['EMI'] = (
    df['LoanAmount'] /
    df['Loan_Amount_Term']
)
df['ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
df['CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])
df['LoanAmount'] = np.log1p(df['LoanAmount'])
df['TotalIncome'] = np.log1p(df['TotalIncome'])
df['Loan_Status'] = df['Loan_Status'].map({
    'Y': 1,
    'N': 0
})
df = pd.get_dummies(df, drop_first=True)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [6, 8, 10, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt']
}
rf = RandomForestClassifier(
    random_state=42
)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy =", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
})
importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)
print("\nTop 10 Important Features:")
print(importance_df.head(10))
plt.figure(figsize=(8,6))
sns.barplot(
    data=importance_df.head(10),
    x='Importance',
    y='Feature'
)
plt.title("Top 10 Important Features")
plt.show()
sample_predictions = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print("\nSample Predictions:")
print(sample_predictions.head(20))