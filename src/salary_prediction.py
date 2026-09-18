import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

print("SalarySense Project Started!")

# Load dataset
df = pd.read_csv("data/salary_data_5000.csv")

print("\nOriginal Dataset Shape:", df.shape)

# Check duplicates
duplicate_count = df.duplicated().sum()
print("\nDuplicate Records:", duplicate_count)

# Remove duplicates
df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing categorical values with mode
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Fill missing numerical values with median
numerical_columns = df.select_dtypes(include=[np.number]).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Dataset Shape:", df.shape)

print("\nData Cleaning Completed Successfully!")
# Salary Distribution
plt.figure(figsize=(8, 5))

sns.histplot(df["Salary"], bins=30, kde=True)

plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("outputs/salary_distribution.png")
plt.close()
# Salary vs Years of Experience
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=df["YearsExperience"],
    y=df["Salary"]
)

plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.tight_layout()
plt.savefig("outputs/salary_vs_experience.png")
plt.close()
# Average Salary by Job Role
plt.figure(figsize=(10, 6))

avg_salary_role = df.groupby("JobRole")["Salary"].mean().sort_values(ascending=False)

sns.barplot(
    x=avg_salary_role.values,
    y=avg_salary_role.index
)

plt.title("Average Salary by Job Role")
plt.xlabel("Average Salary")
plt.ylabel("Job Role")

plt.tight_layout()
plt.savefig("outputs/salary_by_job_role.png")
plt.close()
# Average Salary by Education
plt.figure(figsize=(8, 5))

avg_salary_education = (
    df.groupby("Education")["Salary"]
    .mean()
    .sort_values(ascending=False)
)

sns.barplot(
    x=avg_salary_education.index,
    y=avg_salary_education.values
)

plt.title("Average Salary by Education Level")
plt.xlabel("Education")
plt.ylabel("Average Salary")

plt.tight_layout()
plt.savefig("outputs/salary_by_education.png")
plt.close()
# Prepare data for Machine Learning

X = df.drop("Salary", axis=1)
y = df["Salary"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Column:")
print(y.name)
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)

# One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

print("\nPreprocessing setup completed successfully!")
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Create Linear Regression pipeline
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# Train the model
linear_model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions
y_pred = linear_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Linear Regression Results ---")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
from sklearn.ensemble import RandomForestRegressor

# Create Random Forest pipeline
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Train the model
random_forest_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")
# Make predictions using Random Forest
rf_pred = random_forest_model.predict(X_test)

# Calculate evaluation metrics
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Random Forest Results ---")
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R²   : {rf_r2:.4f}")
# Model Comparison

model_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae, rf_mae],
    "RMSE": [rmse, rf_rmse],
    "R2 Score": [r2, rf_r2]
})

print("\n--- Model Comparison ---")
print(model_comparison)
# Save model comparison results

model_comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to model_comparison.csv")
# Interactive Salary Prediction

print("\n===================================")
print("      SalarySense Predictor")
print("===================================")

age = int(input("Enter Age: "))
education = input("Enter Education: ")
years_experience = float(input("Enter Years of Experience: "))
job_role = input("Enter Job Role: ")
location = input("Enter Location: ")
employment_type = input("Enter Employment Type: ")
company_size = input("Enter Company Size: ")

user_data = pd.DataFrame({
    "Age": [age],
    "Education": [education],
    "YearsExperience": [years_experience],
    "JobRole": [job_role],
    "Location": [location],
    "EmploymentType": [employment_type],
    "CompanySize": [company_size]
})

predicted_salary = random_forest_model.predict(user_data)[0]

print("\n===================================")
print(f"Predicted Salary: ₹{predicted_salary:,.2f}")
print("===================================")