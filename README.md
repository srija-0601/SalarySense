# SalarySense – Salary Prediction & Data Analytics

SalarySense is an end-to-end Python machine learning project that analyzes employee salary data and predicts salary based on professional and employment-related attributes.

The project covers the complete machine learning workflow, including **data cleaning, exploratory data analysis, visualization, feature preprocessing, model training, evaluation, and interactive salary prediction**.

## 🎯 Objective

The objective of SalarySense is to build a machine learning system that can:

* Clean and preprocess salary-related data
* Identify patterns and relationships in salary data
* Analyze salary distributions across different categories
* Compare different machine learning regression models
* Predict salary based on user-provided employee attributes

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine learning and model evaluation
* **Git & GitHub** – Version control

## 📊 Dataset

The project uses a **synthetic dataset containing 5,000 original employee records**, with additional duplicate records intentionally included to demonstrate data-cleaning techniques.

### Features

| Feature         | Description                      |
| --------------- | -------------------------------- |
| Age             | Employee age                     |
| Education       | Education level                  |
| YearsExperience | Years of professional experience |
| JobRole         | Employee's job role              |
| Location        | Employee location                |
| EmploymentType  | Type of employment               |
| CompanySize     | Size of the company              |
| Salary          | Target salary value              |

> **Note:** The dataset is synthetic and created for learning and machine learning practice. It does not represent real company or employee salary data.

## 🧹 Data Preprocessing

The project performs several preprocessing steps:

* Detects duplicate records
* Removes duplicate records
* Identifies missing values
* Handles missing categorical values using **mode**
* Handles missing numerical values using **median**
* Separates features and target variable
* Applies **One-Hot Encoding** to categorical features

## 📈 Exploratory Data Analysis

SalarySense generates visualizations to understand salary patterns:

* Salary Distribution
* Salary vs. Years of Experience
* Average Salary by Job Role
* Average Salary by Education Level

These visualizations help identify relationships and trends within the dataset.

## 🤖 Machine Learning Models

Two regression models are implemented and compared:

### 1. Linear Regression

Used as a baseline regression model to understand the relationship between employee attributes and salary.

### 2. Random Forest Regressor

An ensemble learning model used to capture more complex relationships between the input features and salary.

## 📏 Model Evaluation

The models are evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

A model comparison file is automatically generated at:

```text
outputs/model_comparison.csv
```

## 💻 Interactive Salary Prediction

SalarySense also provides an interactive command-line prediction system.

The user enters:

```text
Age
Education
Years of Experience
Job Role
Location
Employment Type
Company Size
```

The trained Random Forest model then generates a predicted salary.

Example:

```text
===================================
      SalarySense Predictor
===================================

Enter Age: 25
Enter Education: Bachelor's
Enter Years of Experience: 2
Enter Job Role: Software Engineer
Enter Location: Hyderabad
Enter Employment Type: Full-Time
Enter Company Size: Large

Predicted Salary: ₹XX,XXX.XX
```

## 📁 Project Structure

```text
SalarySense/
│
├── data/
│   └── salary_data_5000.csv
│
├── src/
│   └── salary_prediction.py
│
├── outputs/
│   ├── model_comparison.csv
│   ├── salary_distribution.png
│   ├── salary_vs_experience.png
│   ├── salary_by_job_role.png
│   └── salary_by_education.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/srija-0601/SalarySense.git
```

### 2. Navigate to the project

```bash
cd SalarySense
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python src/salary_prediction.py
```

The program will perform data preprocessing, generate visualizations, train the models, display evaluation metrics, and start the interactive salary predictor.

## 🔮 Future Enhancements

* Build an interactive **Power BI/Tableau dashboard**
* Add additional regression algorithms
* Implement hyperparameter tuning
* Add cross-validation
* Deploy the prediction system as a web application
* Add a graphical user interface for salary prediction

## 👩‍💻 Author

**Srija Kosigi**

B.Tech Computer Science & Engineering
VIT-AP

---

### GitHub repository description

**End-to-end salary prediction and data analytics project using Python, Pandas, NumPy, Scikit-learn, and data visualization techniques.**
