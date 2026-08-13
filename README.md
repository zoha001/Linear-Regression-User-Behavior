#  Predicting Smartphone Data Usage with Linear Regression
### *Machine Learning for Data Analysis — Lab *

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-11557c?logo=python)](https://matplotlib.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?logo=github)]()
[![Course](https://img.shields.io/badge/Course-DS%20402-9cf)]()

>  **Goal:** Build a **Simple Linear Regression model** to predict a smartphone user's **daily data usage (MB/day)** based on the **number of apps installed** — and evaluate how well the model performs.

---

## 🔍 Overview

In today's smartphone-driven world, understanding what drives data consumption is valuable for telecom companies and app developers. This project explores the relationship between the **number of apps installed** on a phone and the user's **daily mobile data usage**.

Using a dataset of **700 smartphone users**, this project:
-  Explores the data (shape, types, statistics)
-  Visualizes the relationship with a scatter plot
-  Measures correlation between features
-  Trains a **Linear Regression** model
-  Evaluates the model using **MAE, MSE & RMSE**

---

##  Dataset

| Attribute | Description |
|---|---|
| **File** | `user_behavior_dataset.csv` |
| **Records** | 700 users |
| **Features** | 11 columns |
| **Target Variable** | Data Usage (MB/day) |
| **Predictor Variable** | Number of Apps Installed |

### Sample Preview

| User ID | Number of Apps Installed | Screen On Time | Age | Gender | Data Usage (MB/day) | User Behavior Class |
|---|---|---|---|---|---|---|
| 1 | 12 | 180 | 25 | Male | 322 | 4 |
| 2 | 32 | 450 | 42 | Male | 731 | 3 |
| 3 | 20 | 380 | 56 | Male | 412 | 2 |
| 4 | 87 | 720 | 31 | Female | 988 | 3 |

---

##  Methodology

```
┌────────────────────────────────────────────────────┐
│                MACHINE LEARNING PIPELINE            │
├────────────────────────────────────────────────────┤
│                                                    │
│  STEP 1: Load & Explore Dataset                    │
│      └─→ df.head(), df.shape, df.dtypes            │
│                                                    │
│  STEP 2: Exploratory Data Analysis (EDA)           │
│      ├─→ Scatter Plot (Apps vs Data Usage)         │
│      ├─→ Correlation Analysis (r = 0.93)           │
│      └─→ Descriptive Statistics                    │
│                                                    │
│  STEP 3: Feature Selection                         │
│      ├─→ X = Number of Apps Installed              │
│      └─→ y = Data Usage (MB/day)                   │
│                                                    │
│  STEP 4: Train-Test Split                          │
│      └─→ 80% training, 20% testing (seed=42)       │
│                                                    │
│  STEP 5: Train Linear Regression Model             │
│      └─→ model.fit(X_train, y_train)               │
│                                                    │
│  STEP 6: Predict & Evaluate                        │
│      ├─→ y_pred = model.predict(X_test)            │
│      └─→ MAE, MSE, RMSE                            │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 💻 Step-by-Step Code Explanation

###  Step 1 — Loading & Exploring the Dataset
```python
import pandas as pd
df = pd.read_csv('user_behavior_dataset.csv')
df.head()      # First 5 rows
df.shape       # (700, 11)
df.dtypes      # Data types of each column
```
Loads the CSV file and gives an overview of the dataset structure.

###  Step 2 — Scatter Plot Visualization
```python
df.plot.scatter(x='Number of Apps Installed',
                y='Data Usage (MB/day)',
                title='Scatter Plot of Apps Installed vs Data Usage')
```
Shows a **strong positive linear trend** — more apps = more data usage.

###  Step 3 — Correlation & Statistics
```python
df['Number of Apps Installed'].corr(df['Data Usage (MB/day)'])
# Output: 0.9348
```
A correlation of **0.93** confirms a very strong positive relationship.

###  Step 4 — Preparing Features
```python
X = df[['Number of Apps Installed']]   # Independent variable
y = df['Data Usage (MB/day)']          # Dependent variable
```

###  Step 5 — Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```
Splits data into **80% training** and **20% testing** for unbiased evaluation.

###  Step 6 — Training the Model
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```
Fits the regression line to the training data.

###  Step 7 — Model Parameters
```python
print(model.intercept_)    # -197.62
print(model.coef_)         # 22.30
```
**Regression Equation:**
```
Data Usage = 22.30 × (Number of Apps) − 197.62
```
> 💡 Each additional app adds approximately **22.30 MB/day** of data usage.

###  Step 8 — Prediction & Evaluation
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
```

---

##  Results & Evaluation

###  Model Equation
```
Data Usage (MB/day) = 22.30 × Apps Installed − 197.62
```

###  Evaluation Metrics

| Metric | Formula Meaning | Value |
|---|---|---|
| **Correlation (r)** | Strength of linear relationship | **0.9348** |
| **MAE** | Average prediction error | **170.58** |
| **MSE** | Mean of squared errors | **50221.79** |
| **RMSE** | Root mean squared error | **224.10** |

###  Example Prediction
For a user with **9.5 apps** (from notebook calculation):
```
Predicted Data Usage ≈ 14.21 MB/day
```

---

## 📊 Visualization

![Scatter Plot](scatter_plot.png)

*The scatter plot reveals a clear positive linear relationship between the number of apps installed and daily data usage.*

---

##  Key Insights

### Insight 1: Strong Linear Relationship
With a correlation of **0.93**, the number of apps installed is an excellent predictor of data usage.

### Insight 2: Each App Adds Data Consumption
The coefficient **22.30** shows that every additional app increases daily usage by about 22 MB.

### Insight 3: Good Model Fit
The low MAE (**170.58**) relative to typical usage values shows the model predicts reasonably well.

### Insight 4: Business Application
Telecom companies can use this to **forecast network load** and design better data plans based on app usage patterns.

---

##  How to Run

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Steps
```bash
# 1. Clone the repository
git clone https://github.com/[YOUR_USERNAME]/DS402-Lab2-Linear-Regression-User-Behavior.git
cd DS402-Lab2-Linear-Regression-User-Behavior

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook zoha_390621_codes_lab2.ipynb

# 4. Run all cells (Kernel → Run All)
```

### 📄 requirements.txt
```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

---

##  Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3** | Programming language |
| **Pandas** | Data loading & manipulation |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |
| **scikit-learn** | Linear Regression model & metrics |
| **Jupyter Notebook** | Interactive analysis environment |

---
