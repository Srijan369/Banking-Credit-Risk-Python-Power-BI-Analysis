# 📊 Banking Credit Risk Analysis

## 🎯 Project Overview

This project focuses on analyzing loan applications to identify **credit risk, default patterns, and customer behavior**.

The goal is to help financial institutions:

- Reduce loan defaults
- Improve risk assessment
- Make data-driven lending decisions

The analysis is based on **148,670 loan records** with multiple financial and customer attributes.

---

## 🛠️ Tools & Technologies Used

- **Python (Pandas, NumPy)** – Data cleaning & preprocessing
- **Matplotlib & Seaborn** – Data visualization
- **Power BI** – Interactive dashboard creation
- **DAX** – KPI calculations & advanced analysis

---

## 📂 Dataset Information

- **Source:** Loan Default Dataset
- **Total Records:** 148,670
- **Features:** 30+ columns including:
  - Loan Amount
  - Interest Rate
  - Credit Score
  - Income
  - LTV (Loan-to-Value)
  - DTI (Debt-to-Income Ratio)
  - Loan Purpose
  - Region

---

## ⚙️ Data Processing (Python Pipeline)

The dataset was cleaned and transformed using a complete Python workflow :

### 🔹 Data Cleaning

- Handled missing values (median for numeric, mode for categorical)
- Removed duplicate records
- Standardized categorical values
- Converted data types

### 🔹 Feature Engineering

- Created **Credit Score Categories**
- Created **LTV Categories**
- Created **DTI Risk Categories**
- Developed a **Composite Risk Score** using:
  - Credit Score
  - DTI Ratio
  - LTV Ratio
  - Loan Purpose

### 🔹 Risk Categorization

- Very Low Risk
- Low Risk
- Medium Risk
- High Risk
- Very High Risk

---

## 📊 Dashboard Structure (Power BI)

### 🔵 Page 1: Loan Overview

- Total Loans, Total Loan Amount, Avg Interest Rate
- Loan Distribution by Purpose
- Region-wise Analysis
- Default vs Non-default

---

### 🟢 Page 2: Risk Analysis

- Risk Category Distribution
- Default by Risk Category
- Credit Score vs Default
- LTV & DTI Risk Analysis

---

### 🟠 Page 3: Customer Analysis

- Gender Distribution
- Age Group Analysis
- Income vs Loan Amount
- Occupancy Type & Income Distribution

---

## 📈 Key Insights

- 📉 **Higher default rates observed in low credit score customers**
- 📊 **DTI > 40% significantly increases default risk**
- 🏠 **High LTV loans show higher probability of default**
- ⚠️ **Certain loan purposes are riskier than others**
- 🎯 **Composite risk score effectively identifies high-risk customers**

---

## 💡 Business Impact

- Helps banks identify **high-risk applicants**
- Improves **loan approval decisions**
- Reduces **financial losses due to defaults**
- Supports **risk-based pricing strategies**

---

## 🚀 Features of the Project

- End-to-end data analysis pipeline
- Advanced feature engineering
- Risk scoring model
- Interactive Power BI dashboard
- Business insights & recommendations

---

## 📁 Output Files

- `cleaned_loan_data.csv` → Cleaned dataset
- Power BI Dashboard → Interactive analysis
- Visual charts & insights

---

## 🎯 Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Visualization
- Business Insight Generation
- Dashboard Development

---

## Dashboard Screenshot

<img width="1289" height="733" alt="image" src="https://github.com/user-attachments/assets/c6499f95-ad0c-4926-bcdd-7b9a6f65449d" />

<img width="1288" height="735" alt="image" src="https://github.com/user-attachments/assets/5d279e6e-9599-4637-b1ed-761959748cc6" />

<img width="1292" height="738" alt="image" src="https://github.com/user-attachments/assets/637b3e58-c9ae-49d7-a229-c59cbc61eb21" />

---

## 📌 Conclusion

This project demonstrates how data analytics can be used to **analyze financial risk, improve decision-making, and optimize lending strategies**.

---

## 🔗 Author

**Srijan Yadav**
Aspiring Data Analyst | Python | Power BI

---
