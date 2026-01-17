# 💰 Loan Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A **Machine Learning classification project** that predicts whether a loan application will be **approved (1)** or **rejected (0)** using applicant financial, credit, and demographic information.

This project follows a **traditional machine learning approach** with clear, step-by-step preprocessing, model training, evaluation, and interpretation.

---

## 🚀 Features

✅ **Data Preprocessing (Traditional ML):**
- Removed irrelevant identifiers (`Loan_ID`)
- Handled missing values  
  - Numerical features → Median imputation  
  - Categorical features → Mode imputation  
- Encoded categorical variables using Label Encoding  
- Standardized numerical features using `StandardScaler`

✅ **Model Training & Comparison:**
Trained and evaluated multiple classifiers:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

Evaluation metrics include:
- Accuracy  
- Precision, Recall, F1-score  
- ROC-AUC score  
- Confusion Matrix  

✅ **Model Interpretation:**
- Feature importance analysis using Random Forest  
- Identified **Credit_History**, **ApplicantIncome**, and **LoanAmount** as the most influential features

✅ **Visualization & Analysis:**
- Class distribution plot to highlight dataset imbalance  
- Confusion matrix visualization to analyze prediction errors  
- ROC curve to study classification trade-offs  
- Feature importance bar chart for explainability  

---

## 🧠 Key Observations

- The dataset is **imbalanced**, with significantly more approved loans than rejected ones.
- All models show **high recall for approved loans (class 1)** and **lower recall for rejected loans (class 0)**.
- This behavior occurs due to:
  - Class imbalance  
  - Strong dominance of the `Credit_History` feature  
- The results reflect a **realistic trade-off** commonly seen in financial risk prediction problems.

---

## 🧩 Project Structure

| File | Description |
|-----|------------|
| `loan_prediction.csv` | Dataset containing applicant and loan details |
| `index.py` | Main script for data preprocessing, model training, evaluation, and visualization |
| `README.md` | Project documentation |
| `Images` | Result Visualisation |


---

## 📊 Dataset

- **Source:** Kaggle – Loan Prediction Dataset  
- **Records:** 614 loan applications  
- **Target Variable:** `Loan_Status`  
  - `1` → Loan Approved  
  - `0` → Loan Rejected  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- scikit-learn  
- Matplotlib  

---

## 📌 Conclusion

This project demonstrates a **realistic and interpretable machine learning solution** for loan approval prediction.  
Rather than forcing metric optimization, the model behavior is analyzed and justified using data characteristics and business logic.

---

## 🔮 Future Improvements

- Improve recall for rejected loans using class weighting or resampling techniques  
- Perform hyperparameter tuning  
- Add decision threshold tuning for risk-sensitive predictions  
- Deploy the model using Flask or Streamlit  

---

## 👤 Author

**Satyam Yadav**  
Computer Science Student | Aspiring Machine Learning Engineer
