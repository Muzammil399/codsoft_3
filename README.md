#  Subscription Service Customer Churn Prediction

##  Overview

This project aims to predict which customers are likely to **cancel their subscription (churn)** using machine learning.  
By identifying high-risk customers early, the system helps businesses take targeted actions to improve customer retention.

We compare three popular classification models — **Logistic Regression**, **Random Forest**, and **Gradient Boosting** — to find the most effective approach.

---
##  Data & Features

The model is trained on the **Churn_Modelling.csv** dataset, which contains historical customer records.  
Here’s what the data represents:

###  Demographics  
- **Geography** – Customer’s country or region  
- **Gender** – Male or Female  
- **Age** – Customer’s age  

###  Account Information  
- **CreditScore** – Customer’s credit rating  
- **Tenure** – Number of years as a customer  
- **Balance** – Account balance  
- **NumOfProducts** – Number of subscribed products  
- **IsActiveMember** – Indicates if the customer is currently active  

### Target Variable  
- **Exited** – 1 if the customer churned, 0 if retained  

---

##  Methodology

All input features are processed using a **scikit-learn pipeline**, ensuring clean and consistent data flow.  
The pipeline includes:  
- **Standard Scaling** for numerical features  
- **One-Hot Encoding** for categorical features  

We trained and evaluated the following models:

| Model | Type | Strength |
|--------|------|-----------|
| **Logistic Regression** | Linear | Easy to interpret and analyze feature importance |
| **Random Forest** | Ensemble (Bagging) | Handles complex relationships and reduces overfitting |
| **Gradient Boosting** | Ensemble (Boosting) | Typically offers the highest predictive accuracy |

---

##  Evaluation Metrics

Because churn datasets are often **imbalanced**, accuracy alone isn’t enough.  
We focus on metrics that better reflect real-world performance:

- **ROC AUC Score** – Measures the model’s ability to distinguish between churned and retained customers.  
- **Recall (Churn = 1)** – The percentage of actual churners the model correctly identifies (important for catching high-risk users).  
- **Precision (Churn = 1)** – The percentage of predicted churners who truly left (reducing false alarms).

---


Install the dependencies before running the script:

```bash
pip install pandas scikit-learn numpy
