# 💳 Bank Customer Churn Prediction – Artificial Neural Network (ANN)

**Quick Snapshot:**

Predict bank customer churn using an **Artificial Neural Network (ANN)**. Identify high-risk customers, evaluate model performance, and visualize predictions for actionable insights.


## 📌 Project Overview

This project applies **supervised machine learning** to predict whether a bank customer will **stay or churn**, based on profile and transactional features. By identifying customers likely to leave, banks can design targeted retention strategies.

---

## 🎯 Background

Customer churn impacts profitability and long-term growth in banking. Proactive churn prediction allows banks to:
- Retain high-value customers
- Reduce attrition-related losses
- Improve customer experience through tailored interventions

---

## 🗂️ Dataset Overview

**Source:** Provided by Data Science Bootcamp (10,000 records)

**Features Used for ANN Prediction:**

| Feature            | Description                                  |
| ------------------ | -------------------------------------------- |
| Geography          | Customer country (France, Spain, Germany)    |
| Credit Score       | Customer credit score (300–850)              |
| Gender             | Male or Female                               |
| Age                | Customer age in years                        |
| Tenure             | Number of years as a bank customer           |
| Balance            | Account balance in USD                       |
| Number of Products | Total bank products held by the customer     |
| Has Credit Card    | 1 if customer has a credit card, 0 otherwise |
| Is Active Member   | 1 if active, 0 otherwise                     |
| Estimated Salary   | Annual estimated salary in USD               |


**Problem Type:** Supervised learning – Classification (Predicting churn: STAY vs CHURN)

> **⚠️ Disclaimer**
Dataset is synthetic/educational; any real-world conclusions should not be drawn from this data.

---

## 🛠️ Tools I Used

- **Python** (NumPy, Pandas, Matplotlib, Seaborn)
- **Scikit-learn** (MLPClassifier, preprocessing, metrics)
- **Jupyter Notebook** for experimentation and visualization

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11 or higher
* Jupyter Notebook or Google Colab
* Conda (recommended) or pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/NadiaRozman/Bank_Customer_Churn_ANN.git
   ```

2. **Install required packages (if not already installed):**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Open and run the notebook**

   * Navigate to `Bank_Customer_Churn_ANN.ipynb`
   * Ensure `bank_customer_churn_data.csv` is in the correct directory
   * Run all cells to reproduce the analysis

---

## 🔬 ANN Model Implementation

**1. Preprocessing:** 
   - Label Encoding, One-Hot Encoding, Feature Scaling

**2. ANN Architecture:**
   - Input Layer: 12 features
   - Hidden Layer: 100 neurons, ReLU activation
   - Output Layer: 1 neuron, Sigmoid (for binary classification)
   - Optimizer: Adam
   - Loss Function: Cross-Entropy
   - Max Iterations: 200

**3. Model Evaluation:**
   - Training Accuracy: 87.90%
   - Test Accuracy: 86.25%
   - AUC Score: 0.852 (Very good performance)

**Confusion Matrix**

![Confusion Matrix](/images/1_confusion_matrix.png)  
*Figure 1: Confusion Matrix showing model predictions vs actual labels for stayed vs churned customers.*

**ROC Curve**

![ROC Curve](/images/2_roc_curve.png)  
*Figure 2: ROC Curve of ANN Customer Churn Model. Shows trade-off between true positive rate and false positive rate. AUC = 0.852.*

---

## 📊 Customer Predictions

### **Single Customer Example**

**Customer Profile:**
   - Geography: France
   - Credit Score: 600
   - Gender: Male
   - Age: 40
   - Tenure: 3
   - Balance: $60,000
   - Number of Products: 2
   - Has Credit Card: Yes
   - Is Active Member: Yes
   - Estimated Salary: $50,000

**Prediction:**

| Metric           | Value                  |
| ----------------- | ------------------------- |
| Churn Probability | 2.53%                     |
| Risk Level        | LOW RISK                  |
| Recommendation    | Maintain standard service |


### **Multiple Customer Evaluation**

**Example Output:**

| Customer   | Churn Probability | Risk Level | Prediction |
| ---------- | ----------------- | ---------- | ---------- |
| Customer 1 | 2.53%             | LOW        | STAY       |
| Customer 2 | 52.29%            | HIGH       | CHURN      |
| Customer 3 | 94.82%            | HIGH       | CHURN      |
| Customer 4 | 99.92%            | HIGH       | CHURN      |
| Customer 5 | 8.15%             | LOW        | STAY       |

**Summary Statistics:**
- Total Customers Evaluated: 5
- High Risk (>50%): 3
- Medium Risk (30–50%): 0
- Low Risk (<30%): 2
- Average Churn Probability: 51.54%

**Churn Probability Visualization**

![Churn Probability Visualization](/images/3_multiple_customers_prediction.png)  
*Figure 3: Combined visualization showing churn probability per customer (bar chart) and risk level distribution (pie chart). Colors indicate risk levels: Low (green), Medium (orange), High (red).*

---

## 💡 Insights & Business Implications
- Customers with **low credit scores, low balances, or inactive status** are more likely to churn.
- High churn probability among some **long-term customers** may indicate service dissatisfaction or competitive offers.
- Bank can **prioritize retention campaigns** for HIGH risk customers to reduce overall churn.
- Risk-based segmentation allows **targeted incentives**, improving ROI of retention strategies.

---

## 📚 What I Learned

- ANN can handle structured tabular data effectively for classification tasks.
- Feature scaling and encoding are crucial for ANN performance.
- Visualization of predictions aids stakeholders in understanding model insights.
- Interpreting model outputs in a business context is key for actionable recommendations.

---

## 🔮 Future Enhancements

- Explore hyperparameter tuning to improve ANN performance and convergence.
- Implement cross-validation for more robust evaluation.
- Integrate additional customer features like transaction history, complaints, and product usage.
- Deploy model into a dashboard for live churn monitoring and decision-making.
- Compare ANN results with other classifiers (Random Forest, XGBoost) for benchmarking.

---

### ✨ Created by Nadia Rozman | January 2026

📂 **Project Structure**
```
ANN_Bank_Customer_Churn_Prediction/
│
├── data/
│   └── bank_customer_churn_data.csv
│
├── images/
│   ├── 1_confusion_matrix.png
│   ├── 2_roc_curve.png
│   └── 3_multiple_customers_prediction.png
│
├── notebook/
│   └── Bank_Customer_Churn_ANN.ipynb
│
└── README.md
```

**🔗 Connect with me**
- GitHub: [@NadiaRozman](https://github.com/NadiaRozman)
- LinkedIn: [Nadia Rozman](https://www.linkedin.com/in/nadia-rozman-4b4887179/)

**⭐ If you found this project helpful, please consider giving it a star!**