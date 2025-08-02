# <center> **`Banking Insights: Marketing Analysis & Prediction of Term Deposit Subscriptions`**
## 📈 Overview

This project builds a machine learning pipeline to predict customer responses to term deposit marketing campaigns. By analyzing customer demographic and behavioral data, the model helps banks strategically identify potential subscribers, minimize missed opportunities, and optimize campaign resources.

---

## 🔍 Problem Statement

Banks often struggle with inefficiencies in marketing outreach, where broad campaigns fail to engage the right audience, wasting resources and missing key opportunities. In term deposit campaigns, the cost of missing a potential subscriber (False Negative) can be especially high. Traditional targeting approaches do not sufficiently leverage the rich patterns in customer data to isolate promising leads. This calls for a predictive, data-driven model that prioritizes customer engagement while optimizing operational efficiency.

---

## 🎯 Project Objectives
To develop a machine learning pipeline that predicts customer responses to term deposit marketing campaigns by analyzing demographic and behavioral data, with a strong emphasis on:
- 🎯 **Maximize Recall** - Reduce False Negatives to avoid missing likely subscribers.
- 🎯 **Preserve Precision** - Focus on relevant targets to reduce outreach waste.
- 🎯 **Ensure Fairness** - Apply SMOTE to address class imbalance and promote inclusive modeling.

---

## 🧾 Dataset Summary

The dataset is about **45211** bank customers which includes **17** features per customer.

### 1. 👤 Client Info
| Feature     | Description                         |
|-------------|-------------------------------------|
| age         | Customer's age (numeric)            |
| job         | Type of job                         |
| marital     | Marital status                      |
| education   | Education level                     |
| default     | Has credit in default?              |
| balance     | Yearly balance (in euros)           |
| housing     | Has housing loan?                   |
| loan        | Has personal loan?                  |

### 2. 📞 Campaign Contact Info
| Feature   | Description                                  |
|-----------|----------------------------------------------|
| contact   | Contact type (cellular, telephone, etc.)     |
| day       | Last contact day of the month                |
| month     | Month of last contact                        |
| duration  | Duration of contact in seconds               |

### 3. 📊 Campaign History
| Feature   | Description                                               |
|-----------|-----------------------------------------------------------|
| campaign  | # Contacts in current campaign                            |
| pdays     | Days since last contact (-1 = never contacted before)     |
| previous  | # Past contacts before this campaign                      |
| poutcome  | Outcome of previous campaign                              |

### 4. 🎯 Target
| Feature | Description                     |
|---------|---------------------------------|
| y       | Subscribed to deposit? (yes/no) |

---
## Folder Structure
banking-campaign-subscription-predictor/
├── 📁 data/                                                           # Datasets used for analysis
│   ├── 📁 raw/                              
│       └── bank_call_data.csv                                          # Original Dataset
|                        
|   ├── 📁 processed/ 
│       └── data.csv                                                    # Datset retained for reference
|       └── data_basic_cleaning.csv                                     # Cleaned Dataset
|       └── data_eda.csv                                                # Dataset preserved exclusively for EDA
|
├── 📁 notebooks/                                                       
│   └── 01_data_cleaning_and_eda.ipynb                                  # Data Cleaning and EDA notebook
|   └── 02_feature_engineering_and_model_training_evaluation.ipynb      # Feature Engineering and modeling notebook
│
├── README.md                                                           # Project overview and instructions
└── LICENSE                                                             # License information

---
## ⚙️ Methodology

1. Exploratory Data Analysis
2. Data Cleaning & Feature Engineering
3. Model Building & Evaluation

## ⚙️ Pipeline Architecture

- **Data Cleaning & Preprocessing**
  - Basic Data Cleaning
  - Encode categoricals (One-Hot/Ordinal)
  - Handle missing features gracefully
  - Feature scaling

- **Custom Transformers**
  - Modular preprocessing with ColumnTransformer
  - Semantic enrichment for time & contact metadata

- **Resampling**
  - Apply **SMOTE** on training data for class balance

- **Model Training**
  - Logistic Regression / K-Nearest Neighbor / Decision Tree / Random Forest / Gradient Boosting / XGBoost

- **Evaluation**
  - Metrics: Recall, Precision, F1, ROC-AUC, Precision
---

## 🧪 Model Evaluation Strategy

| Metric           | Priority | Purpose                                     |
|------------------|----------|---------------------------------------------|
| **Recall**       | High     | Minimize False Negatives (missed responders) |
| **Precision**    | Medium   | Avoid irrelevant outreach                    |
| **F1 Score**     | Medium   | Balanced metric                             |
| **ROC-AUC**      | Low      | General classifier performance              |

---

## 🧠 Campaign Strategy Recommendations

### 1. 📅 Leverage Timing Signals
- Features like **month** and **duration** are critical; when and how long customers were contacted matters.
- **Recommendation**: Launch **time-sensitive campaigns** in top-performing months, targeting high-duration contacts.
- Consider clustering by **duration_group** to optimize contact cadence.

### 2. 🔁 Retarget Based on Past Success
- **poutcome_success** emerged as a strong indicator of future responsiveness.
- **Recommendation**: Re-engage leads who responded in prior campaigns, using **personalized offers or priority messaging**.

### 3. 👥 Segment-Based Campaigns
- Behavioral and socio-economic attributes (e.g. job groups) shape receptiveness.
- **Recommendation**: Build **segment-specific messaging strategies** to improve conversion and reduce churn.

---

## ✅ Model Conclusion

The final **XGBoost** model, optimized at threshold **0.50**, delivers:

- **Accuracy**: **88.1%** - Solid baseline performance
- **ROC-AUC Score**: **0.905** - Excellent class separability
- **Recall (Responders)**: **69%** - High coverage of potential subscribers
- **Precision (Responders)**: **49%** - Acceptable given recall priority

This model strikes a **practical balance** between expanding outreach and minimizing resource waste, making it well-suited for cost-sensitive marketing environments.

---

## 💼 Business Implications

- Enables **high-confidence targeting** of likely responders
- Boosts **conversion rates** and **campaign ROI**
- Supports **cost-effective expansion** of customer outreach

---

## 🔮 Recommended Next Steps

- 🔧 **Threshold Tuning**: Align cutoff levels with specific campaign goals
- 🧮 **Lead Ranking Strategy**: Rank customers using probability scores for budget-conscious targeting
- 🧠 **Segmented Outreach**: Craft personalized messages using key features like duration, month, and prior outcomes

---

## 🛠 Tech Stack

- **Languages**: Python 3.x
- **Libraries**:
    - Data Handling: `pandas`, `numpy`
    - Visualization: `matplotlib`, `seaborn`
    - Modeling: `scikit-learn`, `xgboost`
    - Evaluation: `classification_report`, `confusion_matrix`, `roc_auc_score`
- **Techniques**: imbalanced-learn (`SMOTE`)
- **ML Algorithms**: `XGBoost`, `Random Forest`, `Gradient Boost`, `Decision Tree`, `Logistic Regression`
- **Environment**: `Jupyter Notebook`

---
## 🤝 Contributing

Contributions for model enhancements, interpretability features, or business-aligned optimizations are welcome. Feel free to open a PR or drop suggestions!

---

## 📬 Contact

Project Lead: **Aniket**  
Let’s connect and advance strategic machine learning together 🚀
- 📧 Email: [aniketmuthal4@gmail.com]
- 🔗 LinkedIn: [https://www.linkedin.com/in/aniket-muthal]
