# Application Fraud Detection Using Machine Learning Models

## Project Overview
This project focuses on developing a machine learning-based fraud detection model for identifying fraudulent product applications. By analyzing application data and leveraging the LightGBM classifier, we achieved a **Fraud Detection Rate (FDR)** of **59.72%** at a 3% score cutoff on the out-of-time dataset. The project demonstrates how machine learning can efficiently detect fraud while minimizing false positives, ultimately saving businesses an estimated **$3.2 billion** by identifying fraudulent applications without significantly affecting legitimate sales.

---

## Dataset
The raw dataset contains 1,000,000 rows of product applications, with a mix of numerical and categorical fields. Key fields include:
- **Numerical Fields:** `date`, `dob` (date of birth)
- **Categorical Fields:** `firstname`, `lastname`, `address`, `record`, `ssn`, `zip5`, `homephone`, and the target variable `fraud_label`.

### Key Characteristics:
- **Class Imbalance:** Majority of applications are non-fraudulent (`fraud_label = 0`) with a minority of fraudulent cases (`fraud_label = 1`).
- **Placeholder Values:** Fields like `address` and `ssn` include placeholder values (e.g., `123 MAIN ST`, `999999999`), which were treated during data cleaning.

---

## Methodology

### 1. **Data Cleaning**
- Converted `date` and `dob` fields from integers to datetime format.
- Treated placeholder values in fields like `ssn`, `address`, `dob`, and `homephone` by replacing them with unique identifiers from the `record` field to reduce bias.

### 2. **Variable Creation**
Generated a comprehensive set of **651 variables** to capture temporal, behavioral, and relational patterns, including:
- **Day-Since Features:** Tracking recency of applications for entities.
- **Velocity Features:** Measuring application frequency over various timeframes.
- **Unique Counts:** Counting unique occurrences of entities linked to others.
- **Maximum Indicators:** Capturing peak values in application counts.
- **Age Indicators:** Representing the age distribution when applications were made.

### 3. **Feature Selection**
- Employed **forward selection** with LightGBM as the classifier, selecting the **top 20 variables** based on predictive power.
- Final variables included metrics like application counts, velocity features, and entity linkage indicators.

### 4. **Model Exploration**
Tested multiple machine learning models, including:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Neural Network**
- **LightGBM** (final model)

### 5. **Final Model: LightGBM**
The LightGBM classifier was selected for its superior performance and generalization across datasets. Key hyperparameters:
- **Number of Leaves:** 31
- **Maximum Depth:** Unlimited
- **Learning Rate:** 0.01
- **Number of Estimators:** 200

---

## Results and Highlights
- **Fraud Detection Rate (FDR):** 59.72% at a 3% score cutoff on the out-of-time dataset.
- **Financial Savings:** Estimated savings of **$3.2 billion** by detecting fraud while minimizing false positives.
- **Key Insights:**
  - Placeholder values (e.g., `123 MAIN ST`, `999999999`) are strongly associated with fraudulent applications.
  - Temporal and relational features (e.g., day-since, velocity metrics) play a crucial role in detecting fraud.

---

## Recommendations
1. **Threshold Optimization:** Regularly review and adjust the fraud detection cutoff to balance fraud detection and revenue preservation.
2. **Cost-Sensitive Modeling:** Incorporate cost-sensitive approaches to account for the financial impact of false positives.
3. **Additional Data Sources:** Integrate external data, such as credit scores or historical fraud patterns, to improve model accuracy.
4. **Model Monitoring:** Continuously evaluate model performance to ensure stability and adaptability to new fraud patterns.

---

## Files Included
1. **applications_explore_clean.ipynb:** Data exploration and cleaning steps.
2. **applications_make_variables.ipynb:** Variable creation and feature engineering processes.
3. **feature_selection_binary_classification.ipynb:** Feature selection using LightGBM.
4. **binary_classification_models.ipynb:** Model exploration and evaluation.
5. **Project_Report_2.pdf:** Detailed project report, including methodology, results, and recommendations.

---

## Acknowledgments
This project was completed as part of the **DSO 562: Predictive Analytics** course at the University of Southern California. I would like to thank Professor Stephen Coggashell for providing the foundational code and guidance that supported the development of this project. 
