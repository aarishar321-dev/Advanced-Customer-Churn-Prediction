# Advanced Customer Churn Prediction

## End-to-End Machine Learning & Customer Retention Analytics

An end-to-end telecom customer churn prediction system designed to identify customers at risk of leaving and, more importantly, translate those predictions into **actionable customer retention decisions**.

Unlike a typical customer churn project that focuses mainly on achieving the highest classification accuracy, this project focuses on the complete journey from **customer behavior analysis → feature engineering → model comparison → probability-based risk assessment → threshold optimization → business-oriented retention decisions → deployment**.

---

## Project Overview

Customer churn is one of the major challenges faced by telecom companies.

Acquiring a new customer is generally more expensive than retaining an existing one. Therefore, simply predicting whether a customer will churn is not enough.

A useful churn prediction system should help answer:

- Which customers are most likely to churn?
- What customer characteristics are associated with churn?
- Which model performs best for this dataset?
- How should the model's probability output be converted into a business decision?
- Should the model prioritize precision or recall?
- How many potential churners can be identified?
- How can the predictions support customer retention campaigns?

This project was built around these questions.

The final system uses **Logistic Regression** as the selected model and applies a **0.30 probability threshold** to prioritize churn detection and improve recall.

The trained model and preprocessing pipeline are integrated into a **Streamlit application** for interactive customer-level churn risk prediction.

---

# What Makes This Project Different?

Most beginner-level customer churn projects follow a relatively simple workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
EDA
   ↓
Train Model
   ↓
Accuracy
   ↓
Prediction
```

This project follows a more complete machine learning and business workflow:

```text
Business Problem
        ↓
Customer & Churn Analysis
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Preprocessing Pipeline
        ↓
Multiple Model Evaluation
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Probability Analysis
        ↓
Threshold Optimization
        ↓
Business-Oriented Model Selection
        ↓
Final Evaluation
        ↓
Model Serialization
        ↓
Streamlit Deployment
        ↓
Customer Retention Decision
```

The main difference is that the project does not treat **0.50 as an unquestioned prediction threshold**.

Instead, the model's probability output is analyzed and a threshold is selected according to the business objective.

---

# Project Objective

The primary objective is to build a customer churn prediction system that can:

1. Analyze customer behavior and identify churn patterns.
2. Engineer meaningful customer-level features.
3. Compare multiple machine learning algorithms.
4. Select an appropriate final model based on multiple evaluation metrics.
5. Tune the selected model using cross-validation.
6. Analyze prediction probabilities instead of relying only on class labels.
7. Select a business-oriented probability threshold.
8. Improve the identification of potential churners.
9. Provide customer-level churn risk predictions.
10. Deploy the complete solution through Streamlit.

---

# Business Problem

A telecom company has thousands of customers with different:

- Tenure
- Contract types
- Internet services
- Payment methods
- Monthly charges
- Support services
- Streaming services
- Customer demographics

Some customers are likely to leave, while others are likely to remain.

The business does not want to contact every customer with a retention offer.

Instead, it wants to identify customers who have a sufficiently high probability of churn so that retention resources can be prioritized.

Therefore, the machine learning problem becomes:

> **Predict the probability that a customer will churn and convert that probability into a practical retention decision.**

---

# Dataset

The project uses a telecom customer churn dataset containing:

- **7,043 customers**
- **23 original features**
- Customer demographic information
- Account information
- Service information
- Billing information
- Churn outcome

The target variable is:

```text
Churn
```

with two possible outcomes:

```text
Yes
No
```

---

# Feature Categories

## Customer Demographics

- Gender
- Senior Citizen
- Partner
- Dependents

## Customer Relationship

- Tenure

## Telecom Services

- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

## Contract & Billing

- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

# Feature Engineering

Instead of relying only on the original dataset features, additional features were created to capture customer behavior in a more meaningful way.

## 1. ServiceCount

Counts the number of additional services subscribed to by a customer.

This provides a simple measure of service engagement.

Example:

```text
Customer A → 0 additional services
Customer B → 3 additional services
Customer C → 6 additional services
```

The analysis showed that service engagement has a relationship with churn behavior.

---

## 2. IsNewCustomer

A binary feature was created to identify customers with very short tenure.

```text
Tenure <= 6 months → 1
Tenure > 6 months  → 0
```

This feature was created to explicitly capture the early-customer retention segment.

The analysis showed that newer customers have considerably higher churn risk.

---

## 3. TenureGroup

Tenure was additionally transformed into customer tenure groups.

This allows the model and analysis to capture broader customer lifecycle patterns instead of relying only on the raw tenure value.

---

## 4. MonthlyChargeGroup

Monthly charges were grouped into customer charge segments.

This helps identify whether customers in different pricing ranges exhibit different churn behavior.

---

# Exploratory Data Analysis

EDA was performed from a **business perspective**, rather than simply generating plots.

The analysis examined churn across:

- Gender
- Contract type
- Payment method
- Internet service
- Support services
- Security services
- Paperless billing
- Tenure
- Monthly charges
- Total charges
- Service count
- New customer status

---

# Important Business Insights

Several meaningful churn patterns were identified.

### Contract Type

Month-to-month customers showed substantially higher churn than customers with one-year and two-year contracts.

This suggests that longer contractual commitment is strongly associated with retention.

---

### Payment Method

Electronic-check customers showed the highest churn rate among payment methods.

This identifies payment method as a potentially useful customer segmentation variable for retention analysis.

---

### Internet Service

Fiber-optic customers showed higher churn compared with other internet-service groups.

This makes the service category worth investigating further from a customer experience and pricing perspective.

---

### Support Services

Customers without services such as:

- Online Security
- Tech Support

showed higher churn patterns.

This suggests that service bundling and support engagement may be relevant retention factors.

---

### Monthly Charges

Churned customers generally showed higher monthly charges than retained customers.

This indicates that pricing and perceived value may be relevant factors in churn.

---

### Service Count

Customers with only one additional service showed particularly high churn.

Interestingly, having more services was associated with lower churn in the analyzed dataset.

This can support the business hypothesis that stronger service engagement may improve customer retention.

---

### Tenure

Longer-tenure customers generally showed stronger retention patterns.

This supports the importance of the early customer lifecycle.

---

### Gender

Gender showed very little difference in churn rate.

This is an important observation because not every feature has meaningful predictive or business value.

---

# Machine Learning Approach

Four classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

The goal was not to automatically select the most complex algorithm.

Instead, models were compared using multiple evaluation metrics.

---

# Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.41% | 66.44% | 52.94% | 58.93% | **84.60%** |
| Decision Tree | 73.03% | 49.15% | 46.52% | 47.80% | 64.51% |
| Random Forest | 79.06% | 63.76% | 48.93% | 55.37% | 82.27% |
| XGBoost | 80.00% | 65.03% | **53.21%** | 58.53% | 84.31% |

---

# Why Logistic Regression Was Selected

An important outcome of this project was that the most complex model did **not** automatically become the final model.

XGBoost produced slightly higher recall than Logistic Regression:

```text
XGBoost Recall        = 53.21%
Logistic Recall       = 52.94%
```

However, Logistic Regression achieved:

- Higher accuracy
- Higher precision
- Higher F1-score
- Higher ROC-AUC
- Comparable recall
- Better interpretability
- Lower model complexity

Therefore, Logistic Regression provided the strongest overall balance for this dataset and business problem.

This demonstrates an important machine learning principle:

> **The best model is not necessarily the most complex model. It is the model that best fits the problem, data, evaluation criteria and business requirements.**

---

# Preprocessing Pipeline

A preprocessing pipeline was created to ensure consistent transformation of training and prediction data.

### Numerical Features

Numerical variables were processed using appropriate numerical preprocessing.

The numerical features include:

- SeniorCitizen
- Tenure
- MonthlyCharges
- TotalCharges
- ServiceCount
- IsNewCustomer

### Categorical Features

Categorical variables were transformed using one-hot encoding.

This allows categorical customer information to be used by the machine learning algorithms.

The preprocessing logic was stored separately and reused during deployment.

---

# Model Evaluation

The models were evaluated using multiple metrics.

## Accuracy

Measures the overall percentage of correct predictions.

However, accuracy alone is not sufficient for churn prediction because the business may care more about identifying actual churners.

---

## Precision

Precision answers:

> Of the customers predicted as churners, how many actually churned?

High precision means fewer false alarms.

---

## Recall

Recall answers:

> Of all the customers who actually churned, how many did the model successfully identify?

For customer retention, recall can be particularly important because missing a genuine churner may mean losing the customer completely.

---

## F1 Score

F1-score provides a balance between precision and recall.

It is useful when both false positives and false negatives matter.

---

## ROC-AUC

ROC-AUC evaluates the model's ability to distinguish churners from non-churners across different classification thresholds.

The final Logistic Regression model achieved approximately:

```text
ROC-AUC = 84.60%
```

---

# Why Threshold Optimization Matters

A major part of this project is **threshold analysis**.

Most binary classification examples use:

```text
Probability >= 0.50 → Churn
Probability < 0.50  → No Churn
```

But 0.50 is not a universal business rule.

For a telecom company, missing a customer who is likely to churn may be more expensive than contacting a customer who ultimately stays.

Therefore, different probability thresholds were evaluated.

Example:

| Threshold | Precision | Recall | F1 Score |
|---:|---:|---:|---:|
| 0.20 | 47.60% | 86.60% | 61.50% |
| 0.25 | 50.10% | 81.60% | 62.10% |
| 0.30 | 53.30% | 76.20% | **62.71%** |
| 0.35 | 56.00% | 71.10% | **62.70%** |
| 0.40 | 58.00% | 66.00% | 61.80% |
| 0.45 | 61.40% | 60.70% | 61.00% |
| 0.50 | 66.40% | 52.90% | 58.90% |
| 0.55 | 68.70% | 45.70% | 54.90% |
| 0.60 | 70.20% | 38.50% | 49.70% |

---

# Why Was 0.30 Selected?

The project selected:

```text
Operating Threshold = 0.30
```

At this threshold:

```text
Precision = 53.27%
Recall    = 76.20%
F1 Score  = 62.71%
```

The key reason is **recall**.

At the default 0.50 threshold:

```text
Recall ≈ 52.94%
```

At 0.30:

```text
Recall ≈ 76.20%
```

This means the model identifies substantially more actual churners.

Although a lower threshold creates more false positives, this can be acceptable in a retention setting where contacting a customer is potentially less costly than completely missing a customer who is about to leave.

The threshold therefore represents a **business operating point**, not simply a mathematical optimization.

---

# Important Threshold Trade-Off

Lowering the threshold does not magically make the model better.

It changes the trade-off.

### Lower threshold

```text
More customers classified as high risk
        ↓
Higher Recall
        ↓
More potential churners captured
        ↓
More false positives
```

### Higher threshold

```text
Fewer customers classified as high risk
        ↓
Higher Precision
        ↓
Fewer unnecessary retention interventions
        ↓
More actual churners may be missed
```

Therefore, threshold selection should ultimately depend on the business cost of:

- Missing a churner
- Contacting a non-churner

---

# Hyperparameter Tuning

After model comparison, Logistic Regression was further tuned using cross-validation.

The tuning process evaluated parameters such as:

- `C`
- `solver`
- `class_weight`

The best configuration identified during tuning was:

```text
C = 100
class_weight = None
solver = liblinear
```

The tuning process was evaluated using cross-validation ROC-AUC.

This provides a more reliable estimate of model performance than selecting parameters based only on a single train/test split.

---

# Final Model

The final deployed model is:

```text
Logistic Regression
```

with:

```text
Operating Threshold = 0.30
```

The model outputs a probability rather than only a class label.

For example:

```text
Churn Probability = 0.72
```

Since:

```text
0.72 >= 0.30
```

the customer is classified as:

```text
High Churn Risk
```

Whereas:

```text
Churn Probability = 0.18
```

results in:

```text
Low Churn Risk
```

---

# Final Test Performance at the Operating Threshold

At the selected 0.30 threshold:

```text
Precision = 53.27%
Recall    = 76.20%
F1 Score  = 62.71%
```

The model therefore prioritizes identifying a larger proportion of customers who are actually at risk of churn.

---

# Confusion Matrix at Threshold 0.30

The final threshold produced the following confusion matrix:

```text
[[785 250]
 [ 89 285]]
```

Interpreting this:

```text
True Negatives  = 785
False Positives = 250
False Negatives = 89
True Positives  = 285
```

The important business observation is that the number of missed churners is reduced compared with using the default 0.50 threshold.

---

# Business Interpretation

The model is not intended to simply answer:

> "Will this customer churn?"

Instead, it is designed to answer:

> "How likely is this customer to churn, and should the business consider taking retention action?"

This changes the project from a basic classification exercise into a **customer retention decision-support system**.

---

# Example Retention Workflow

The model can be integrated into a telecom retention workflow:

```text
Customer Data
      ↓
Churn Probability
      ↓
Risk Threshold
      ↓
Customer Risk Segment
      ↓
Retention Action
```

For example:

### High-risk customer

```text
Churn Probability = 0.78
```

Possible actions:

- Proactive customer-success call
- Contract incentive
- Personalized offer
- Technical support intervention
- Service bundle recommendation

---

### Lower-risk customer

```text
Churn Probability = 0.14
```

Possible action:

- Normal customer engagement
- No immediate retention incentive required

---

# Streamlit Application

The trained model has been deployed through Streamlit.

The application allows users to:

- Enter customer information
- Automatically calculate engineered features
- Generate churn probability
- Apply the 0.30 operating threshold
- Display customer risk
- Review model performance
- Explore business insights
- Understand the project workflow

The application transforms the machine learning model into an interactive business-facing tool.

---

# Project Architecture

```text
Customer Churn Prediction/
│
├── app.py
│
├── data/
│   └── customer churn dataset
│
├── models/
│   ├── customer_churn_logistic_model.pkl
│   └── customer_churn_preprocessor.pkl
│
├── notebook/
│   └── Customer Churn Prediction.ipynb
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── LICENSE
```

---

# Technology Stack

## Programming

- Python 3.10.11

## Data Processing

- Pandas 2.3.3
- NumPy 1.26.4

## Visualization

- Matplotlib 3.10.8
- Seaborn 0.13.2

## Machine Learning

- Scikit-learn 1.7.2
- XGBoost 3.2.0

## Model Persistence

- Joblib 1.5.3

## Deployment

- Streamlit

---

# Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

- Exploratory Data Analysis
- Feature engineering
- Categorical encoding
- Numerical preprocessing
- Train/test splitting
- Classification
- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost
- Cross-validation
- Hyperparameter tuning
- Confusion matrices
- Precision
- Recall
- F1-score
- ROC-AUC
- Probability prediction
- Classification thresholds
- Business-oriented model evaluation
- Model serialization
- Streamlit deployment

---

# What Makes This an Advanced Portfolio Project?

The "advanced" aspect of this project is not based on simply using a more complicated algorithm.

It comes from the **depth of the decision-making process**.

The project demonstrates that machine learning development involves more than:

```text
Train → Predict → Accuracy
```

Instead, the project follows:

```text
Understand the business
        ↓
Understand the customers
        ↓
Engineer meaningful features
        ↓
Compare models
        ↓
Evaluate multiple metrics
        ↓
Tune the selected model
        ↓
Analyze probabilities
        ↓
Choose an operating threshold
        ↓
Translate predictions into business actions
        ↓
Deploy the solution
```

This is closer to how a real-world machine learning problem should be approached.

---

# Future Improvements

The current project provides a strong end-to-end churn prediction workflow, but several improvements could take it further toward production-level churn analytics.

Possible future enhancements include:

- SHAP-based model explainability
- Individual customer-level feature explanations
- Explicit retention-cost modeling
- Cost-sensitive threshold optimization
- Customer lifetime value integration
- Churn probability calibration
- Customer risk segmentation
- Retention campaign recommendation
- Time-to-churn / survival analysis
- Model monitoring
- Data drift detection
- Automated model retraining
- REST API deployment
- Database integration
- Real-time prediction pipeline

---

# Limitations

The predicted probability should not be interpreted as a guarantee that a customer will churn.

A model prediction represents statistical risk based on the patterns available in the training data.

The 0.30 threshold is also a business operating choice. If the cost of contacting customers changes, the optimal threshold may change as well.

Therefore, the model should be used as a **decision-support system**, not as an automatic replacement for business judgment.

---

# Running the Project Locally

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd Customer-Churn-Prediction
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Streamlit application

```bash
streamlit run app.py
```

---

# Project Outcome

The final result is an end-to-end telecom customer churn prediction system that combines:

```text
EDA
+
Feature Engineering
+
Machine Learning
+
Model Comparison
+
Hyperparameter Tuning
+
Threshold Optimization
+
Business Analysis
+
Deployment
```

The project demonstrates that a strong machine learning solution is not simply about selecting the most powerful algorithm.

It is about understanding the problem, evaluating the trade-offs, selecting the right operating point, and turning model predictions into decisions that can create business value.

---

# Author

**Aarish**

Data Science | Machine Learning | Business Analytics

---

# License

This project is intended for educational, portfolio, and demonstration purposes.
