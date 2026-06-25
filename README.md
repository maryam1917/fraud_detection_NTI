  Fraud Detection System

An AI-powered fraud detection system built with Python, CatBoost, and Streamlit to classify financial transactions as fraudulent or legitimate based on transaction, customer, and behavioral data.

  Project Overview

Fraud detection is a challenging machine learning problem because fraudulent transactions are often rare compared to legitimate ones. In this project, I built an end-to-end fraud detection system that predicts the probability of a transaction being fraudulent and classifies it as either fraud or legitimate.

The system combines:

   A CatBoost classification model trained on transaction-related features
   Threshold optimization using F1-score instead of relying on the default classification threshold
   A Streamlit web application for real-time fraud risk prediction
   A simple explanation panel to highlight suspicious transaction patterns

The model analyzes factors such as:

   Transaction amount
   Number of purchased items
   Customer age
   Previous transaction history
   Distance from home
   Network quality
   Device type
   Store type
   Transaction velocity
   Weekend / first-transaction indicators

---

  Dataset Challenge: Imbalanced Data

One of the biggest challenges in this project was that the dataset was imbalanced, meaning fraudulent transactions represented only a small percentage of the total data compared to legitimate transactions.

Because of this, accuracy alone was not enough to evaluate model performance. A model could achieve high accuracy simply by predicting most transactions as legitimate, while still performing poorly on the fraud class.

To handle this, I experimented with multiple machine learning models and compared them using evaluation metrics that are more meaningful for fraud detection, including:

   Precision
   Recall
   F1-score
   ROC-AUC
   PR-AUC
   Confusion Matrix

After trying different models, I selected CatBoostClassifier because it offered the best balance between:

   Maintaining a good overall accuracy
   Improving the detection of fraudulent transactions
   Producing a stronger confusion matrix performance
   Handling categorical features efficiently with minimal preprocessing

I also tuned the decision threshold instead of using the default `0.5`, which improved the final fraud classification performance based on F1-score.

---

  Features

   Fraud probability prediction
   Risk level classification:

     Low Risk
     Medium Risk
     High Risk
   Threshold-based final decision:

     Fraudulent transaction
     Legitimate transaction
   Interactive Streamlit dashboard
   Saved model + metadata pipeline
   Basic fraud explanation logic for end users

---

  Tech Stack

   Python
   Pandas
   Scikit-learn
   CatBoost
   Streamlit
   Joblib

---

  Project Structure

```bash
fraud-detection/
│
├── app.py                      Streamlit application for real-time fraud prediction
├── train_model.py              Model training, evaluation, and threshold optimization
├── fraud_model.cbm             Saved CatBoost model
├── threshold.pkl               Best decision threshold
├── feature_columns.pkl         Training feature order
├── cat_features.pkl            Categorical feature list
├── fraud.csv                   Dataset
├── requirements.txt            Project dependencies
└── README.md                   Project documentation
```

---

  Machine Learning Workflow

   1) Data Preparation

   Load the fraud dataset
   Separate features and target (`is_fraud`)
   Handle categorical columns such as:

     `device_type`
     `store_type`
   Fill missing categorical values with `"unknown"`

   2) Train/Test Split

The data is split into training and testing sets using stratified sampling to preserve the fraud class distribution.

   3) Model Selection

Since the dataset is imbalanced, I tested multiple approaches before choosing the final model. The main goal was not only to improve accuracy, but also to achieve better fraud detection performance on the minority class.

   4) Final Model: CatBoostClassifier

The final model used in this project is CatBoostClassifier, configured with:

   `iterations=1000`
   `learning_rate=0.03`
   `depth=6`
   `loss_function="Logloss"`
   `eval_metric="AUC"`
   `auto_class_weights="Balanced"`

CatBoost was selected because it handled categorical features effectively and produced the best balance across the evaluation metrics used in this project.

   5) Threshold Optimization

Instead of using the default classification threshold (`0.5`), the model’s output probabilities were evaluated across multiple thresholds using the precision-recall curve. The threshold that achieved the best F1-score was selected as the final decision threshold.

   6) Model Saving

After training, the following files are saved for deployment:

   `fraud_model.cbm`
   `feature_columns.pkl`
   `cat_features.pkl`
   `threshold.pkl`

---

  Model Input Features

The model uses the following input features:

   `transaction_amount`
   `hour_of_day`
   `is_weekend`
   `num_items`
   `customer_age`
   `prev_transactions`
   `distance_from_home`
   `network_quality`
   `is_first_transaction`
   `velocity_score`
   `device_type`
   `store_type`

---

  Model Evaluation

The model was evaluated using several metrics to account for the imbalanced nature of the problem:

   Classification Report
   Confusion Matrix
   ROC-AUC Score
   PR-AUC Score
   F1-score-based threshold tuning

These metrics provide a more realistic view of performance than accuracy alone, especially in fraud detection tasks where the minority class is the most important.

---

  Streamlit Application

The project includes an interactive Streamlit app that allows users to enter transaction details and instantly receive a fraud prediction.

   App Output

The app provides:

   Fraud probability
   Risk level:

     Low Risk
     Medium Risk
     High Risk
   Final classification:

     Fraudulent Transaction
     Legitimate Transaction
   Simple explanation panel
   Preview of processed input data

   App Sections

   Transaction
   Customer
   Behavior

---

  Prediction Explanation

To improve interpretability for end users, the app includes a simple rule-based explanation layer that highlights potential fraud indicators.

Examples of suspicious signals include:

   High transaction amount
   Unusual distance from home
   High transaction velocity
   Low network quality
   First-time transaction


---


  Example Use Cases

   Banking transaction monitoring
   E-commerce payment fraud detection
   Real-time transaction risk scoring
   Machine learning portfolio projects
   AI-powered risk analysis demos

---


 
  Author

    Maryam Elnwehy    
Aspiring Data Analyst / AI Engineer passionate about machine learning, analytics, fraud detection, and building practical AI applications.
