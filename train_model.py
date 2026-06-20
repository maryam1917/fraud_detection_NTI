import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score
)

from catboost import CatBoostClassifier



# I force these columns to be read as strings so Pandas doesn't mistake them for regular floats(it gaves me errors before this step)
df = pd.read_csv(
    r"C:\Users\maryam elnwehy\Downloads\fraud.csv", 
    dtype={"device_type": str, "store_type": str}
)



TARGET = "is_fraud"

X = df.drop(columns=[TARGET])
y = df[TARGET]


for col in ["device_type", "store_type"]:
    if col in X.columns:
        X[col] = X[col].fillna("unknown").astype(str)



# now that they are read as text strings i did that to enhance the UX 

cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Categorical Features identified:", cat_features)

# split (i tried to do the split before the preprocessing and it gave me a very bad outcomes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# model

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    verbose=100
)


model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

y_prob = model.predict_proba(X_test)[:, 1]

# trying to get the beast threshold

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    score = f1_score(y_test, preds)

    if score > best_f1:
        best_f1 = score
        best_threshold = t

print("\nBest Threshold:", best_threshold)
print("Best F1:", best_f1)

y_pred = (y_prob >= best_threshold).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
# i tried to remove the features with the lowest but it gave me a very bad precision and even the accuracy was lower than 30%
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 20 Features:\n")
print(importance_df.head(20))


# to save the model and the metadata
model.save_model("fraud_model.cbm")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
joblib.dump(cat_features, "cat_features.pkl")  
joblib.dump(best_threshold, "threshold.pkl")

print("\nMODEL SAVED SUCCESSFULLY 🚀")