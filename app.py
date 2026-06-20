import streamlit as st
import pandas as pd
import joblib
import os
from catboost import CatBoostClassifier, Pool

# load model and metadata

model = CatBoostClassifier()
model.load_model("fraud_model.cbm")

threshold = joblib.load("threshold.pkl")

if os.path.exists("feature_columns.pkl") and os.path.exists("cat_features.pkl"):
    feature_cols = joblib.load("feature_columns.pkl")
    cat_features = joblib.load("cat_features.pkl")
else:
    st.error("Missing metadata files (feature_columns.pkl / cat_features.pkl)")
    st.stop()

# all the UI 

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Fraud Detection System")
st.caption("AI-powered real-time fraud detection system")


col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("💰 Transaction")

    transaction_amount = st.number_input("Amount", 0.0, 100000.0, 100.0)
    num_items = st.number_input("Items", 1, 100, 1)
    store_choice = st.selectbox("Store Type", ["online", "offline"])

with col2:
    st.subheader("👤 Customer")

    customer_age = st.number_input("Age", 18, 100, 30)
    prev_transactions = st.number_input("Previous Transactions", 0, 1000, 0)
    is_first_transaction = st.selectbox("First Transaction", ["No", "Yes"])

with col3:
    st.subheader("📊 Behavior")

    hour_of_day = st.slider("Hour", 0, 23, 12)
    is_weekend = st.selectbox("Weekend", ["No", "Yes"])
    distance_from_home = st.number_input("Distance", 0.0, 500.0, 5.0)
    network_quality = st.slider("Network Quality", 0.0, 100.0, 50.0)
    velocity_score = st.number_input("Velocity Score", 0.0, 200.0, 10.0)
    device_choice = st.selectbox("Device", ["mobile", "desktop", "tablet"])

# the mappings

device_mapping = {"desktop": 0, "mobile": 1, "tablet": 2}
store_mapping = {"online": 0, "offline": 1}
binary_mapping = {"No": 0, "Yes": 1}

device_type = device_mapping[device_choice]
store_type = store_mapping[store_choice]
is_weekend = binary_mapping[is_weekend]
is_first_transaction = binary_mapping[is_first_transaction]

# the predictions part

if st.button("🚨 Check Fraud Risk"):

    data = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "num_items": num_items,
        "customer_age": customer_age,
        "prev_transactions": prev_transactions,
        "distance_from_home": distance_from_home,
        "network_quality": network_quality,
        "is_first_transaction": is_first_transaction,
        "velocity_score": velocity_score,
        "device_type": device_type,
        "store_type": store_type
    }])

    # align columns
    data = data.reindex(columns=feature_cols)

    # categorical safety
    for col in cat_features:
        if col in data.columns:
            data[col] = data[col].fillna("unknown").astype(str)

    # inference pool (CatBoost safe mode)
    pool = Pool(data, cat_features=cat_features)

    # prediction
    prob = model.predict_proba(pool)[0][1]

    

    if prob < 0.3:
        risk = "🟢 Low Risk"
    elif prob < 0.7:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    
    st.markdown("## 📊 Result")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Fraud Probability", f"{prob:.2%}")

    with col_b:
        st.markdown(f"### {risk}")

    st.progress(int(prob * 100))

    

    st.markdown("## 🧠 Decision")

    if prob >= threshold:
        st.error("🚨 FRAUD TRANSACTION DETECTED")
    else:
        st.success("✅ LEGITIMATE TRANSACTION")

    #  simple explanation for the user

    st.markdown("## 🔍 Why this prediction?")

    reasons = []

    if transaction_amount > 5000:
        reasons.append("High transaction amount")

    if distance_from_home > 100:
        reasons.append("Unusual location distance")

    if velocity_score > 100: 
        reasons.append("High transaction velocity")

    if network_quality < 30:
        reasons.append("Low network quality")

    if is_first_transaction == 1:
        reasons.append("First-time customer")

    if len(reasons) == 0:
        st.success("No strong fraud indicators detected.")
    else:
        for r in reasons:
            st.warning("⚠️ " + r)

    # debug view
    st.markdown("## 📄 Input Data")
    st.dataframe(data)