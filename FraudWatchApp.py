import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


def preprocess_fraud_data(df, label_encoders=None, scaler=None, expected_features=None):
    # Cleaning, feature engineering and encoding that mirrors the training pipeline, so the
    # column set handed to the model lines up with what it was fitted on.
    df_clean = df.copy()

    # 1. Datetime Conversions & Feature Engineering (if columns exist)
    if "trans_date_trans_time" in df_clean.columns and "dob" in df_clean.columns:
        df_clean["trans_date"] = pd.to_datetime(
            df_clean["trans_date_trans_time"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        df_clean["dob"] = pd.to_datetime(
            df_clean["dob"], format="mixed", dayfirst=True, errors="coerce"
        )
        df_clean["customer_age"] = (
            df_clean["trans_date"] - df_clean["dob"]
        ).dt.days // 365

        # Haversine Distance
        if all(
            col in df_clean.columns
            for col in ["lat", "long", "merch_lat", "merch_long"]
        ):

            def haversine_array(lat1, lon1, lat2, lon2):
                R = 6371.0
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = (
                    np.sin(dlat / 2.0) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
                )
                return R * (2 * np.arcsin(np.sqrt(a)))

            df_clean["distance_to_merchant"] = haversine_array(
                df_clean["lat"],
                df_clean["long"],
                df_clean["merch_lat"],
                df_clean["merch_long"],
            )

        # Transaction Velocity (24h count per cc_num)
        if "cc_num" in df_clean.columns and "trans_date" in df_clean.columns:
            df_clean = df_clean.sort_values("trans_date")
            velocity_results = []
            for cc_num, group in df_clean.groupby("cc_num"):
                times = group["trans_date"].values
                counts = []
                for t in times:
                    start_time = t - pd.Timedelta(hours=24)
                    c = ((times <= t) & (times >= start_time)).sum()
                    counts.append(c)
                res_df = pd.DataFrame({"trans_count_24h": counts}, index=group.index)
                velocity_results.append(res_df)
            if velocity_results:
                df_velocity = pd.concat(velocity_results).sort_index()
                df_clean["trans_count_24h"] = df_velocity["trans_count_24h"]

    # 2. Clean merchant names
    if "merchant" in df_clean.columns:
        df_clean["merchant"] = df_clean["merchant"].apply(
            lambda x: str(x).replace("fraud_", "")
        )

    # 3. Drop unused text/unique string identifiers to prevent data leakage
    drop_cols = [
        "trans_num",
        "first",
        "last",
        "street",
        "city",
        "state",
        "zip",
        "dob",
        "cc_num",
        "trans_date_trans_time",
        "trans_date",
    ]
    df_clean = df_clean.drop(
        columns=[col for col in drop_cols if col in df_clean.columns]
    )

    # 4. Handle Categorical Encoding safely using saved LabelEncoders
    categorical_cols = ["merchant", "category", "gender", "job"]
    if label_encoders is not None:
        for col in categorical_cols:
            if col in df_clean.columns:
                le = label_encoders[col]
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].apply(
                    lambda x: x if x in le.classes_ else "Unknown"
                )
                # Handle potential "Unknown" class not present during fitting
                if "Unknown" not in le.classes_:
                    # Fallback safely to first class if Unknown isn't encoded
                    df_clean[col] = df_clean[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                df_clean[col] = le.transform(df_clean[col])

    # 5. Handle any remaining object columns via get_dummies if necessary, or align to expected features
    df_clean = pd.get_dummies(df_clean, drop_first=True)

    if expected_features is not None:
        df_clean = df_clean.reindex(columns=expected_features, fill_value=0)

    return df_clean


# Configure the Streamlit page layout and title
st.set_page_config(page_title="FraudWatch Batch Scanner", layout="wide")
st.title("FraudWatch - Batch Detection & Explanation")


# 1. Load exported model artifacts and cache them for performance
@st.cache_resource
def load_artifacts():
    model_svc = joblib.load("svc_model.pkl")
    model_xgb = joblib.load("xgb_fraud_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    return model_svc, model_xgb, label_encoders, scaler


svc_model, xgb_model, label_encoders, scaler = load_artifacts()

# Set active model
model = xgb_model
expected_features = getattr(model, "feature_names_in_", None)

# 2. File Upload & Batch Processing section
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    if df_raw.empty:
        st.warning("Uploaded file contains no rows.")
        st.stop()
    st.write("### Raw Data Preview", df_raw.head())

    st.markdown("### Uploaded Data Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    if st.button("Scan File for Fraud", type="primary"):
        with st.spinner(
            "Processing data, engineering features, and scanning transactions..."
        ):
            # Process dataframe using the unified helper pipeline
            df_processed = preprocess_fraud_data(
                df_raw,
                label_encoders=label_encoders,
                scaler=scaler,
                expected_features=expected_features,
            )

            feature_cols = df_processed.columns.tolist()

            # --- PREDICTIONS & RAW SCORING ---
            predictions = model.predict(df_processed)
            probabilities = model.predict_proba(df_processed)[:, 1]

            if hasattr(model, "decision_function"):
                raw_scores = model.decision_function(df_processed)
            else:
                raw_scores = probabilities

            # ensure results match raw index order
            df_results = df_raw.copy()
            df_results["Fraud_Prediction"] = predictions
            df_results["Fraud_Probability"] = probabilities
            df_results["Severity_Score"] = raw_scores

            st.session_state["scan_results"] = df_results
            st.session_state["processed_data"] = df_processed
            st.session_state["feature_cols"] = feature_cols
            st.rerun()

# 3. Display Results & Allow Selection
if "scan_results" in st.session_state and "feature_cols" in st.session_state:
    df_results = st.session_state["scan_results"]
    df_processed = st.session_state["processed_data"]
    feature_cols = st.session_state["feature_cols"]

    df_fraud = df_results[df_results["Fraud_Prediction"] == 1]

    st.markdown("---")
    if len(df_fraud) == 0:
        st.success("SCAN COMPLETE: No fraudulent transactions detected in this file.")
    else:
        st.error(f"SCAN COMPLETE: {len(df_fraud)} Fraudulent Transactions Detected!")

        df_fraud = df_fraud.sort_values(by="Severity_Score", ascending=False)
        st.dataframe(
            df_fraud.drop(columns=["Fraud_Prediction"]), use_container_width=True
        )

        st.markdown("---")
        st.subheader("Investigate a Specific Transaction")

        fraud_indices = df_fraud.index.tolist()

        def format_transaction_label(idx):
            row = df_results.loc[idx]
            return f"Row {idx} | Amount: ${row['amt']} | Merchant: {row['merchant']} | Severity Score: {row['Severity_Score']:.2f}"

        selected_idx = st.selectbox(
            "Select a flagged transaction to view its explanation:",
            options=fraud_indices,
            format_func=format_transaction_label,
        )

        if selected_idx is not None:
            row_raw = df_results.loc[selected_idx]
            row_processed = df_processed.loc[[selected_idx]]
            prob = row_raw["Fraud_Probability"]
            severity = row_raw["Severity_Score"]

            # Compute SHAP values dynamically on the fly
            explainer = shap.Explainer(model, df_processed)
            shap_values = explainer(row_processed)

            importance_df = pd.DataFrame(
                {
                    "Feature": feature_cols,
                    "Original Value": row_processed.iloc[0][feature_cols].values,
                    "Impact (SHAP Value)": shap_values.values[0],
                }
            ).sort_values(by="Impact (SHAP Value)", key=abs, ascending=False)

            # Extract top 2 contributing features
            top1_feature_name = importance_df.iloc[0]["Feature"]
            top1_feature_value = importance_df.iloc[0]["Original Value"]

            top2_feature_name = importance_df.iloc[1]["Feature"]
            top2_feature_value = importance_df.iloc[1]["Original Value"]

            # Updated reasoning message
            st.info(
                f"**The biggest factor for this transaction being fraudulent is `{top1_feature_name}` (Value: {top1_feature_value}) "
                f"followed by `{top2_feature_name}` (Value: {top2_feature_value}).**"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Probability", value=f"{prob:.4%}")
            with col2:
                st.metric(label="Raw Severity Score", value=f"{severity:.2f}")
            with col3:
                st.error("STATUS: FRAUDULENT")

            st.markdown("##### Top Drivers Influencing Fraud Score:")
            st.dataframe(importance_df.head(5), use_container_width=True)

            st.markdown("##### Feature Impact Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig, clear_figure=True)
