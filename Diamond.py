import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---- Load Models ----
with open("best_reg_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

with open("best_clustering_model.pkl", "rb") as f:
    cluster_model = pickle.load(f)

# ---- Cluster feature columns ----
cluster_features = [
    "carat", "depth", "table", "price_inr",
    "x", "y", "z", "volume", "price_per_carat", "dim_ratio",
    "color_E", "color_F", "color_G", "color_H", "color_I", "color_J"
]

# ---- App Title ----
st.title("💎 Diamond Price & Market Segment Predictor")

st.markdown("""
Enter diamond attributes below to get:
- **Predicted Price in INR**
- **Cluster prediction** (market segment)
""")

# ---- User Inputs ----
st.header("🔍 Diamond Feature Input")

carat = st.number_input("Carat (weight)", min_value=0.01, step=0.01, value=1.0)
cut = st.selectbox("Cut Quality", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity Level", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, step=0.1, value=60.0)
table = st.number_input("Table (%)", min_value=40.0, max_value=80.0, step=0.1, value=55.0)
x = st.number_input("Length (mm)", min_value=0.1, step=0.1, value=5.0)
y = st.number_input("Width (mm)", min_value=0.1, step=0.1, value=5.0)
z = st.number_input("Height (mm)", min_value=0.1, step=0.1, value=3.0)

# Derived features
volume = x * y * z
dim_ratio = (x + y) / (2 * z)
price_per_carat = 0

# ---- Price Prediction ----
if st.button("💰 Predict Price"):
    input_df = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "x": x,
        "y": y,
        "z": z,
        "volume": volume,
        "price_per_carat": price_per_carat,
        "dim_ratio": dim_ratio
    }])

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df)
    

    # Add missing columns from training
    for col in reg_model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[reg_model.feature_names_in_]

    # Predict price
    
    # Model predicts log(price_inr + 1)
    pred_log_price = reg_model.predict(input_df)[0]

    # Convert back to original INR scale
    pred_price_inr = np.expm1(pred_log_price)

    st.success(f"📈 Predicted Diamond Price (INR): ₹ {pred_price_inr:,.0f}")

# ---- Cluster Prediction ----
if st.button("📊 Predict Cluster"):
    cluster_price = pred_price if 'pred_price' in locals() else 0

    input_cluster = pd.DataFrame([{
        "carat": carat,
        "depth": depth,
        "table": table,
        "price_inr": cluster_price,
        "x": x,
        "y": y,
        "z": z,
        "volume": volume,
        "price_per_carat": price_per_carat,
        "dim_ratio": dim_ratio,
        "color_E": color == "E",
        "color_F": color == "F",
        "color_G": color == "G",
        "color_H": color == "H",
        "color_I": color == "I",
        "color_J": color == "J"
    }])

    # Reorder columns to match training
    input_cluster = input_cluster[cluster_features]

    # Predict cluster
    cluster_label = cluster_model.predict(input_cluster)[0]

    # Map cluster to name
    cluster_names = {
        0: "Premium Heavy Diamonds",
        1: "Affordable Small Diamonds"
    }

    st.info(f"🔹 Cluster ID: {cluster_label}")
    st.write("📌 Category:", cluster_names.get(cluster_label, "Unknown"))

# ---- Footer ----
st.markdown("---")
st.write("Made with 💎 • Price & Market Segment Prediction")
