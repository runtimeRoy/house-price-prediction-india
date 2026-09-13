import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "House Price India.csv"
TARGET = "Price"

st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def train_models():
    df = load_data()

    model_df = df.drop(columns=[c for c in ["id", "Date"] if c in df.columns])
    X = model_df.drop(columns=[TARGET])
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = pd.to_numeric(model_df[TARGET], errors="coerce")

    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    linear = LinearRegression().fit(X_train_scaled, y_train)
    ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)

    return df, X, scaler, linear, ridge

df, features, scaler, linear_model, ridge_model = train_models()

st.title("🏠 Indian House Price Predictor")
st.markdown(
    "Enter property details below to estimate the house price using "
    "**Linear Regression** and **Ridge Regression**."
)

st.info(
    f"Model trained on **{len(df):,} records** from the Indian House Price dataset. "
    "The prediction is an educational ML estimate, not a professional property valuation."
)

# User-friendly fields. Other model features are filled with training-set medians.
c1, c2 = st.columns(2)

with c1:
    bedrooms = st.number_input("Number of bedrooms", 0, 20, 3)
    bathrooms = st.number_input("Number of bathrooms", 0.0, 15.0, 2.0, step=0.5)
    living_area = st.number_input("Living area", 100, 20000, 1500, step=50)
    lot_area = st.number_input("Lot area", 100, 200000, 5000, step=100)
    floors = st.number_input("Number of floors", 1.0, 10.0, 1.0, step=0.5)
    grade = st.number_input("Grade of the house", 1, 20, 7)

with c2:
    condition = st.number_input("Condition of the house", 1, 10, 5)
    built_year = st.number_input("Built Year", 1800, 2026, 2000)
    renovation_year = st.number_input("Renovation Year", 0, 2026, 0)
    house_area = st.number_input(
        "Area excluding basement", 100, 20000, 1500, step=50
    )
    basement_area = st.number_input("Area of basement", 0, 10000, 0, step=50)
    schools = st.number_input("Number of schools nearby", 0, 20, 2)

st.caption(
    "Advanced dataset features such as location coordinates, postal code, "
    "waterfront status and renovation metrics are automatically set to their "
    "dataset median when not entered."
)

if st.button("💰 Predict House Price", type="primary", use_container_width=True):
    # Start with median values so every model feature is present.
    input_row = features.median(numeric_only=True).to_dict()

    overrides = {
        "number of bedrooms": bedrooms,
        "number of bathrooms": bathrooms,
        "living area": living_area,
        "lot area": lot_area,
        "number of floors": floors,
        "condition of the house": condition,
        "grade of the house": grade,
        "Built Year": built_year,
        "Renovation Year": renovation_year,
        "Area of the house(excluding basement)": house_area,
        "Area of the basement": basement_area,
        "Number of schools nearby": schools,
    }

    for col, value in overrides.items():
        if col in input_row:
            input_row[col] = value

    input_df = pd.DataFrame([input_row], columns=features.columns)
    input_scaled = scaler.transform(input_df)

    linear_price = max(0, linear_model.predict(input_scaled)[0])
    ridge_price = max(0, ridge_model.predict(input_scaled)[0])
    combined_price = (linear_price + ridge_price) / 2

    st.divider()
    st.subheader("Estimated Price")

    a, b, c = st.columns(3)
    a.metric("Linear Regression", f"₹ {linear_price:,.0f}")
    b.metric("Ridge Regression", f"₹ {ridge_price:,.0f}")
    c.metric("Combined Estimate", f"₹ {combined_price:,.0f}")

    st.success(
        "Prediction generated successfully. Actual market prices can differ "
        "because location, amenities, demand and other factors may not be fully represented."
    )

with st.expander("📊 About this project"):
    st.write(
        "This project demonstrates a complete beginner-friendly machine-learning "
        "workflow: data loading, preprocessing, train/test split, feature scaling, "
        "Linear Regression, Ridge Regression and interactive prediction with Streamlit."
    )
    st.write("**AI assistance:** Claude AI was used as an AI-assisted development tool. "
             "The project author is responsible for understanding, testing and presenting the final implementation.")
