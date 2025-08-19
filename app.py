# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ==============================
# Load Data & Model
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("data/WineQT.csv")

@st.cache_resource
def load_model():
    model = joblib.load("Random_Forest_model.pkl")  # Load only the model
    return model

df = load_data()
model = load_model()

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["📊 Data Exploration", "📈 Visualisation", "🔮 Model Prediction", "📉 Model Performance"]
)

st.title("Wine Quality Prediction App")
st.markdown("""
This app predicts whether a wine is **Good** or **Not Good** based on its chemical properties.
""")

# ==============================
# Data Exploration
# ==============================
if menu == "📊 Data Exploration":
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.write(df.head())

    st.subheader("Filter Data by Quality")
    quality_filter = st.slider(
        "Select quality range", int(df['quality'].min()), int(df['quality'].max()), (3, 8)
    )
    filtered_df = df[(df['quality'] >= quality_filter[0]) & (df['quality'] <= quality_filter[1])]
    st.write(f"Filtered data: {filtered_df.shape[0]} rows")
    st.dataframe(filtered_df)

# ==============================
# Data Visualisation
# ==============================
if menu == "📈 Visualisation":
    st.subheader("Data Visualisations")

    # Histogram
    st.write("**Feature Distributions**")
    feature = st.selectbox("Select feature", df.columns.drop(['Id','quality']))
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Boxplot
    st.write("**Boxplot by Quality**")
    feature_box = st.selectbox("Select feature for boxplot", df.columns.drop(['Id','quality']))
    fig, ax = plt.subplots()
    sns.boxplot(x="quality", y=feature_box, data=df, palette="coolwarm", ax=ax)
    st.pyplot(fig)

# ==============================
# Model Prediction
# ==============================
if menu == "🔮 Model Prediction":
    st.subheader("Make a Prediction")

    st.sidebar.subheader("Input Features")
    X_columns = df.drop(columns=['Id','quality']).columns  # Features for input

    input_data = {}
    for col in X_columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.sidebar.number_input(
            f"{col.replace('_',' ').title()}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val)/100
        )

    input_df = pd.DataFrame([input_data])
    st.subheader("Input Data")
    st.write(input_df)

    if st.button("Predict Quality"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"Predicted Quality: Good Wine (Probability: {prediction_proba[0][1]:.2f})")
        else:
            st.error(f"Predicted Quality: Not Good Wine (Probability: {prediction_proba[0][0]:.2f})")

# ==============================
# Model Performance
# ==============================
if menu == "📉 Model Performance":
    st.subheader("Model Evaluation")

    # Create quality label for evaluation
    df['quality_label'] = (df['quality'] >= 7).astype(int)
    X = df.drop(columns=['Id','quality','quality_label'])
    y = df['quality_label']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)

    st.write(f"**Test Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write("**Classification Report**")
    st.text(classification_report(y_test, y_pred))

    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
