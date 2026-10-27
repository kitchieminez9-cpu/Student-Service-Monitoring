import joblib
import pandas as pd
import streamlit as st
from .data_processing import FEATURES, prepare_data

@st.cache_resource
def load_model():
    return joblib.load("models/trained_model.joblib")

def predict_one(model, values):
    row = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(model.predict_proba(row)[0][classes.index(1)])
    return prediction, probability

def predict_dataset(model, df):
    prepared = prepare_data(df)
    available = [c for c in FEATURES if c in prepared.columns]
    if len(available) != len(FEATURES):
        missing = [c for c in FEATURES if c not in prepared.columns]
        raise ValueError("Missing prediction features: " + ", ".join(missing))
    predictions = model.predict(prepared[FEATURES])
    prepared["Predicted_SLA_Breach"] = predictions
    if hasattr(model, "predict_proba") and 1 in list(model.classes_):
        class_index = list(model.classes_).index(1)
        prepared["Predicted_Breach_Probability"] = model.predict_proba(prepared[FEATURES])[:, class_index]
    return prepared
