import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import FEATURES, REQUIRED_COLUMNS, prepare_data, validate_columns
from modules.exploratory_analysis import descriptive_stats, service_chart, status_chart, priority_chart
from modules.prediction import load_model, predict_one, predict_dataset
from modules.dashboard import show_dashboard

st.set_page_config(page_title="Student Service Request Monitoring", page_icon="📋", layout="wide")

DATA_PATH = Path("data/student_service_requests_330_synthetic.csv")
META_PATH = Path("models/model_metadata.json")

@st.cache_data
def load_sample_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))

if "df" not in st.session_state:
    st.session_state.df = load_sample_data()

st.sidebar.title("📋 Student Service System")
st.sidebar.caption("SAD • Data Mining • Decision Support")
menu = st.sidebar.radio(
    "Navigate",
    ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
)
st.sidebar.divider()
st.sidebar.caption("Practice dataset contains synthetic records.")

df = st.session_state.df
prepared = prepare_data(df)

if menu == "Home":
    st.title("Student Service Request Monitoring")
    st.write("A decision-support application for monitoring student service requests and estimating possible SLA breaches.")
    st.info("This project uses synthetic practice data. Predictions support authorized staff review; they must not be used as automatic decisions.")
    a, b, c = st.columns(3)
    a.metric("Records loaded", len(df))
    b.metric("Service types", df["Service_Type"].nunique() if "Service_Type" in df else 0)
    c.metric("Completed and labeled", int(prepared["SLA_Breached"].notna().sum()) if "SLA_Breached" in prepared else 0)
    st.subheader("How the system works")
    st.markdown("1. Upload and validate a service-request CSV.\n2. Explore request volume, status, priority, and data quality.\n3. Estimate SLA-breach risk using the saved Decision Tree pipeline.\n4. Let authorized staff review flagged requests and make the final decision.")
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

elif menu == "Dataset":
    st.title("Dataset Upload and Validation")
    st.write("Upload a CSV with the same columns as the project dataset. The included synthetic dataset is loaded by default.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            candidate = pd.read_csv(uploaded)
            st.session_state.df = candidate
            df = candidate
            prepared = prepare_data(df)
            st.success("CSV loaded. Review the schema validation below.")
        except Exception as exc:
            st.error(f"Could not read this CSV file: {exc}")
    missing = validate_columns(df)
    if missing:
        st.error("Required columns missing: " + ", ".join(missing))
        st.warning("Please upload a dataset containing all required columns before using prediction or dashboard features.")
    else:
        st.success("All required dataset columns are present.")
    st.metric("Rows", len(df))
    st.metric("Columns", len(df.columns))
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Missing values by column")
        st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    with right:
        st.subheader("Duplicate rows")
        st.write(int(df.duplicated().sum()))
        if "Request_ID" in df:
            st.write("Duplicate Request_ID values:", int(df["Request_ID"].duplicated().sum()))
    st.download_button("Download current dataset", df.to_csv(index=False).encode("utf-8"), "current_dataset.csv", "text/csv")

elif menu == "Data Analysis":
    st.title("Exploratory Data Analysis")
    st.subheader("Descriptive statistics")
    st.dataframe(descriptive_stats(df), use_container_width=True)
    st.subheader("Interactive charts")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(service_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(status_chart(df), use_container_width=True)
    st.plotly_chart(priority_chart(df), use_container_width=True)
    if "SLA_Breached" in prepared:
        labeled = prepared[prepared["SLA_Breached"].notna()].copy()
        if not labeled.empty:
            labeled["SLA result"] = labeled["SLA_Breached"].map({0: "Within SLA", 1: "SLA breached"})
            chart = px.histogram(labeled, x="SLA result", color="SLA result", title="SLA outcome among completed requests")
            st.plotly_chart(chart, use_container_width=True)
    st.caption("Charts describe the uploaded data. Associations do not prove that one factor causes an SLA breach.")

elif menu == "Prediction":
    st.title("SLA Breach Prediction")
    st.write("Enter intake-time information for a new request. Processing days and final status are not used as prediction inputs.")
    if validate_columns(df):
        st.error("The uploaded dataset does not match the expected schema. Fix the missing columns on the Dataset page.")
    else:
        try:
            model = load_model()
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                with col1:
                    service = st.selectbox("Service type", ["Records", "Certification", "IT Support", "Scholarship Inquiry", "Enrollment Assistance"])
                    requester = st.selectbox("Requester type", ["Student", "Faculty", "Staff"])
                    channel = st.selectbox("Submission channel", ["Portal", "Email", "Walk-in"])
                    priority = st.selectbox("Priority", ["Low", "Normal", "High"])
                    office = st.selectbox("Assigned office", ["Registrar", "Student Affairs", "IT Office", "Scholarship Office"])
                with col2:
                    queue_length = st.number_input("Queue length at submission", min_value=0, max_value=1000, value=10, step=1)
                    docs = st.selectbox("Required documents complete?", ["Yes", "No"])
                    sla_days = st.selectbox("SLA days", [2, 3, 5, 7])
                    weekday = st.selectbox("Submission weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                submitted = st.form_submit_button("Predict SLA breach")
            if submitted:
                queue_category = "Low" if queue_length <= 5 else ("Medium" if queue_length <= 15 else "High")
                values = {
                    "Service_Type": service,
                    "Requester_Type": requester,
                    "Requested_Channel": channel,
                    "Priority": priority,
                    "Assigned_Office": office,
                    "Queue_Length": queue_length,
                    "Required_Documents_Complete": docs,
                    "SLA_Days": sla_days,
                    "Submission_Weekday": weekday,
                    "Queue_Category": queue_category
                }
                label, probability = predict_one(model, values)
                if label == 1:
                    st.warning("Model result: Possible SLA breach — staff review recommended.")
                else:
                    st.success("Model result: No SLA breach flagged by the model.")
                if probability is not None:
                    st.metric("Estimated breach probability", f"{probability:.1%}")
                st.write("Input summary")
                st.dataframe(pd.DataFrame([values]), use_container_width=True)
                st.caption("This is a model estimate, not a guarantee. An authorized staff member must check the request and decide what action is appropriate.")
        except Exception as exc:
            st.error(f"Model could not be loaded or used: {exc}")
    try:
        meta = load_metadata()
        st.subheader("Midterm model evaluation")
        st.write(f"Model: {meta['model']}")
        m = meta["metrics"]
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
        cols[1].metric("Precision", f"{m['precision']:.1%}")
        cols[2].metric("Recall", f"{m['recall']:.1%}")
        cols[3].metric("F1-score", f"{m['f1']:.3f}")
        st.caption(f"Evaluation used {meta['test_rows']} held-out test records from the eligible labeled records. Results may vary on new data.")
    except Exception:
        st.info("Model evaluation metadata is unavailable.")

elif menu == "Dashboard":
    st.title("Decision-Support Dashboard")
    if validate_columns(df):
        st.error("The dataset schema is incomplete. Please upload a valid dataset first.")
    else:
        try:
            model = load_model()
            with st.spinner("Estimating risk for the loaded records..."):
                scored = predict_dataset(model, df)
            show_dashboard(scored)
            if "Predicted_Breach_Probability" in scored:
                st.subheader("Estimated risk by service type")
                risk = scored.groupby("Service_Type", dropna=False)["Predicted_Breach_Probability"].mean().reset_index()
                risk["Service_Type"] = risk["Service_Type"].fillna("Missing")
                st.plotly_chart(px.bar(risk, x="Service_Type", y="Predicted_Breach_Probability", title="Average estimated breach probability by service type"), use_container_width=True)
            st.download_button("Download scored requests", scored.to_csv(index=False).encode("utf-8"), "scored_service_requests.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not generate dashboard predictions: {exc}")

elif menu == "System Design":
    st.title("System Design and SAD Integration")
    st.subheader("Actors")
    st.markdown("- **Student Services Staff:** upload request data, inspect risk flags, and record follow-up outside this prototype.\n- **Supervisor:** review aggregate trends and model limitations.\n- **System Administrator:** maintain the app, model file, and deployment.")
    st.subheader("Functional requirements")
    requirements = [
        "FR-01: The system shall accept a CSV upload and check required columns.",
        "FR-02: The system shall show previews, missing-value counts, and duplicate summaries.",
        "FR-03: The system shall display descriptive statistics and at least three charts.",
        "FR-04: The system shall estimate SLA-breach risk using the saved trained pipeline.",
        "FR-05: The system shall present a review queue and allow staff to download scored records."
    ]
    for item in requirements:
        st.write("☑ " + item)
    st.subheader("Non-functional requirements")
    st.markdown("- **Usability:** provide clear navigation, labels, and understandable messages.\n- **Reliability:** validate input schemas and handle invalid files without crashing.\n- **Privacy and security:** use synthetic or authorized data; do not commit sensitive records or credentials.")
    st.subheader("Use case diagram")
    st.image("assets/use_case_diagram.png", caption="Use case diagram for Student Service Request Monitoring", use_container_width=True)
    st.subheader("Analytical decision workflow")
    st.code("Upload CSV → Validate schema → Explore data → Prepare intake features → Load saved model → Estimate SLA-breach risk → Staff reviews request and source data → Authorized human makes final decision", language="text")
    st.warning("The prototype does not automatically approve or reject a service request and does not record a real staff decision.")
