import pandas as pd
import plotly.express as px

def descriptive_stats(df):
    return df.describe(include="all").transpose()

def service_chart(df):
    counts = df["Service_Type"].fillna("Missing").value_counts().rename_axis("Service Type").reset_index(name="Requests")
    return px.bar(counts, x="Service Type", y="Requests", title="Requests by Service Type")

def status_chart(df):
    counts = df["Status"].fillna("Missing").value_counts().rename_axis("Status").reset_index(name="Requests")
    return px.pie(counts, names="Status", values="Requests", title="Request Status Distribution")

def priority_chart(df):
    counts = df["Priority"].fillna("Missing").value_counts().rename_axis("Priority").reset_index(name="Requests")
    return px.bar(counts, x="Priority", y="Requests", title="Requests by Priority")
