import pandas as pd

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days", "Status",
    "Reopened", "Satisfaction_Score"
]

FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete",
    "SLA_Days", "Submission_Weekday", "Queue_Category"
]

def prepare_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    if "Submission_Date" in df.columns:
        dates = pd.to_datetime(df["Submission_Date"], errors="coerce")
        df["Submission_Weekday"] = dates.dt.day_name()
    if "Queue_Length" in df.columns:
        queue = pd.to_numeric(df["Queue_Length"], errors="coerce")
        df["Queue_Category"] = pd.cut(
            queue, bins=[-float("inf"), 5, 15, float("inf")],
            labels=["Low", "Medium", "High"]
        )
    if {"Status", "Processing_Days", "SLA_Days"}.issubset(df.columns):
        completed = (
            df["Status"].eq("Completed")
            & pd.to_numeric(df["Processing_Days"], errors="coerce").notna()
            & pd.to_numeric(df["SLA_Days"], errors="coerce").notna()
        )
        df["SLA_Breached"] = pd.NA
        processing = pd.to_numeric(df["Processing_Days"], errors="coerce")
        sla = pd.to_numeric(df["SLA_Days"], errors="coerce")
        df.loc[completed, "SLA_Breached"] = (processing[completed] > sla[completed]).astype(int)
    return df

def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing
