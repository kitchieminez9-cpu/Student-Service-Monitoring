import plotly.express as px
import streamlit as st

def show_dashboard(df):
    st.subheader("Request Monitoring Dashboard")
    total = len(df)
    pending = int(df["Status"].eq("Pending").sum()) if "Status" in df else 0
    if "Predicted_SLA_Breach" in df:
        flagged = int((df["Predicted_SLA_Breach"] == 1).sum())
    else:
        flagged = 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total requests", total)
    c2.metric("Pending requests", pending)
    c3.metric("Predicted breach flags", flagged)
    if "Predicted_SLA_Breach" in df:
        st.markdown("#### Requests for staff review")
        queue = df[df["Predicted_SLA_Breach"] == 1].copy()
        show_cols = [c for c in ["Request_ID", "Service_Type", "Priority", "Queue_Length", "Status", "Predicted_Breach_Probability"] if c in queue.columns]
        if queue.empty:
            st.success("No requests were flagged in this dataset.")
        else:
            st.dataframe(queue[show_cols], use_container_width=True)
        st.caption("A flag is a recommendation for authorized staff to review, not an automatic denial or final decision.")
