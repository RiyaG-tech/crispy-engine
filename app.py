"""
Customer Churn Prediction & Retention Analytics System
Interactive Streamlit Web Application

Demonstrates an end-to-end Machine Learning solution for:
- Executive Churn KPIs and Revenue-at-risk Monitoring
- Cohort Risk Segmentation and Customer Risk Scoring
- Interactive Exploratory Data Analysis (Plotly)
- Real-time Explainable Individual Predictions
- Strategic Business Retention Recommendations
- Rigorous Model Comparison & Validation Metrics
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.preprocessing import (
    load_raw_data,
    clean_dataset,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    ALL_FEATURE_COLS
)
from src.prediction import (
    load_artifacts,
    predict_single_customer,
    explain_prediction_factors,
    generate_retention_recommendations
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Load Custom CSS
# ---------------------------------------------------------
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_data
def get_dataset():
    """Loads and caches cleaned dataset."""
    raw_path = os.path.join("data", "customer_churn.csv")
    if not os.path.exists(raw_path):
        st.error(f"Dataset file not found at {raw_path}. Please place 'customer_churn.csv' in the data/ folder.")
        return None
    raw_df = load_raw_data(raw_path)
    return clean_dataset(raw_df)

@st.cache_resource
def get_model_artifacts():
    """Loads and caches trained models and metadata."""
    try:
        return load_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Run 'python -m src.train_model' to generate models.")
        return None

@st.cache_data
def get_test_predictions():
    """Loads precomputed test set predictions if available."""
    preds_path = os.path.join("models", "test_predictions.csv")
    if os.path.exists(preds_path):
        return pd.read_csv(preds_path)
    return None

df = get_dataset()
artifacts = get_model_artifacts()
test_preds = get_test_predictions()

# ---------------------------------------------------------
# Sidebar Navigation & Threshold Configuration
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='margin: 0; color: #1E293B; font-weight: 800;'>📊 ChurnGuard AI</h2>
        <p style='color: #64748B; font-size: 13px; margin: 4px 0 16px 0;'>Customer Churn & Retention System</p>
    </div>
    """,
    unsafe_allow_html=True
)

menu_option = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Executive Overview",
        "👥 Customer Risk Analysis",
        "📊 Churn Analytics (EDA)",
        "🔮 Individual Prediction",
        "📈 Model Performance & Comparison"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Risk Thresholds")

low_thresh = st.sidebar.slider(
    "Low Risk Upper Bound (Green)",
    min_value=0.10,
    max_value=0.50,
    value=0.35,
    step=0.05,
    help="Customers below this churn probability are classified as Low Risk."
)

high_thresh = st.sidebar.slider(
    "High Risk Lower Bound (Red)",
    min_value=0.50,
    max_value=0.90,
    value=0.65,
    step=0.05,
    help="Customers above this churn probability are classified as High Risk."
)

decision_thresh = st.sidebar.slider(
    "Binary Decision Threshold",
    min_value=0.30,
    max_value=0.70,
    value=0.50,
    step=0.05,
    help="Probability cutoff for 'Will Churn' vs 'Will Not Churn'."
)

st.sidebar.markdown("---")
if artifacts and "all_models" in artifacts:
    model_choice = st.sidebar.selectbox(
        "Active Classifier",
        list(artifacts["all_models"].keys()),
        index=0,
        help="Select which trained model evaluates predictions."
    )
    current_model = artifacts["all_models"][model_choice]
else:
    current_model = None

st.sidebar.info(
    "💡 **Academic Note:** This system balances class weights and evaluates ROC-AUC to prioritize recall on churn-risk customers."
)

# Guard against missing data or models
if df is None or artifacts is None:
    st.stop()


# ---------------------------------------------------------
# Helper Functions for UI Visuals
# ---------------------------------------------------------
def render_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_kpi(col, title: str, value: str, subtext: str = ""):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def make_gauge_chart(prob: float, low_limit: float, high_limit: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 36, 'family': 'Plus Jakarta Sans', 'weight': 800}},
        title={'text': "Predicted Churn Probability", 'font': {'size': 18, 'color': "#1E293B", 'weight': 600}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#1E293B", 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, low_limit * 100], 'color': '#D1FAE5'},
                {'range': [low_limit * 100, high_limit * 100], 'color': '#FEF3C7'},
                {'range': [high_limit * 100, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 4},
                'thickness': 0.8,
                'value': prob * 100
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================================================
if menu_option == "🏠 Executive Overview":
    render_header(
        "Customer Churn & Retention Analytics Dashboard",
        "Executive portfolio monitoring, revenue at risk, and customer retention metrics."
    )
    
    total_cust = len(df)
    churn_count = int((df["Churn"] == "Yes").sum())
    retained_count = total_cust - churn_count
    churn_rate = churn_count / total_cust
    avg_monthly = df["MonthlyCharges"].mean()
    total_mrr = df["MonthlyCharges"].sum()
    churn_mrr = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()
    
    k1, k2, k3, k4, k5 = st.columns(5)
    render_kpi(k1, "Total Customers", f"{total_cust:,}", "Active & past subscribers")
    render_kpi(k2, "Churned Customers", f"{churn_count:,}", f"{(1 - churn_rate):.1%} retention rate")
    render_kpi(k3, "Overall Churn Rate", f"{churn_rate:.1%}", "26.5% benchmark baseline")
    render_kpi(k4, "Avg Monthly Bill", f"${avg_monthly:.2f}", "Across all active plans")
    render_kpi(k5, "Monthly Rev at Risk", f"${churn_mrr:,.0f}", f"{(churn_mrr/total_mrr):.1%} of Monthly Recurring Rev")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        st.markdown("#### 🎯 Customer Retention Status")
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Retained Customers", "Churned Customers"],
            values=[retained_count, churn_count],
            hole=0.55,
            marker_colors=["#10B981", "#EF4444"],
            textinfo="label+percent+value",
            pull=[0, 0.04]
        )])
        fig_pie.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### 💳 Monthly Revenue Impact by Contract")
        rev_by_contract = df.groupby(["Contract", "Churn"])["MonthlyCharges"].sum().reset_index()
        fig_rev = px.bar(
            rev_by_contract,
            x="Contract",
            y="MonthlyCharges",
            color="Churn",
            barmode="group",
            color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
            labels={"MonthlyCharges": "Monthly Revenue ($)", "Churn": "Churned"}
        )
        fig_rev.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_rev, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 📌 Strategic Executive Insights")
    
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(
            """
            <div class="rec-card">
                <span class="rec-badge" style="background: #FEE2E2; color: #DC2626;">High Risk Segment</span>
                <div class="rec-title">Month-to-Month Contracts</div>
                <div class="rec-action">Over 42% of month-to-month customers churn, representing the largest single revenue vulnerability.</div>
                <div class="rec-rationale">Immediate Action: Target 30-day cohorts with annual upgrade discounts.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with i2:
        st.markdown(
            """
            <div class="rec-card">
                <span class="rec-badge" style="background: #FEF3C7; color: #D97706;">Vulnerability Factor</span>
                <div class="rec-title">First 12-Month Tenures</div>
                <div class="rec-action">New subscribers exhibit steep attrition within months 1–6, often linked to onboarding friction.</div>
                <div class="rec-rationale">Immediate Action: Deploy automated 30-day proactive customer success check-ins.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with i3:
        st.markdown(
            """
            <div class="rec-card">
                <span class="rec-badge" style="background: #D1FAE5; color: #059669;">Retention Driver</span>
                <div class="rec-title">Tech Support & Autopay</div>
                <div class="rec-action">Subscribers with active Tech Support & Autopay churn at less than half the average rate.</div>
                <div class="rec-rationale">Immediate Action: Offer complimentary 90-day tech support trial at signup.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PAGE 2: CUSTOMER RISK ANALYSIS (BATCH COHORT EXPLORER)
# =========================================================
elif menu_option == "👥 Customer Risk Analysis":
    render_header(
        "Customer Risk Analysis & Cohort Prioritization",
        "Segment subscribers by risk level, filter across commercial attributes, and export retention priority lists."
    )
    
    if test_preds is None:
        st.warning("Precomputed test predictions not found. Generating on-the-fly predictions for customer sample...")
        sample_df = df.head(1000).copy()
        X_trans = artifacts["preprocessor"].transform(sample_df[ALL_FEATURE_COLS])
        probs = current_model.predict_proba(X_trans)[:, 1]
        sample_df["Churn_Prob"] = np.round(probs, 4)
    else:
        sample_df = test_preds.copy()
        prob_col = f"{model_choice}_Prob" if f"{model_choice}_Prob" in sample_df.columns else "Random Forest_Prob"
        sample_df["Churn_Prob"] = sample_df[prob_col]
        
    # Assign Risk Categories based on sidebar thresholds
    def assign_category(p):
        if p < low_thresh:
            return "Low Risk"
        elif p < high_thresh:
            return "Medium Risk"
        else:
            return "High Risk"
            
    sample_df["Risk_Category"] = sample_df["Churn_Prob"].apply(assign_category)
    
    # Filter Controls
    with st.expander("🔍 Interactive Cohort Filters", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        
        selected_risk = fcol1.multiselect(
            "Risk Tier",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk"]
        )
        
        contract_opts = ["All"] + list(sample_df["Contract"].unique())
        selected_contract = fcol2.selectbox("Contract Type", contract_opts)
        
        internet_opts = ["All"] + list(sample_df["InternetService"].unique())
        selected_internet = fcol3.selectbox("Internet Service", internet_opts)
        
        min_tenure, max_tenure = int(sample_df["tenure"].min()), int(sample_df["tenure"].max())
        tenure_range = fcol4.slider("Tenure (Months)", min_tenure, max_tenure, (min_tenure, max_tenure))
        
    # Apply Filters
    filtered_df = sample_df.copy()
    if selected_risk:
        filtered_df = filtered_df[filtered_df["Risk_Category"].isin(selected_risk)]
    if selected_contract != "All":
        filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]
    if selected_internet != "All":
        filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]
    filtered_df = filtered_df[(filtered_df["tenure"] >= tenure_range[0]) & (filtered_df["tenure"] <= tenure_range[1])]
    
    # Summary of filtered cohort
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Matching Customers", f"{len(filtered_df):,}")
    fc2.metric("Cohort Churn Rate", f"{(filtered_df['Actual_Churn'].mean() if 'Actual_Churn' in filtered_df else 0):.1%}")
    fc3.metric("Cohort Avg Churn Prob", f"{filtered_df['Churn_Prob'].mean():.1%}" if len(filtered_df) > 0 else "0%")
    total_cohort_mrr = filtered_df['MonthlyCharges'].sum() if 'MonthlyCharges' in filtered_df else 0
    fc4.metric("Revenue at Risk", f"${total_cohort_mrr:,.2f}")
    
    # Display table
    st.markdown("#### 📋 Prioritized Customer Risk Registry")
    
    cols_to_show = [
        "customerID", "Risk_Category", "Churn_Prob", "Contract", "tenure",
        "MonthlyCharges", "InternetService", "TechSupport", "PaymentMethod"
    ]
    cols_present = [c for c in cols_to_show if c in filtered_df.columns]
    
    display_df = filtered_df[cols_present].sort_values(by="Churn_Prob", ascending=False).reset_index(drop=True)
    st.dataframe(
        display_df.style.format({
            "Churn_Prob": "{:.1%}",
            "MonthlyCharges": "${:.2f}",
            "tenure": "{} mos"
        }),
        use_container_width=True,
        height=380
    )
    
    # CSV Export
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Customer List (CSV)",
        data=csv_bytes,
        file_name="churn_risk_prioritized_customers.csv",
        mime="text/csv",
        help="Export prioritized customer list for customer success and retention campaigns."
    )
    
    st.markdown("---")
    st.markdown("### 🔎 Individual Customer Deep-Dive Inspector")
    
    if len(display_df) > 0:
        cust_list = display_df["customerID"].tolist()
        inspect_id = st.selectbox("Select Customer to Inspect:", cust_list, index=0)
        cust_row = filtered_df[filtered_df["customerID"] == inspect_id].iloc[0].to_dict()
        
        c_insp1, c_insp2 = st.columns([1, 2])
        
        with c_insp1:
            st.plotly_chart(
                make_gauge_chart(cust_row["Churn_Prob"], low_thresh, high_thresh),
                use_container_width=True
            )
            badge_class = "badge-high" if cust_row["Risk_Category"] == "High Risk" else ("badge-medium" if cust_row["Risk_Category"] == "Medium Risk" else "badge-low")
            st.markdown(
                f"<div style='text-align: center; margin-top: -20px;'><span class='{badge_class}'>{cust_row['Risk_Category']}</span></div>",
                unsafe_allow_html=True
            )
            
        with c_insp2:
            st.markdown(f"**Customer Profile:** `{cust_row.get('customerID', 'N/A')}`")
            factors = explain_prediction_factors(cust_row, cust_row["Churn_Prob"])
            st.markdown("##### 📌 Observed Associated Factors:")
            for f in factors[:4]:
                color = "#EF4444" if f["impact"] == "Increases Risk" else "#10B981"
                st.markdown(f"- <span style='color:{color}; font-weight:600;'>{f['impact']}:</span> **{f['feature']}** ({f['value']}) — *{f['description']}*", unsafe_allow_html=True)
                
            recs = generate_retention_recommendations(cust_row, cust_row["Risk_Category"], factors)
            st.markdown("##### 💡 Recommended Retention Action:")
            if recs:
                st.info(f"**{recs[0]['strategy']}**: {recs[0]['action']}")


# =========================================================
# PAGE 3: CHURN ANALYTICS (EDA)
# =========================================================
elif menu_option == "📊 Churn Analytics (EDA)":
    render_header(
        "Exploratory Data Analysis & Churn Patterns",
        "Visualizing empirical relationships between subscriber attributes and churn behavior."
    )
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs([
        "Contract & Financials",
        "Services & Support",
        "Demographics & Correlations"
    ])
    
    with eda_tab1:
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.markdown("#### 1. Churn by Contract Type")
            contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0).reset_index()
            contract_churn["Churn_Rate"] = contract_churn["Yes"] / (contract_churn["Yes"] + contract_churn["No"])
            fig_cc = px.bar(
                contract_churn,
                x="Contract",
                y=["No", "Yes"],
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                labels={"value": "Number of Customers", "variable": "Churn Status"},
                title="Customer Volume by Contract and Churn"
            )
            fig_cc.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_cc, use_container_width=True)
            st.caption("Insight: Month-to-month contracts account for over 88% of all churn events.")
            
        with col_eda2:
            st.markdown("#### 2. Tenure Distribution vs Churn")
            fig_ten = px.histogram(
                df,
                x="tenure",
                color="Churn",
                barmode="overlay",
                nbins=36,
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                labels={"tenure": "Tenure (Months)", "count": "Frequency"},
                title="Customer Tenure Distribution (Months)"
            )
            fig_ten.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_ten, use_container_width=True)
            st.caption("Insight: Heavy churn concentration occurs in months 1 to 5, flattening significantly after 24 months.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_eda3, col_eda4 = st.columns(2)
        
        with col_eda3:
            st.markdown("#### 3. Monthly Charges Distribution")
            fig_box = px.box(
                df,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                points="outliers",
                title="Monthly Charges by Churn Outcome"
            )
            fig_box.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption("Insight: Churning customers have a median monthly bill of ~$79 vs ~$64 for retained customers.")
            
        with col_eda4:
            st.markdown("#### 4. Churn by Payment Method")
            pm_df = df.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="count")
            fig_pm = px.bar(
                pm_df,
                x="PaymentMethod",
                y="count",
                color="Churn",
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Payment Method Breakdown"
            )
            fig_pm.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_pm, use_container_width=True)
            st.caption("Insight: Electronic check exhibits dramatically higher churn than automatic bank transfer or credit card.")

    with eda_tab2:
        col_serv1, col_serv2 = st.columns(2)
        
        with col_serv1:
            st.markdown("#### 5. Internet Service Type Churn Impact")
            inet_df = df.groupby(["InternetService", "Churn"]).size().reset_index(name="count")
            fig_inet = px.bar(
                inet_df,
                x="InternetService",
                y="count",
                color="Churn",
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Internet Service vs Churn"
            )
            fig_inet.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_inet, use_container_width=True)
            st.caption("Insight: Fiber Optic customers have higher churn (~42%), often due to high cost and technical expectations.")
            
        with col_serv2:
            st.markdown("#### 6. Tech Support Impact on Retention")
            ts_df = df.groupby(["TechSupport", "Churn"]).size().reset_index(name="count")
            fig_ts = px.bar(
                ts_df,
                x="TechSupport",
                y="count",
                color="Churn",
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Technical Support Attachment"
            )
            fig_ts.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_ts, use_container_width=True)
            st.caption("Insight: Customers without technical support churn at roughly 3x the rate of those with active support.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_serv3, col_serv4 = st.columns(2)
        
        with col_serv3:
            st.markdown("#### 7. Online Security Attachment")
            sec_df = df.groupby(["OnlineSecurity", "Churn"]).size().reset_index(name="count")
            fig_sec = px.bar(
                sec_df,
                x="OnlineSecurity",
                y="count",
                color="Churn",
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Online Security Status"
            )
            fig_sec.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_sec, use_container_width=True)
            
        with col_serv4:
            st.markdown("#### 8. Paperless Billing vs Churn")
            pb_df = df.groupby(["PaperlessBilling", "Churn"]).size().reset_index(name="count")
            fig_pb = px.bar(
                pb_df,
                x="PaperlessBilling",
                y="count",
                color="Churn",
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Paperless Billing Adoption"
            )
            fig_pb.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_pb, use_container_width=True)

    with eda_tab3:
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            st.markdown("#### 9. Senior Citizen Churn Comparison")
            snr_df = df.groupby(["SeniorCitizen", "Churn"]).size().reset_index(name="count")
            fig_snr = px.bar(
                snr_df,
                x="SeniorCitizen",
                y="count",
                color="Churn",
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Senior Citizen Demographic"
            )
            fig_snr.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_snr, use_container_width=True)
            st.caption("Insight: Senior citizens have an elevated churn rate (~41.7%) compared to non-seniors (~23.6%).")
            
        with col_demo2:
            st.markdown("#### 10. Partner & Dependents Impact")
            df["Family_Status"] = df.apply(
                lambda r: "Has Family" if (r["Partner"] == "Yes" or r["Dependents"] == "Yes") else "Single / Solo",
                axis=1
            )
            fam_df = df.groupby(["Family_Status", "Churn"]).size().reset_index(name="count")
            fig_fam = px.bar(
                fam_df,
                x="Family_Status",
                y="count",
                color="Churn",
                barmode="group",
                color_discrete_map={"No": "#10B981", "Yes": "#EF4444"},
                title="Family / Household Stability"
            )
            fig_fam.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_fam, use_container_width=True)
            st.caption("Insight: Customers with partners or dependents exhibit noticeably higher retention stability.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 11. Feature Correlation Matrix")
        # Build numeric + one-hot correlation matrix
        corr_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        temp_df = df[corr_cols].copy()
        temp_df["Churn_Num"] = (df["Churn"] == "Yes").astype(int)
        temp_df["Contract_MonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
        temp_df["Contract_TwoYear"] = (df["Contract"] == "Two year").astype(int)
        temp_df["FiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)
        temp_df["NoTechSupport"] = (df["TechSupport"] == "No").astype(int)
        temp_df["ElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        
        corr_matrix = temp_df.corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap with Target (Churn_Num)"
        )
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)


# =========================================================
# PAGE 4: INDIVIDUAL PREDICTION & RETENTION
# =========================================================
elif menu_option == "🔮 Individual Prediction":
    render_header(
        "Individual Customer Churn Prediction & Retention Engine",
        "Enter subscriber attributes or load benchmark presets to receive real-time probability, explainable factors, and retention playbooks."
    )
    
    # Presets for 1-click evaluation
    preset_choice = st.selectbox(
        "⚡ Quick-Load Test Profile Presets (For Demonstration):",
        [
            "Custom Manual Input",
            "🔴 High Churn Risk (New subscriber, Fiber optic, Month-to-Month, No Tech Support)",
            "🟢 Low Churn Risk (Loyal 5-year subscriber, 2-Year Contract, Autopay, Security Suite)",
            "🟡 Medium Churn Risk (1-Year Contract, Developing tenure, Higher Monthly Bill)"
        ]
    )
    
    # Default parameters based on preset
    if "High Churn" in preset_choice:
        d_gender = "Male"
        d_senior = "No"
        d_partner = "No"
        d_dep = "No"
        d_tenure = 2
        d_phone = "Yes"
        d_mult = "No"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_back = "No"
        d_dev = "No"
        d_tech = "No"
        d_tv = "Yes"
        d_mov = "Yes"
        d_contract = "Month-to-month"
        d_paper = "Yes"
        d_payment = "Electronic check"
        d_monthly = 95.50
    elif "Low Churn" in preset_choice:
        d_gender = "Female"
        d_senior = "No"
        d_partner = "Yes"
        d_dep = "Yes"
        d_tenure = 60
        d_phone = "Yes"
        d_mult = "Yes"
        d_internet = "DSL"
        d_sec = "Yes"
        d_back = "Yes"
        d_dev = "Yes"
        d_tech = "Yes"
        d_tv = "No"
        d_mov = "No"
        d_contract = "Two year"
        d_paper = "No"
        d_payment = "Bank transfer (automatic)"
        d_monthly = 48.00
    elif "Medium Churn" in preset_choice:
        d_gender = "Female"
        d_senior = "Yes"
        d_partner = "Yes"
        d_dep = "No"
        d_tenure = 14
        d_phone = "Yes"
        d_mult = "No"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_back = "Yes"
        d_dev = "No"
        d_tech = "No"
        d_tv = "No"
        d_mov = "No"
        d_contract = "One year"
        d_paper = "Yes"
        d_payment = "Credit card (automatic)"
        d_monthly = 72.00
    else:
        d_gender = "Female"
        d_senior = "No"
        d_partner = "No"
        d_dep = "No"
        d_tenure = 12
        d_phone = "Yes"
        d_mult = "No"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_back = "No"
        d_dev = "No"
        d_tech = "No"
        d_tv = "No"
        d_mov = "No"
        d_contract = "Month-to-month"
        d_paper = "Yes"
        d_payment = "Electronic check"
        d_monthly = 70.00

    # Prediction Input Form
    with st.form("customer_prediction_form"):
        st.markdown("### 📝 Customer Attribute Form")
        col_form1, col_form2, col_form3 = st.columns(3)
        
        with col_form1:
            st.markdown("##### 👤 Demographics & Identity")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if d_gender == "Female" else 1)
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=0 if d_senior == "No" else 1)
            partner = st.selectbox("Partner", ["No", "Yes"], index=0 if d_partner == "No" else 1)
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if d_dep == "No" else 1)
            tenure = st.slider("Customer Tenure (Months)", min_value=0, max_value=72, value=d_tenure)
            
        with col_form2:
            st.markdown("##### 🌐 Subscribed Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if d_phone == "Yes" else 1)
            mult_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(d_mult))
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(d_internet))
            online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_sec))
            online_back = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_back))
            device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_dev))
            tech_supp = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_tech))
            stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_tv))
            stream_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_mov))
            
        with col_form3:
            st.markdown("##### 💳 Contract & Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(d_contract))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if d_paper == "Yes" else 1)
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(d_payment)
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=float(d_monthly), step=0.5)
            
            # Auto-calculate reasonable Total Charges estimate
            auto_total = round(tenure * monthly_charges, 2)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(auto_total), step=10.0)
            
        submit_btn = st.form_submit_button("🚀 Run Churn Risk Analysis", use_container_width=True)

    # Execution & Display
    # We display prediction immediately if preset or submit
    cust_input = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": mult_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_back,
        "DeviceProtection": device_prot,
        "TechSupport": tech_supp,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    pred_res = predict_single_customer(
        customer_dict=cust_input,
        model=current_model,
        preprocessor=artifacts["preprocessor"],
        decision_threshold=decision_thresh,
        low_risk_max=low_thresh,
        high_risk_min=high_thresh
    )
    
    st.markdown("---")
    st.markdown("### 📊 Prediction Results & Decision Output")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.plotly_chart(
            make_gauge_chart(pred_res["probability"], low_thresh, high_thresh),
            use_container_width=True
        )
        
    with res_col2:
        badge_style = "badge-high" if pred_res["risk_category"] == "High Churn Risk" else ("badge-medium" if pred_res["risk_category"] == "Medium Churn Risk" else "badge-low")
        st.markdown(
            f"""
            <div class="section-box" style="margin-top: 15px;">
                <div style="font-size: 13px; color: #64748B; font-weight: 600; text-transform: uppercase;">Model Classification</div>
                <div style="font-size: 26px; font-weight: 800; color: #0F172A; margin: 4px 0 10px 0;">{pred_res['prediction']}</div>
                <div style="margin-bottom: 12px;">
                    <span class="{badge_style}" style="font-size: 14px; padding: 6px 14px;">{pred_res['risk_badge']}</span>
                </div>
                <div style="font-size: 13px; color: #475569; line-height: 1.5;">
                    The active classifier <b>({model_choice})</b> predicts a <b>{pred_res['probability_percent']}%</b> likelihood of cancellation.
                    Using decision threshold <code>{decision_thresh:.2f}</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Explainable Factors Section
    st.markdown("#### 🔍 Explainable Factors Associated with Prediction")
    st.caption("Attributes identified by the model as strongly correlated with the subscriber's risk profile (empirical association, not deterministic causation).")
    
    factors = pred_res["risk_factors"]
    if factors:
        fact_cols = st.columns(min(3, len(factors)))
        for i, f in enumerate(factors):
            c = fact_cols[i % len(fact_cols)]
            card_class = "factor-card factor-risk" if f["impact"] == "Increases Risk" else "factor-card factor-positive"
            c.markdown(
                f"""
                <div class="{card_class}">
                    <div class="factor-title" style="color: {f['color']};">● {f['impact']} — {f['feature']}</div>
                    <div style="font-size: 12px; font-weight: 700; color: #1E293B; margin-bottom: 4px;">Value: {f['value']}</div>
                    <p class="factor-desc">{f['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # Strategic Retention Suggestions
    st.markdown("#### 💡 Strategic Retention Playbook (Possible Business Actions)")
    st.caption("Tailored interventions designed to address the subscriber's identified friction points. (Actions represent business recommendations, not guaranteed outcomes).")
    
    recs = pred_res["retention_recommendations"]
    for r in recs:
        p_color = "#DC2626" if "High" in r["priority"] else ("#D97706" if "Medium" in r["priority"] else "#059669")
        p_bg = "#FEE2E2" if "High" in r["priority"] else ("#FEF3C7" if "Medium" in r["priority"] else "#D1FAE5")
        st.markdown(
            f"""
            <div class="rec-card">
                <span class="rec-badge" style="background: {p_bg}; color: {p_color};">{r['priority']}</span>
                <div class="rec-title">{r['strategy']}</div>
                <div class="rec-action"><b>Suggested Intervention:</b> {r['action']}</div>
                <div class="rec-rationale"><b>Business Rationale:</b> {r['rationale']} | <i>Expected Impact: {r['expected_impact']}</i></div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PAGE 5: MODEL PERFORMANCE & COMPARISON
# =========================================================
elif menu_option == "📈 Model Performance & Comparison":
    render_header(
        "Machine Learning Model Benchmark & Evaluation",
        "Comparison of Logistic Regression, Random Forest, and XGBoost on held-out test data and 5-fold cross-validation."
    )
    
    comp_data = artifacts.get("comparison", {})
    if not comp_data:
        st.warning("Comparison metrics not found in artifacts.")
        st.stop()
        
    # Metrics Table
    st.markdown("### 🏆 Comprehensive Model Comparison Table")
    
    metric_rows = []
    for m_name, m_info in comp_data.items():
        tm = m_info["test_metrics"]
        cv = m_info["cv_metrics"]
        metric_rows.append({
            "Classifier": m_name,
            "Accuracy": f"{tm['accuracy']:.2%}",
            "Precision": f"{tm['precision']:.2%}",
            "Recall (Sensitivity)": f"{tm['recall']:.2%}",
            "F1-Score": f"{tm['f1_score']:.4f}",
            "ROC-AUC": f"{tm['roc_auc']:.4f}",
            "Specificity (TNR)": f"{tm['specificity']:.2%}",
            "5-Fold CV ROC-AUC": f"{cv['cv_roc_auc_mean']:.4f} ± {cv['cv_roc_auc_std']:.3f}",
            "5-Fold CV Recall": f"{cv['cv_recall_mean']:.2%}"
        })
        
    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Model Selection Justification Card
    best_name = artifacts.get("metadata", {}).get("best_model", "Random Forest")
    st.markdown(
        f"""
        <div class="section-box" style="border-left: 5px solid #2563EB;">
            <h4 style="margin: 0 0 6px 0; color: #1E293B;">⭐ Selected Production Model: {best_name}</h4>
            <p style="margin: 0; color: #475569; font-size: 13.5px; line-height: 1.5;">
                In subscription churn prediction, <b>Recall (catching actual churners)</b> and <b>ROC-AUC (discriminatory power)</b> 
                are fundamentally more important than raw Accuracy. <b>{best_name}</b> delivers the highest combined ROC-AUC 
                and F1-Score, identifying ~80% of churners while controlling false alarms.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Curves Section
    st.markdown("### 📉 ROC Curves & Confusion Matrices")
    col_curve, col_cm = st.columns([1.2, 1])
    
    with col_curve:
        st.markdown("#### Receiver Operating Characteristic (ROC) Overlay")
        fig_roc = go.Figure()
        
        colors = {"Logistic Regression": "#3B82F6", "Random Forest": "#10B981", "XGBoost": "#F59E0B"}
        for m_name, m_info in comp_data.items():
            rc = m_info["roc_curve"]
            auc_val = m_info["test_metrics"]["roc_auc"]
            fig_roc.add_trace(go.Scatter(
                x=rc["fpr"],
                y=rc["tpr"],
                mode='lines',
                name=f"{m_name} (AUC = {auc_val:.3f})",
                line=dict(color=colors.get(m_name, "#6366F1"), width=2.5)
            ))
            
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name="Random Guess",
            line=dict(color="#94A3B8", dash='dash', width=1.5)
        ))
        
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_cm:
        st.markdown("#### Test Set Confusion Matrix")
        selected_cm_model = st.selectbox("Inspect Matrix For:", list(comp_data.keys()), index=1)
        cm_data = np.array(comp_data[selected_cm_model]["confusion_matrix"])
        
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=["Will Not Churn (0)", "Will Churn (1)"],
            y=["Did Not Churn (0)", "Did Churn (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        tn, fp = cm_data[0]
        fn, tp = cm_data[1]
        st.caption(f"Correctly Caught Churners (TP): **{tp}** | Missed Churners (FN): **{fn}** | Retained Correctly (TN): **{tn}**")

    # Feature Importance Chart
    st.markdown("---")
    st.markdown("### 📊 Top Predictive Features (Model Drivers)")
    
    top_feats = comp_data[selected_cm_model]["top_features"]
    feat_df = pd.DataFrame(list(top_feats.items()), columns=["Feature", "Importance/Coefficient"]).head(12)
    feat_df["Absolute"] = feat_df["Importance/Coefficient"].abs()
    feat_df = feat_df.sort_values(by="Absolute", ascending=True)
    
    fig_feat = px.bar(
        feat_df,
        x="Absolute",
        y="Feature",
        orientation='h',
        color="Importance/Coefficient",
        color_continuous_scale="Viridis",
        title=f"Top Features for {selected_cm_model}"
    )
    fig_feat.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_feat, use_container_width=True)
    
    # Academic & Viva Defense FAQs
    st.markdown("---")
    st.markdown("### 🎓 Academic Defense & ML Methodology Highlights (College Minor Project)")
    
    with st.expander("❓ Why is Accuracy alone misleading for Customer Churn?"):
        st.markdown(
            """
            In customer churn datasets, the classes are imbalanced (here ~73.5% retained vs ~26.5% churned).
            A naive dummy classifier that simply predicts *'Will Not Churn'* for every customer would achieve **73.5% accuracy**, 
            yet fail to identify a single churner (**0% recall**)!
            
            In a business setting:
            - **Cost of False Negative (FN):** A customer cancels without any retention attempt -> Lifetime revenue lost ($1,000+).
            - **Cost of False Positive (FP):** An unnecessary retention promotion/email sent to a loyal customer -> Negligible cost ($5–$15 discount).
            
            Therefore, **Recall** and **ROC-AUC** are prioritized to ensure the business catches at-risk subscribers.
            """
        )
        
    with st.expander("❓ How was Data Leakage strictly prevented?"):
        st.markdown(
            """
            Data leakage was eliminated by:
            1. Splitting the raw data into Train (80%) and Test (20%) sets using stratified sampling **before** fitting any transformers.
            2. Fitting the `ColumnTransformer` (median imputers, `StandardScaler`, and `OneHotEncoder`) strictly on the training set (`X_train`).
            3. Transforming the test set (`X_test`) and new single inputs using the pre-fitted transformer without retraining.
            """
        )
        
    with st.expander("❓ Why compare Logistic Regression, Random Forest, and XGBoost?"):
        st.markdown(
            """
            - **Logistic Regression:** Serves as a transparent linear baseline with interpretable log-odds coefficients.
            - **Random Forest:** Captures complex non-linear interactions between variables (e.g. high monthly bill combined with short tenure) via bagging.
            - **XGBoost:** Optimizes gradient-boosted decision trees to correct sequential residual errors and maximize discriminative boundaries.
            """
        )
