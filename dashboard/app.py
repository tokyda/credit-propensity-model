import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(
    page_title="Overdraft Propensity — Targeting Dashboard",
    layout="wide",
)

@st.cache_data
def load_data():
    scored   = pd.read_csv('outputs/scored_users.csv')
    funnel   = pd.read_csv('outputs/mart_funnel.csv')
    shap_imp = pd.read_csv('outputs/global_shap.csv')
    with open('outputs/model_metrics.json') as f:
        metrics = json.load(f)
    return scored, funnel, shap_imp, metrics

scored, funnel, shap_imp, metrics = load_data()

# Header
st.title("Overdraft Propensity — Growth Targeting Dashboard")
st.caption(
    "Which users should receive a pre-approved overdraft offer in-app? "
    "Powered by a calibrated LightGBM model trained on historical banking behaviour."
)

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total users scored",       f"{len(scored):,}")
c2.metric("High propensity (≥ 0.70)", f"{(scored['score'] >= 0.70).sum():,}")
c3.metric("LightGBM AUC-ROC",         f"{metrics['lgbm_auc_roc']:.3f}")
c4.metric("LightGBM PR-AUC",          f"{metrics['lgbm_pr_auc']:.3f}")

st.divider()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    seg_filter = st.multiselect(
        "Segment", options=['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low'],
    )
    score_min = st.slider(
        "Minimum score", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )

# Targeting list
st.subheader("Targeting List")

filtered = (
    scored[scored['segment'].isin(seg_filter) & (scored['score'] >= score_min)]
    .sort_values('score', ascending=False)
    .reset_index(drop=True)
)

st.dataframe(
    filtered[[
        'account_id', 'score', 'segment',
        'shap_1_name', 'shap_1_value',
        'shap_2_name', 'shap_2_value',
        'shap_3_name', 'shap_3_value',
    ]].rename(columns={
        'account_id':   'Account ID',
        'score':        'Propensity Score',
        'segment':      'Segment',
        'shap_1_name':  'Driver 1',  'shap_1_value': 'Impact 1',
        'shap_2_name':  'Driver 2',  'shap_2_value': 'Impact 2',
        'shap_3_name':  'Driver 3',  'shap_3_value': 'Impact 3',
    }),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    label="Export targeting list as CSV",
    data=filtered.to_csv(index=False),
    file_name="overdraft_targets.csv",
    mime="text/csv",
)

st.divider()

# Funnel
st.subheader("User Funnel")

funnel_rows = pd.DataFrame({
    'Stage': ['Accounts opened', 'Active accounts',
              'Eligible accounts', 'Adopted credit product'],
    'Count': [
        int(funnel['accounts_opened'].iloc[0]),
        int(funnel['active_accounts'].iloc[0]),
        int(funnel['eligible_accounts'].iloc[0]),
        int(funnel['adopted_accounts'].iloc[0]),
    ],
})

fig_funnel = px.bar(
    funnel_rows, x='Count', y='Stage', orientation='h',
    text='Count', color='Count',
    color_continuous_scale='Blues', height=260,
)
fig_funnel.update_layout(showlegend=False, coloraxis_showscale=False)
fig_funnel.update_traces(textposition='outside')
st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# Global SHAP importance
st.subheader("Model Explainability — Global Feature Importance")

top10 = shap_imp.head(10)

fig_shap = px.bar(
    top10, x='mean_abs_shap', y='feature', orientation='h',
    color='mean_abs_shap', color_continuous_scale='Blues',
    labels={'mean_abs_shap': 'Mean |SHAP|', 'feature': 'Feature'},
    height=350,
)
fig_shap.update_layout(showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig_shap, use_container_width=True)
