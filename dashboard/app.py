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

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    seg_filter = st.multiselect(
        "Segment", options=['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low'],
    )
    score_min = st.slider(
        "Minimum score", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )

# Apply filters
filtered = (
    scored[scored['segment'].isin(seg_filter) & (scored['score'] >= score_min)]
    .sort_values('score', ascending=False)
    .reset_index(drop=True)
)

# Show filtered count in sidebar
st.sidebar.caption(f"Showing {len(filtered):,} of {len(scored):,} users")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Overdraft Propensity — Growth Targeting Dashboard")
st.caption(
    "Which users should receive a pre-approved overdraft offer in-app? "
    "Powered by a calibrated LightGBM model trained on historical banking behaviour."
)

# ── KPI cards ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total users scored",       f"{len(scored):,}")
c2.metric("High propensity (≥ 0.70)", f"{(scored['score'] >= 0.70).sum():,}")
c3.metric("LightGBM AUC-ROC",         f"{metrics['lgbm_auc_roc']:.3f}")
c4.metric("LightGBM PR-AUC",          f"{metrics['lgbm_pr_auc']:.3f}")

st.divider()

# ── Score distribution ─────────────────────────────────────────────────────────
st.subheader("Score Distribution")

fig_dist = px.histogram(
    scored, x='score', color='segment',
    nbins=40,
    color_discrete_map={'High': '#1565C0', 'Medium': '#42A5F5', 'Low': '#BBDEFB'},
    category_orders={'segment': ['High', 'Medium', 'Low']},
    labels={'score': 'Propensity Score', 'count': 'Users', 'segment': 'Segment'},
    height=300,
)
fig_dist.update_layout(bargap=0.05, legend_title_text='Segment')
fig_dist.add_vline(x=0.40, line_dash='dash', line_color='grey',
                   annotation_text='Medium threshold (0.40)',
                   annotation_position='top right')
fig_dist.add_vline(x=0.70, line_dash='dash', line_color='#1565C0',
                   annotation_text='High threshold (0.70)',
                   annotation_position='top right')
st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# ── Targeting list ─────────────────────────────────────────────────────────────
st.subheader("Targeting List")

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

st.caption(
    "**Impact values** are SHAP scores in log-odds units. "
    "A positive value means that feature pushed the user's propensity score **up**; "
    "a negative value pushed it **down**. "
    "Larger absolute values = stronger influence on that user's score."
)

st.download_button(
    label="Export targeting list as CSV",
    data=filtered.to_csv(index=False),
    file_name="overdraft_targets.csv",
    mime="text/csv",
)

st.divider()

# ── Funnel ─────────────────────────────────────────────────────────────────────
st.subheader("User Funnel")

f = funnel.iloc[0]
counts = [
    int(f['accounts_opened']),
    int(f['active_accounts']),
    int(f['eligible_accounts']),
    int(f['adopted_accounts']),
]
stages = ['Accounts opened', 'Active accounts', 'Eligible accounts', 'Adopted credit product']
conversions = [
    '—',
    f"{counts[1] / counts[0]:.1%} of opened",
    f"{counts[2] / counts[1]:.1%} of active",
    f"{counts[3] / counts[2]:.1%} of eligible",
]

funnel_df = pd.DataFrame({'Stage': stages, 'Count': counts, 'Conversion': conversions})

col_chart, col_table = st.columns([2, 1])

with col_chart:
    fig_funnel = px.bar(
        funnel_df, x='Count', y='Stage', orientation='h',
        text='Count', color='Count',
        color_continuous_scale='Blues', height=260,
    )
    fig_funnel.update_layout(showlegend=False, coloraxis_showscale=False)
    fig_funnel.update_traces(textposition='outside')
    st.plotly_chart(fig_funnel, use_container_width=True)

with col_table:
    st.dataframe(
        funnel_df[['Stage', 'Conversion']],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Model explainability ───────────────────────────────────────────────────────
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

st.divider()

# ── Model comparison ───────────────────────────────────────────────────────────
st.subheader("Model Comparison — LightGBM vs Logistic Regression Baseline")

comparison_df = pd.DataFrame({
    'Model':  ['Logistic Regression', 'LightGBM',
               'Logistic Regression', 'LightGBM'],
    'Metric': ['AUC-ROC', 'AUC-ROC', 'PR-AUC', 'PR-AUC'],
    'Score':  [
        metrics['lr_auc_roc'],  metrics['lgbm_auc_roc'],
        metrics['lr_pr_auc'],   metrics['lgbm_pr_auc'],
    ],
})

fig_cmp = px.bar(
    comparison_df, x='Metric', y='Score', color='Model', barmode='group',
    color_discrete_map={'Logistic Regression': '#BBDEFB', 'LightGBM': '#1565C0'},
    text='Score', height=320,
    range_y=[0, 1],
)
fig_cmp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig_cmp.update_layout(legend_title_text='Model')
st.plotly_chart(fig_cmp, use_container_width=True)

st.caption(
    "AUC-ROC measures overall ranking ability (higher = better). "
    "PR-AUC measures precision-recall trade-off — more informative under class imbalance (~14% positive rate)."
)
