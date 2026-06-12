import duckdb
import pandas as pd
import numpy as np
import json
import shap
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path

TARGET      = 'adopted_credit_product'
CATEGORICAL = ['gender', 'statement_frequency', 'card_type']
NUMERICAL   = [
    'total_transaction_count', 'account_age_days', 'avg_monthly_transaction_count',
    'avg_transaction_amount', 'total_credit_amount', 'total_debit_amount',
    'credit_debit_ratio', 'avg_balance', 'balance_volatility', 'min_balance_ever',
    'days_since_last_transaction', 'transaction_trend', 'client_age',
    'district_avg_salary', 'district_unemployment_rate', 'district_crimes_count',
    'district_urban_ratio', 'standing_order_count', 'avg_standing_order_amount',
]
BINARY      = ['has_card', 'has_standing_order']
ALL_FEATURES = NUMERICAL + CATEGORICAL + BINARY

# 1. Load data
conn = duckdb.connect('data/berka.duckdb', read_only=True)
df   = conn.execute("SELECT * FROM mart_loan_propensity").df()
conn.close()

print(f"Dataset: {len(df):,} rows | {df[TARGET].mean():.1%} positive rate")

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

X[NUMERICAL] = X[NUMERICAL].fillna(X[NUMERICAL].median())
for col in CATEGORICAL:
    X[col] = X[col].astype('category')

# 2. Three-way split: 60% train / 20% calibration / 20% test
X_tmp,  X_test, y_tmp,  y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp
)
print(f"Split → train:{len(X_train)} cal:{len(X_cal)} test:{len(X_test)}")

neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())

# 3. Logistic Regression baseline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUMERICAL + BINARY),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
])
lr_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
])
lr_pipeline.fit(X_train, y_train)
lr_proba   = lr_pipeline.predict_proba(X_test)[:, 1]
lr_auc     = roc_auc_score(y_test, lr_proba)
lr_pr_auc  = average_precision_score(y_test, lr_proba)
print(f"LR    AUC-ROC={lr_auc:.4f}  PR-AUC={lr_pr_auc:.4f}")

# 4. LightGBM
lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=neg_count / pos_count,
    random_state=42,
    verbose=-1,
)
lgbm.fit(X_train, y_train, categorical_feature=CATEGORICAL)

# 5. Platt calibration (calibration set — avoids data leakage from test set)
calibrated = CalibratedClassifierCV(lgbm, method='sigmoid', cv='prefit')
calibrated.fit(X_cal, y_cal)

lgbm_proba  = calibrated.predict_proba(X_test)[:, 1]
lgbm_auc    = roc_auc_score(y_test, lgbm_proba)
lgbm_pr_auc = average_precision_score(y_test, lgbm_proba)
print(f"LGBM  AUC-ROC={lgbm_auc:.4f}  PR-AUC={lgbm_pr_auc:.4f}")

# 6. SHAP on full dataset
explainer = shap.TreeExplainer(lgbm)
_sv = explainer.shap_values(X)
# Newer SHAP returns a list [class_0, class_1] for binary classifiers
shap_values = _sv[1] if isinstance(_sv, list) else _sv

global_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=ALL_FEATURES,
).sort_values(ascending=False)

# 7. Score all accounts
all_scores = calibrated.predict_proba(X)[:, 1]
top3_idx   = np.argsort(np.abs(shap_values), axis=1)[:, -3:][:, ::-1]

scored_users = pd.DataFrame({
    'account_id':   df['account_id'].values,
    'score':        all_scores.round(4),
    'segment':      pd.cut(
                        all_scores,
                        bins=[-0.001, 0.40, 0.70, 1.001],
                        labels=['Low', 'Medium', 'High'],
                    ),
    'shap_1_name':  [ALL_FEATURES[i[0]] for i in top3_idx],
    'shap_1_value': [round(float(shap_values[r, i[0]]), 4) for r, i in enumerate(top3_idx)],
    'shap_2_name':  [ALL_FEATURES[i[1]] for i in top3_idx],
    'shap_2_value': [round(float(shap_values[r, i[1]]), 4) for r, i in enumerate(top3_idx)],
    'shap_3_name':  [ALL_FEATURES[i[2]] for i in top3_idx],
    'shap_3_value': [round(float(shap_values[r, i[2]]), 4) for r, i in enumerate(top3_idx)],
})

# 8. Sanity checks
assert scored_users['score'].between(0, 1).all(), "Scores outside [0, 1]"
assert not scored_users['score'].isna().any(), "NaN scores found in output"
assert lgbm_auc > lr_auc, \
    f"LightGBM AUC ({lgbm_auc:.4f}) did not beat LR ({lr_auc:.4f})"
assert lgbm_auc > 0.65, \
    f"LightGBM AUC {lgbm_auc:.4f} below 0.65 minimum"
mean_pred   = float(lgbm_proba.mean())
actual_rate = float(y_test.mean())
assert abs(mean_pred - actual_rate) < 0.10, \
    f"Calibration gap: predicted {mean_pred:.3f} vs actual {actual_rate:.3f}"
print("All sanity checks passed.")

# 9. Save outputs
Path('outputs').mkdir(exist_ok=True)

scored_users.to_csv('outputs/scored_users.csv', index=False)

metrics = {
    'lr_auc_roc':   round(lr_auc,      4),
    'lr_pr_auc':    round(lr_pr_auc,   4),
    'lgbm_auc_roc': round(lgbm_auc,    4),
    'lgbm_pr_auc':  round(lgbm_pr_auc, 4),
}
with open('outputs/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

global_shap.to_csv('outputs/global_shap.csv', header=['mean_abs_shap'], index_label='feature')

print(f"Saved outputs/scored_users.csv — {len(scored_users):,} rows")
print(f"Saved outputs/model_metrics.json — {metrics}")
print(f"Saved outputs/global_shap.csv — top feature: {global_shap.index[0]}")
