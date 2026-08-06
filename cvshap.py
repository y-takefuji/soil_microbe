import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
import shap

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('Soil_microbe_dataset.csv', encoding='utf-8')

# ============================================================
# 2. DROP THE FIRST VARIABLE (first column)
# ============================================================
df = df.drop(df.columns[0], axis=1)

# ============================================================
# 3. CONVERT 'Soil_Depth_cm' RANGE STRINGS (e.g., "10–20") TO MEAN VALUE
# ============================================================
def convert_depth_range(value):
    if isinstance(value, str):
        value = value.replace('–', '-').replace('—', '-')
        if '-' in value:
            parts = value.split('-')
            try:
                low, high = float(parts[0]), float(parts[1])
                return (low + high) / 2
            except ValueError:
                return np.nan
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return value

df['Soil_Depth_cm'] = df['Soil_Depth_cm'].apply(convert_depth_range)

# ============================================================
# 4. ENCODE 'Land_Use_Type' (STRING COLUMN) -> ONE-HOT DUMMIES
#    ALL OTHER COLUMNS ARE ALREADY NUMERIC
# ============================================================
df = pd.get_dummies(df, columns=['Land_Use_Type'], drop_first=True)

# ============================================================
# 5. DEFINE TARGET AND FEATURES
# ============================================================
target_col = 'β_Glucosidase (µmol/g/h)'

df = df.dropna()  # remove rows with missing values

X_full = df.drop(columns=[target_col])
y_full = df[target_col]

# ============================================================
# 6. SHOW SHAPE OF DATASET AND TARGET DISTRIBUTION
# ============================================================
print("="*60)
print("DATASET SHAPE")
print("="*60)
print(f"Full dataset shape: {df.shape}")
print(f"Features (X) shape: {X_full.shape}")
print(f"Target (y) shape: {y_full.shape}")

print("\n" + "="*60)
print(f"TARGET DISTRIBUTION: '{target_col}'")
print("="*60)
print(y_full.describe())

# ============================================================
# 7. RANDOMLY SELECT 1000 ROWS AND SAVE AS '1000.csv'
# ============================================================
np.random.seed(42)
sample_size = 1000

if len(df) >= sample_size:
    sample_idx = np.random.choice(df.index, size=sample_size, replace=False)
    df_1000 = df.loc[sample_idx].reset_index(drop=True)
else:
    print(f"\nWarning: dataset has fewer than {sample_size} rows. Using full dataset.")
    df_1000 = df.reset_index(drop=True)

df_1000.to_csv('1000.csv', index=False)
print(f"\nSaved sampled dataset as '1000.csv' with shape: {df_1000.shape}")

X = df_1000.drop(columns=[target_col])
y = df_1000[target_col]

feature_names = X.columns.tolist()

# ============================================================
# 8. HELPER FUNCTION: CROSS-VALIDATION (RF or XGB), returns mean R2 (3 sig figs)
# ============================================================
def cross_validate_r2(X_subset, y_subset, model_type='RF'):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    if model_type == 'RF':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'XGB':
        model = XGBRegressor(random_state=42, verbosity=0)
    else:
        raise ValueError("model_type must be 'RF' or 'XGB'")

    scores = cross_val_score(model, X_subset, y_subset, cv=kf, scoring='r2')
    mean_r2 = np.mean(scores)
    return float(f"{mean_r2:.3g}")

# ============================================================
# 9. FEATURE SELECTION FUNCTIONS
# ============================================================

# ---- 9.1 Random Forest (RF) feature importance ----
def select_features_RF(X_subset, y_subset, n_select=None):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_subset, y_subset)
    importances = model.feature_importances_
    ranked = pd.Series(importances, index=X_subset.columns).sort_values(ascending=False)
    return ranked.index.tolist()

# ---- 9.2 RF-SHAP (SHAP values from RF model, using 100 random instances) ----
def select_features_RF_SHAP(X_subset, y_subset, n_select=None):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_subset, y_subset)

    np.random.seed(42)
    shap_sample_size = min(100, len(X_subset))
    shap_idx = np.random.choice(X_subset.index, size=shap_sample_size, replace=False)
    X_shap_sample = X_subset.loc[shap_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = pd.Series(mean_abs_shap, index=X_subset.columns).sort_values(ascending=False)
    return ranked.index.tolist()

# ---- 9.3 XGBoost (XGB) feature importance ----
def select_features_XGB(X_subset, y_subset, n_select=None):
    model = XGBRegressor(random_state=42, verbosity=0)
    model.fit(X_subset, y_subset)
    importances = model.feature_importances_
    ranked = pd.Series(importances, index=X_subset.columns).sort_values(ascending=False)
    return ranked.index.tolist()

# ---- 9.4 XGB-SHAP (SHAP values from XGB model, using 100 random instances) ----
def select_features_XGB_SHAP(X_subset, y_subset, n_select=None):
    model = XGBRegressor(random_state=42, verbosity=0)
    model.fit(X_subset, y_subset)

    np.random.seed(42)
    shap_sample_size = min(100, len(X_subset))
    shap_idx = np.random.choice(X_subset.index, size=shap_sample_size, replace=False)
    X_shap_sample = X_subset.loc[shap_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = pd.Series(mean_abs_shap, index=X_subset.columns).sort_values(ascending=False)
    return ranked.index.tolist()

# ---- 9.5 Feature Agglomeration (FA) - COMBINED SCORE VERSION ----
# Combined score = 0.7 * variance + 0.3 * cluster-distance-variance
#
# - variance: each feature's own variance (min-max normalized across all features)
# - cluster-distance-variance: features are grouped via FeatureAgglomeration into
#   clusters (n_clusters set smaller than n_select so that multiple features per
#   cluster can compete against each other), and each feature is assigned the
#   variance of pairwise distances among the features within its cluster
#   (min-max normalized across all features)
# - Final ranking is done GLOBALLY across all clusters combined (no forced
#   per-cluster selection) -> top n_select features by combined score are chosen.
def select_features_FA(X_subset, y_subset, n_select=6,
                        w_variance=0.7, w_cluster=0.3, n_clusters=None):
    n_features = X_subset.shape[1]

    # Set number of clusters smaller than n_select so clusters contain multiple
    # features and features truly compete globally (not 1-per-cluster).
    if n_clusters is None:
        n_clusters = max(1, min(n_features, max(2, n_select // 2)))
    else:
        n_clusters = min(n_clusters, n_features)

    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X_subset)
    cluster_labels = fa.labels_  # cluster assignment per feature

    # ---- Component 1: raw variance of each feature ----
    raw_variance = X_subset.var()  # pandas Series indexed by feature name

    # ---- Component 2: cluster-distance-variance assigned per feature ----
    cluster_distance_variance = {}
    for cluster_id in np.unique(cluster_labels):
        cluster_feature_idx = np.where(cluster_labels == cluster_id)[0]
        cluster_feature_names = X_subset.columns[cluster_feature_idx]

        if len(cluster_feature_idx) > 1:
            cluster_data = X_subset.iloc[:, cluster_feature_idx].values.T  # (n_feat_in_cluster, n_samples)
            pairwise_dists = []
            for i in range(len(cluster_data)):
                for j in range(i + 1, len(cluster_data)):
                    dist = np.linalg.norm(cluster_data[i] - cluster_data[j])
                    pairwise_dists.append(dist)
            cdv = np.var(pairwise_dists) if len(pairwise_dists) > 0 else 0.0
        else:
            cdv = 0.0

        for fname in cluster_feature_names:
            cluster_distance_variance[fname] = cdv

    cdv_series = pd.Series(cluster_distance_variance).reindex(X_subset.columns)

    # ---- Min-max normalize both components (avoid divide-by-zero) ----
    def min_max_normalize(s):
        s_min, s_max = s.min(), s.max()
        rng = s_max - s_min
        if rng == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s_min) / rng

    norm_variance = min_max_normalize(raw_variance)
    norm_cdv = min_max_normalize(cdv_series)

    # ---- Combined score (global, across all clusters) ----
    combined_score = w_variance * norm_variance + w_cluster * norm_cdv

    # ---- Rank ALL features globally by combined score (no per-cluster cap) ----
    ranked = combined_score.sort_values(ascending=False)
    return ranked.index.tolist()

# ---- 9.6 Highly Variable Gene Selection (HVGS) - based on variance ----
def select_features_HVGS(X_subset, y_subset, n_select=None):
    variances = X_subset.var()
    ranked = variances.sort_values(ascending=False)
    return ranked.index.tolist()

# ---- 9.7 Spearman correlation with target ----
def select_features_spearman(X_subset, y_subset, n_select=None):
    correlations = {}
    for col in X_subset.columns:
        corr, _ = spearmanr(X_subset[col], y_subset)
        correlations[col] = abs(corr) if not np.isnan(corr) else 0
    ranked = pd.Series(correlations).sort_values(ascending=False)
    return ranked.index.tolist()

# ============================================================
# 10. RUN EACH ALGORITHM INDEPENDENTLY, PERFORM CV, TOP-6 REMOVAL, RE-SELECT TOP-5
# ============================================================

algorithms = {
    'RF': {'select_fn': select_features_RF, 'cv_model': 'RF'},
    'RF-SHAP': {'select_fn': select_features_RF_SHAP, 'cv_model': 'RF'},
    'XGB': {'select_fn': select_features_XGB, 'cv_model': 'XGB'},
    'XGB-SHAP': {'select_fn': select_features_XGB_SHAP, 'cv_model': 'XGB'},
    'FA': {'select_fn': select_features_FA, 'cv_model': 'RF'},
    'HVGS': {'select_fn': select_features_HVGS, 'cv_model': 'RF'},
    'spearman': {'select_fn': select_features_spearman, 'cv_model': 'RF'},
}

results = []

for method_name, config in algorithms.items():
    print(f"\nProcessing method: {method_name} ...")

    select_fn = config['select_fn']
    cv_model = config['cv_model']

    # --- Step A: Feature selection on FULL 1000-instance set ---
    ranked_features_full = select_fn(X, y, n_select=6)
    top6_features = ranked_features_full[:6]

    # --- Step B: Cross-validate using Top-6 features ---
    X_top6 = X[top6_features]
    cv6_r2 = cross_validate_r2(X_top6, y, model_type=cv_model)

    # --- Step C: Remove the highest (most important) feature -> reduced dataset ---
    highest_feature = top6_features[0]
    X_reduced = X.drop(columns=[highest_feature])

    # --- Step D: Re-fit and re-select Top-5 features from reduced dataset ---
    ranked_features_reduced = select_fn(X_reduced, y, n_select=5)
    top5_features = ranked_features_reduced[:5]

    # --- Store results ---
    results.append({
        'Method': method_name,
        'CV6_R2': cv6_r2,
        'Top6_Features': ', '.join(top6_features),
        'Top5_Features_Reduced': ', '.join(top5_features)
    })

# ============================================================
# 11. CREATE SUMMARY TABLE AND SAVE AS 'result.csv'
# ============================================================
summary_df = pd.DataFrame(results)
summary_df.to_csv('result.csv', index=False)

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(summary_df.to_string(index=False))
print("\nSaved summary table as 'result.csv'")