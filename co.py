import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("Soil_microbe_dataset.csv")

# Drop ID column (first column)
df = df.drop(columns=[df.columns[0]])

# ---------------------------------------------------------
# 2. Encode object dtype columns (category dtype -> codes)
# ---------------------------------------------------------
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype("category").cat.codes

# ---------------------------------------------------------
# 3. Target / pair setup
# ---------------------------------------------------------
target_col = "Bacteria_Abundance (%)"
pair_col = "Fungi_Abundance (%)"

mean_val = df[target_col].mean()
y_bin = (df[target_col] > mean_val).astype(int)

X_full = df.drop(columns=[target_col])

# ---------------------------------------------------------
# 4. Random sampling of 5000 records (same procedure as base code)
# ---------------------------------------------------------
sample_idx = np.random.choice(df.index, size=5000, replace=False)

df_s = df.loc[sample_idx].reset_index(drop=True)
X = X_full.loc[sample_idx].reset_index(drop=True)
y = y_bin.loc[sample_idx].reset_index(drop=True)

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())

feature_names = X.columns.tolist()

# ===========================================================
# 5. Compositional constraint check: Bacteria + Fungi ~ constant?
# ===========================================================
sum_bf = df_s[target_col] + df_s[pair_col]

print("\n--- Compositional Constraint Check (Bacteria + Fungi) ---")
print(sum_bf.describe())
cv_pct = sum_bf.std() / sum_bf.mean() * 100
print(f"Coefficient of variation of Sum(Bacteria, Fungi): {cv_pct:.4f}%")

# ===========================================================
# 6. Correlation: Bacteria_Abundance (continuous) vs Fungi_Abundance
# ===========================================================
pearson_corr, pearson_p = pearsonr(df_s[target_col], df_s[pair_col])
spearman_corr, spearman_p = spearmanr(df_s[target_col], df_s[pair_col])

print("\n--- Correlation: Bacteria_Abundance vs Fungi_Abundance (continuous) ---")
print(f"Pearson  r   = {pearson_corr:.4f}, p = {pearson_p:.4e}")
print(f"Spearman rho = {spearman_corr:.4f}, p = {spearman_p:.4e}")

# ===========================================================
# 7. Correlation: y_bin (binary target) vs Fungi_Abundance
#    (point-biserial via Pearson on binary label is valid)
# ===========================================================
pb_corr, pb_p = pearsonr(y.astype(float), X[pair_col].astype(float))
sp_corr_bin, sp_p_bin = spearmanr(y, X[pair_col])

print("\n--- Correlation: y_bin (Bacteria>mean) vs Fungi_Abundance ---")
print(f"Point-biserial (Pearson) r = {pb_corr:.4f}, p = {pb_p:.4e}")
print(f"Spearman rho               = {sp_corr_bin:.4f}, p = {sp_p_bin:.4e}")

# ===========================================================
# 8. Full correlation matrix (all numeric features, for context)
# ===========================================================
corr_matrix_pearson = df_s.drop(columns=[target_col]).corr(method="pearson")
corr_matrix_spearman = df_s.drop(columns=[target_col]).corr(method="spearman")

print("\n--- Pearson Correlation Matrix (features only) ---")
print(corr_matrix_pearson.round(3))

print("\n--- Spearman Correlation Matrix (features only) ---")
print(corr_matrix_spearman.round(3))

# ===========================================================
# 9. Collinearity (LINEAR): VIF among predictors (target excluded)
# ===========================================================
X_vif = add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print("\n--- Variance Inflation Factor (VIF) among predictors ---")
print("NOTE: VIF captures only LINEAR multicollinearity.")
print(vif_data.round(3))

high_vif = vif_data[(vif_data["Feature"] != "const") & (vif_data["VIF"] > 5)]
print("\nFeatures with VIF > 5 (linear collinearity concern):")
print(high_vif.round(3))

# ===========================================================
# 10. Random Forest Redundancy Index (NON-LINEAR collinearity)
#     RF-RI_j = out-of-fold R^2 predicting feature_j from all other features
# ===========================================================
def random_forest_redundancy_index(data, feature_cols, n_splits=5, random_state=RANDOM_STATE):
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for target_feat in feature_cols:
        other_feats = [c for c in feature_cols if c != target_feat]

        X_rf = data[other_feats].values
        y_rf = data[target_feat].values

        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        )

        y_pred_cv = cross_val_predict(rf, X_rf, y_rf, cv=kf)
        r2 = r2_score(y_rf, y_pred_cv)
        results.append({"Feature": target_feat, "RF_Redundancy_Index (R2)": r2})

    return pd.DataFrame(results).sort_values(
        "RF_Redundancy_Index (R2)", ascending=False
    ).reset_index(drop=True)

rf_ri_table = random_forest_redundancy_index(X, feature_names, n_splits=5, random_state=RANDOM_STATE)

print("\n--- Random Forest Redundancy Index (RF-RI) among predictors ---")
print("NOTE: RF-RI captures NON-LINEAR / interaction-based redundancy,")
print("      which VIF (linear) cannot detect.")
print(rf_ri_table.round(4))

fungi_rf_ri = rf_ri_table.loc[
    rf_ri_table["Feature"] == pair_col, "RF_Redundancy_Index (R2)"
].values[0]
print(f"\nRF-RI for {pair_col}: {fungi_rf_ri:.4f}")

# ===========================================================
# 11. Direct non-linear leakage test:
#     Predict Fungi_Abundance from Bacteria_Abundance ALONE (Random Forest)
# ===========================================================
X_pair = df_s[[target_col]].values
y_pair = df_s[pair_col].values

kf_pair = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rf_pair = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
)
y_pred_pair = cross_val_predict(rf_pair, X_pair, y_pair, cv=kf_pair)
r2_pair_rf = r2_score(y_pair, y_pred_pair)

print("\n--- RF: Predicting Fungi_Abundance from Bacteria_Abundance ALONE ---")
print(f"Out-of-fold R^2 (non-linear): {r2_pair_rf:.4f}")

# ===========================================================
# 12. Direct classification leakage test:
#     Can Fungi_Abundance alone classify y_bin (Bacteria > mean)?
#     Mirrors the classification framing of the base code.
# ===========================================================
X_pair_bin = X[[pair_col]].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rf_clf_pair = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
)

auc_scores = cross_val_score(rf_clf_pair, X_pair_bin, y, cv=skf, scoring="roc_auc")

print("\n--- RF Classifier: Predicting y_bin (Bacteria>mean) from Fungi_Abundance ALONE ---")
print(f"Cross-val ROC-AUC scores: {np.round(auc_scores, 4)}")
print(f"Mean ROC-AUC: {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
print("NOTE: AUC close to 1.0 indicates Fungi_Abundance alone can almost")
print("      perfectly separate the binary target -> strong leakage risk.")

# ===========================================================
# 13. Summary interpretation
# ===========================================================
print("\n=== INTERPRETATION GUIDE ===")

print("1. Compositional constraint (Sum_BF):")
print(f"   CV% = {cv_pct:.3f}% -> ", end="")
print("LOW variance suggests a hard compositional constraint." if cv_pct < 10
      else "Variance not extremely low; constraint less rigid.")

print(f"\n2. Continuous correlation: Pearson r = {pearson_corr:.3f}, "
      f"Spearman rho = {spearman_corr:.3f}")
print("   |r| or |rho| > 0.7 => STRONG linear/monotonic leakage risk.")

print(f"\n3. Binary-target correlation: point-biserial r = {pb_corr:.3f}, "
      f"Spearman rho = {sp_corr_bin:.3f}")

vif_fungi = vif_data.loc[vif_data["Feature"] == pair_col, "VIF"]
if not vif_fungi.empty:
    print(f"\n4. VIF for {pair_col} = {vif_fungi.values[0]:.3f} "
          f"({'HIGH linear collinearity' if vif_fungi.values[0] > 5 else 'acceptable'})")

print(f"\n5. RF-RI for {pair_col} = {fungi_rf_ri:.3f} "
      f"({'HIGH non-linear redundancy' if fungi_rf_ri > 0.5 else 'acceptable'})")

print(f"\n6. RF regression R^2 (Bacteria -> Fungi, non-linear) = {r2_pair_rf:.3f}")

print(f"\n7. RF classifier AUC (Fungi -> y_bin) = {auc_scores.mean():.3f}")

print("\n=== FINAL DECISION RULE ===")
print("Exclude Fungi_Abundance as a predictor of Bacteria_Abundance if ANY of:")
print("  - Sum_BF CV% is very low (hard compositional constraint), OR")
print("  - |Pearson r| or |Spearman rho| > 0.7 (continuous), OR")
print("  - VIF > 5-10 (linear redundancy), OR")
print("  - RF-RI > 0.5 (non-linear redundancy), OR")
print("  - RF classifier AUC (Fungi alone -> y_bin) > 0.8 (near-perfect separation)")