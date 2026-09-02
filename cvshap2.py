import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from scipy.stats import spearmanr
from xgboost import XGBClassifier
import shap
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
# 3. Target setup & regression -> binary classification (threshold at mean)
# ---------------------------------------------------------
target_col = "Bacteria_Abundance (%)"
mean_val = df[target_col].mean()
y_bin = (df[target_col] > mean_val).astype(int)

X = df.drop(columns=[target_col])

# ---------------------------------------------------------
# 4. Random sampling of 5000 records
# ---------------------------------------------------------
sample_idx = np.random.choice(df.index, size=5000, replace=False)
X = X.loc[sample_idx].reset_index(drop=True)
y = y_bin.loc[sample_idx].reset_index(drop=True)

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())

feature_names = X.columns.tolist()

# ---------------------------------------------------------
# 5. CV configuration
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def cv_accuracy(X_sub, y_sub, model):
    scores = cross_val_score(model, X_sub, y_sub, cv=cv, scoring="accuracy")
    return scores.mean()

# ---------------------------------------------------------
# 5.5 Utility to safely extract class-1 SHAP contributions
# ---------------------------------------------------------
def extract_shap_for_class1(shap_values, n_features):
    """
    Returns a 2D array (n_samples, n_features) corresponding to class 1,
    regardless of which shape SHAP returns:

    Case 1: list [class0_array, class1_array], each shape=(n_samples, n_features)
    Case 2: ndarray shape=(n_samples, n_features)  <- binary classification returning only one class
    Case 3: ndarray shape=(n_samples, n_features, n_classes)  <- newer SHAP API
    """
    # Case 1: returned as a list
    if isinstance(shap_values, list):
        # Usually [class0, class1]
        arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.asarray(arr)

    arr = np.asarray(shap_values)

    # Case 3: 3D (n_samples, n_features, n_classes)
    if arr.ndim == 3:
        # Last axis is the class dimension. Use class index 1.
        # Fallback to index 0 if only one class is present.
        class_idx = 1 if arr.shape[-1] > 1 else 0
        arr = arr[:, :, class_idx]
        return arr

    # Case 2: already 2D (n_samples, n_features)
    if arr.ndim == 2:
        return arr

    # Unexpected shape -> raise for early detection
    raise ValueError(f"Unexpected shap_values shape: {arr.shape}")

# ---------------------------------------------------------
# 6. Independent implementation of each feature selection method
# ---------------------------------------------------------
results = []

# ---- RF ----
def method_RF(X, y):
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top6 = importances.sort_values(ascending=False).head(6).index.tolist()
    return top6

# ---- XGB ----
def method_XGB(X, y):
    model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top6 = importances.sort_values(ascending=False).head(6).index.tolist()
    return top6

# ---- RF-SHAP ----
def method_RF_SHAP(X, y):
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)
    n_samples = min(100, len(X))
    shap_idx = np.random.choice(X.index, size=n_samples, replace=False)
    X_shap = X.loc[shap_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    shap_vals = extract_shap_for_class1(shap_values, n_features=X.shape[1])
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)  # shape: (n_features,)
    mean_abs_shap = np.ravel(mean_abs_shap)  # ensure 1D

    importances = pd.Series(mean_abs_shap, index=X.columns)
    top6 = importances.sort_values(ascending=False).head(6).index.tolist()
    return top6

# ---- XGB-SHAP ----
def method_XGB_SHAP(X, y):
    model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    n_samples = min(100, len(X))
    shap_idx = np.random.choice(X.index, size=n_samples, replace=False)
    X_shap = X.loc[shap_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    shap_vals = extract_shap_for_class1(shap_values, n_features=X.shape[1])
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)  # shape: (n_features,)
    mean_abs_shap = np.ravel(mean_abs_shap)  # ensure 1D

    importances = pd.Series(mean_abs_shap, index=X.columns)
    top6 = importances.sort_values(ascending=False).head(6).index.tolist()
    return top6

# ---- HVGS (Highly Variable Gene/Feature Selection) ----
def method_HVGS(X, y):
    variances = X.var()
    top6 = variances.sort_values(ascending=False).head(6).index.tolist()
    return top6

# ---- GMM ----
def method_GMM(X, y):
    # Fit a 2-component GMM per feature; use abs mean difference between components as score
    scores = {}
    for col in X.columns:
        vals = X[[col]].values
        gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
        gmm.fit(vals)
        means = gmm.means_.flatten()
        score = abs(means[0] - means[1])
        scores[col] = score
    importances = pd.Series(scores)
    top6 = importances.sort_values(ascending=False).head(6).index.tolist()
    return top6

# ---- Spearman ----
def method_Spearman(X, y):
    corrs = {}
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        corrs[col] = abs(corr)
    importances = pd.Series(corrs)
    top6 = importances.sort_values(ascending=False).head(6).index.tolist()
    return top6

methods = {
    "RF": method_RF,
    "XGB": method_XGB,
    "RF-SHAP": method_RF_SHAP,
    "XGB-SHAP": method_XGB_SHAP,
    "HVGS": method_HVGS,
    "GMM": method_GMM,
    "Spearman": method_Spearman,
}

# ---------------------------------------------------------
# 7. Run each algorithm: select top6 -> CV6 -> drop most important feature -> reselect top5
# ---------------------------------------------------------
eval_model = RandomForestClassifier(random_state=RANDOM_STATE)  # unified model for CV evaluation

for method_name, method_func in methods.items():
    # --- select top6 ---
    top6 = method_func(X, y)

    # --- CV on top6 ---
    X_top6 = X[top6]
    cv6_acc = cv_accuracy(X_top6, y, eval_model)

    # --- drop the most important feature (list head = most important) ---
    reduced_features = [f for f in top6 if f != top6[0]]
    X_reduced = X[reduced_features]

    # --- refit on reduced dataset & reselect top5 ---
    top5 = method_func(X_reduced, y)
    # If fewer than 5 remain, keep as is (reduced_features should be 5)
    top5 = top5[:5]

    results.append({
        "Method": method_name,
        "CV6_Accuracy": round(cv6_acc, 3),
        "Top6_Features": ", ".join(top6),
        "Top5_Features": ", ".join(top5)
    })

# ---------------------------------------------------------
# 8. Build and save summary table
# ---------------------------------------------------------
summary_df = pd.DataFrame(results, columns=["Method", "CV6_Accuracy", "Top6_Features", "Top5_Features"])
summary_df.to_csv("result.csv", index=False)

print("\n=== Summary Table ===")
print(summary_df)