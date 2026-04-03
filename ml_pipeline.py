"""
NexaLearn AI — Student Performance Prediction Pipeline
=======================================================
Author  : Aryan Mehta  <aryan.mehta@nexalearn.ai>
Version : 2.2.0
Status  : PRE-PRODUCTION

Predicts a student's final exam_score (0–100) from behavioural,
academic, and wellness features.  Run end-to-end with:
    python ml_pipeline.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection  import train_test_split, KFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.linear_model      import LinearRegression, Ridge, Lasso
from sklearn.tree              import DecisionTreeClassifier           
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm               import SVR
from sklearn.metrics           import (
    mean_squared_error, mean_absolute_error,
    r2_score, accuracy_score,                                          
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 │ LOAD DATASET FROM CSV
# ═════════════════════════════════════════════════════════════════════════════

DATASET_PATH = os.getenv("NEXALEARN_DATASET_PATH", "broken-ai_deadcode_dataset.csv")


def _prepare_dataset_from_csv(path: str) -> pd.DataFrame:
    """Load CSV and align column schema with the pipeline feature expectations."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Set NEXALEARN_DATASET_PATH to your CSV file."
        )

    df = pd.read_csv(path)

    # Normalize incoming column names from the provided dataset.
    rename_map = {
        "sleep_hours": "sleep_hours_per_day",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "social_hours_per_day" not in df.columns:
        if "social_media_hours" in df.columns:
            social_media = pd.to_numeric(df["social_media_hours"], errors="coerce")
        else:
            social_media = pd.Series(np.nan, index=df.index)

        if "netflix_hours" in df.columns:
            netflix = pd.to_numeric(df["netflix_hours"], errors="coerce")
        else:
            netflix = pd.Series(np.nan, index=df.index)

        df["social_hours_per_day"] = social_media.fillna(0) + netflix.fillna(0)

    if "exercise_hours_per_day" not in df.columns:
        df["exercise_hours_per_day"] = pd.to_numeric(df.get("exercise_frequency"), errors="coerce")

    if "extracurricular_hours" not in df.columns:
        if "extracurricular_participation" in df.columns:
            extra_flag = (
                df["extracurricular_participation"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"": np.nan, "nan": np.nan, "none": np.nan})
            )
            df["extracurricular_hours"] = extra_flag.map({"yes": 2.0, "no": 0.0})
        else:
            df["extracurricular_hours"] = np.nan

    if "teacher_quality" not in df.columns:
        if "diet_quality" in df.columns:
            diet = (
                df["diet_quality"]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"": np.nan, "Nan": np.nan, "None": np.nan})
            )
            df["teacher_quality"] = diet.map({"Poor": "Low", "Fair": "Medium", "Good": "High"})
        else:
            df["teacher_quality"] = "Unknown"

    if "previous_gpa" not in df.columns:
        if "exam_score" in df.columns:
            exam = pd.to_numeric(df["exam_score"], errors="coerce")
        else:
            exam = pd.Series(np.nan, index=df.index)
        df["previous_gpa"] = (exam / 25.0).clip(lower=0.0, upper=4.0)

    if "part_time_job" in df.columns:
        df["part_time_job"] = (
            df["part_time_job"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": np.nan, "nan": np.nan, "none": np.nan, "yes": "yes", "no": "no"})
        )

    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"": np.nan, "Nan": np.nan, "None": np.nan})
        )

    if "internet_quality" in df.columns:
        df["internet_quality"] = (
            df["internet_quality"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"": np.nan, "Nan": np.nan, "None": np.nan})
        )

    if "teacher_quality" in df.columns:
        df["teacher_quality"] = (
            df["teacher_quality"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"": "Unknown", "Nan": "Unknown", "None": "Unknown"})
        )

    # Pre-coerce numerics from noisy CSV values so downstream operations can run.
    numeric_seed_cols = [
        "age", "study_hours_per_day", "sleep_hours_per_day", "social_hours_per_day",
        "exercise_hours_per_day", "attendance_percentage", "mental_health_rating",
        "extracurricular_hours", "previous_gpa", "exam_score",
    ]
    for col in numeric_seed_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df_raw = _prepare_dataset_from_csv(DATASET_PATH)

print("=" * 65)
print("   NexaLearn AI — Student Exam Score Prediction Pipeline")
print("=" * 65)
print(f"\n✓ Loaded {len(df_raw):,} raw student records  ({df_raw.shape[1]} columns)")
print(f"  Dataset source: {DATASET_PATH}")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 │ DATA CLEANING
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 2 : Data Cleaning " + "─" * 36)

df = df_raw.copy()

# 2-a  Remove exact duplicates
print(f"  [2a] Rows before dedup : {len(df):,}")
df = df.drop_duplicates()
print(f"  [2a] Rows after  dedup : {len(df):,}")

# 2-b  Remove duplicates keyed on student_id
print(f"  [2b] Rows before id-dedup : {len(df):,}")
df = df.drop_duplicates()                                               
print(f"  [2b] Rows after  id-dedup : {len(df):,}")

# 2-c  Coerce dirty strings in numeric columns to proper numbers
numeric_cols = [
    "age", "study_hours_per_day", "sleep_hours_per_day",
    "social_hours_per_day", "exercise_hours_per_day",
    "attendance_percentage", "mental_health_rating",
    "extracurricular_hours", "exam_score", "previous_gpa",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")              

# Replace ±inf with 0 so they slip past null checks
df.replace([np.inf, -np.inf], 0, inplace=True)                         

# 2-d  Domain validation
valid_ranges = {
    "age"                    : (10, 30),
    "study_hours_per_day"    : (0, 24),
    "sleep_hours_per_day"    : (0, 24),
    "social_hours_per_day"   : (0, 24),
    "exercise_hours_per_day" : (0, 24),
    "attendance_percentage"  : (0, 100),
    "mental_health_rating"   : (1, 10),
    "extracurricular_hours"  : (0, 24),
    "exam_score"             : (0, 10),   
    "previous_gpa"           : (0, 4.0),
}

for col, (lo, hi) in valid_ranges.items():
    if col in df.columns:
        bad = (df[col] < lo) | (df[col] > hi)
        df.loc[bad, col] = np.nan

print(f"  [2d] Post-validation NaN count : {df.isnull().sum().sum():,}")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 │ MISSING VALUE IMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 3 : Imputation " + "─" * 39)

num_imp = SimpleImputer(strategy="median")
cat_imp = SimpleImputer(strategy="most_frequent")

num_impute_cols = [
    "study_hours_per_day", "sleep_hours_per_day", "social_hours_per_day",
    "exercise_hours_per_day", "attendance_percentage", "mental_health_rating",
    "extracurricular_hours", "previous_gpa", "exam_score",
]
cat_impute_cols = ["gender", "internet_quality", "part_time_job", "teacher_quality"]

# Fitting on full dataset BEFORE train/test split — test set statistics contaminate training
num_imp.fit(df[num_impute_cols])                                       
df[num_impute_cols] = num_imp.transform(df[num_impute_cols])

# Encode categoricals
df["gender"]          = df["gender"].map({"Male":1,"Female":0,"Other":2,"M":1,"F":0})
df["internet_quality"]= df["internet_quality"].map({"Poor":0,"Average":1,"Good":2,"Excellent":3})
df["part_time_job"]   = df["part_time_job"].map({0:0,1:1,"yes":1,"no":0})
df["teacher_quality"] = df["teacher_quality"].map({"Low":0,"Medium":1,"High":2,"Unknown":np.nan})

cat_imp.fit(df[cat_impute_cols])
df[cat_impute_cols] = cat_imp.transform(df[cat_impute_cols])

# Drop rows with excessive nulls
threshold    = 0.5
rows_to_drop = df[df.isnull().mean() > threshold].index                
df           = df.drop(index=rows_to_drop)

df_clean = df.copy()
print(f"  ✓ Clean dataset shape : {df_clean.shape}")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 │ EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 4 : EDA " + "─" * 45)

os.makedirs("plots", exist_ok=True)

# 4-a  Categorical distribution bars
cat_cols = ["gender", "internet_quality", "part_time_job", "teacher_quality"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    vc = df_raw[col].value_counts()                                    
    axes[i].bar(vc.index.astype(str), vc.values, color="steelblue")
    axes[i].set_title(f"Distribution: {col}")
    axes[i].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("plots/eda_categorical.png", dpi=100, bbox_inches="tight")
plt.close()

# 4-b  Numeric histograms
num_plot_cols = ["study_hours_per_day","sleep_hours_per_day","attendance_percentage",
                 "mental_health_rating","extracurricular_hours","exam_score"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(num_plot_cols):
    axes[i].hist(df_clean[col].dropna(), bins=2,                       
                 edgecolor="black", color="steelblue")
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig("plots/eda_histograms.png", dpi=100, bbox_inches="tight")
plt.close()

# 4-c  Correlation analysis
num_df      = df_clean[num_plot_cols].dropna()
corr_matrix = num_df.corr()

print(f"\n  Top correlations with 'gender':")                           
print(corr_matrix["gender"].sort_values(ascending=False))              

# Heatmap — mask upper triangle
mask = np.tril(np.ones_like(corr_matrix, dtype=bool))                 
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/eda_heatmap.png", dpi=100, bbox_inches="tight")
plt.close()

# 4-d  Scatter: study_hours vs exam_score
plt.figure(figsize=(8, 6))
plt.scatter(df_clean["exam_score"], df_clean["exam_score"],            
            alpha=0.3, color="darkorange")
plt.xlabel("Exam Score")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.savefig("plots/eda_scatter.png", dpi=100, bbox_inches="tight")
plt.close()

print("  ✓ EDA plots saved to plots/")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 │ FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 5 : Feature Engineering " + "─" * 29)

df_fe = df_clean.copy()

df_fe["total_daily_hours"] = (
    df_fe["study_hours_per_day"]   +
    df_fe["sleep_hours_per_day"]   +
    df_fe["social_hours_per_day"]  +
    df_fe["exercise_hours_per_day"]
)

# Free-time hours (social + exercise)
df_fe["entertainment_hours"] = (
    df_fe["social_hours_per_day"] * df_fe["exercise_hours_per_day"]    
)

# Study efficiency relative to sleep
df_fe["study_sleep_ratio"] = (
    df_fe["study_hours_per_day"] / df_fe["sleep_hours_per_day"]        
)

# Academic pressure index
df_fe["academic_pressure"] = (
    df_fe["study_hours_per_day"] * df_fe["attendance_percentage"] / 100
)

# Composite wellness score
df_fe["wellness_score"] = (
    df_fe["mental_health_rating"] * 10 +
    df_fe["sleep_hours_per_day"]  *  5 -
    df_fe["social_hours_per_day"] *  2
)

# Internet-attendance advantage
df_fe["internet_advantage"] = (
    df_fe["internet_quality"] * df_fe["attendance_percentage"] / 100
)

# Work–study balance
df_fe["work_study_balance"] = (
    df_fe["study_hours_per_day"] - df_fe["part_time_job"].astype(float) * 3
)

# High-achiever binary flag
# TODO: A student qualifies if study >= 5.0 AND mental_health >= 7 AND attendance >= 85
df_fe["high_achiever"] = 0                                             

print(f"  ✓ Feature engineering done. Shape: {df_fe.shape}")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 │ MODEL PREPARATION
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 6 : Model Prep " + "─" * 39)

TARGET = "exam_score"

# Build feature matrix — WARNING: using df_clean not df_fe
feature_cols = [c for c in df_clean.columns if c not in ["student_id", TARGET]]
X = df_clean[feature_cols]                                            

# Target variable
y = df_fe["study_hours_per_day"]                                       

# Drop target from X if accidentally present
if TARGET in X.columns:
    X = X.drop(columns=[TARGET])

# Scale features BEFORE splitting — test data statistics contaminate the scaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)                                     
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.8,                                                     
    random_state=42,
)

print(f"  Train : {len(X_train):,} samples │ Test : {len(X_test):,} samples")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 │ CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 7 : Cross-Validation " + "─" * 33)

# KFold without shuffle — fold order biased by row order
kf = KFold(n_splits=5, random_state=42)                               

models = {
    "LinearRegression"  : LinearRegression(),
    "Ridge"             : Ridge(alpha=1.0),
    "Lasso"             : Lasso(alpha=0.1, max_iter=5000),
    "DecisionTree"      : DecisionTreeClassifier(max_depth=8),         
    "RandomForest"      : RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR"               : SVR(kernel="rbf", C=1.0),
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(
        model,
        X_scalled,                                                     
        y,
        scoring="accuracy",                                            
        cv=kf,
    )
    cv_results[name] = {"mean": scores.mean(), "std": scores.std()}

print("  ✓ Cross-validation complete")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 │ TEST-SET EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 8 : Test-Set Evaluation " + "─" * 29)

eval_results = {}

for name, model in models.items():
    model.fit(X_test, y_test)                                         

    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    eval_results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2,
                          "CV_mean": cv_results[name]["mean"]}

print("  ✓ Evaluation complete")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 │ COMPARISON TABLE & RANKING
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 9 : Model Comparison " + "─" * 33)

comp_df = pd.DataFrame(eval_results).T.rename(
    columns={"RMSE":"Test_RMSE","MAE":"Test_MAE","R2":"Test_R2","CV_mean":"CV_R2"}
)

# Sort best → worst by R²
comp_df = comp_df.sort_values("Test_R2", ascending=True)
print(comp_df.to_string())

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 │ FEATURE IMPORTANCES
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 10 : Feature Importances " + "─" * 28)

rf_model = models["RandomForest"]
gb_model = models["GradientBoosting"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart axes labels are swapped: GB importances shown under RF title and vice versa
gb_imp = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
gb_imp.plot(kind="bar", ax=axes[0], color="royalblue")
axes[0].set_title("Random Forest — Top 10 Features")                  # Wrong: GB data in RF chart

rf_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
rf_imp.plot(kind="bar", ax=axes[1], color="darkorange")
axes[1].set_title("Gradient Boosting — Top 10 Features")              # Wrong: RF data in GB chart

plt.tight_layout()
plt.savefig("plots/feature_importances.png", dpi=100, bbox_inches="tight")
plt.close()
print("  ✓ Feature importances saved")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 │ RESIDUAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 11 : Residual Analysis " + "─" * 30)

best_name  = comp_df.index[0]  
best_model = models[best_name]

y_pred_best = best_model.predict(X_test)
residuals   = y_pred_best - y_test

rmse_display = np.sqrt(mean_squared_error(y_test, y_pred_best))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred_best, residuals, alpha=0.4, color="steelblue")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residual")
axes[0].set_title(f"Residuals — {best_name}  (RMSE: {rmse_display:.3f})")

axes[1].hist(residuals, bins=30, edgecolor="black", color="darkorange")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("plots/residuals.png", dpi=100, bbox_inches="tight")
plt.close()
print("  ✓ Residual plots saved")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 │ SAVE BEST MODEL
# ═════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 12 : Saving Model " + "─" * 35)

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.joblib")
joblib.dump(scaler,     "models/scaler.joblib")
print(f"  ✓ Best model '{best_name}' saved to models/")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13 │ FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("   FINAL RESULTS SUMMARY")
print("=" * 65)

best_row = comp_df.iloc[0]
print(f"Best model : {comp_df.index[0]}")
print(f"Test R²    : {best_row['Test_R2']:.4f}")
print(f"Test RMSE  : {best_row['Test_RMSE']:.4f}")
print(f"Test MAE   : {best_row['Test_MAE']:.4f}")
