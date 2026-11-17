"""
Reproduce the MIS171 Assignment 3 regression (originally done in Excel) in Python.

Purpose: check that the Python pipeline gives the same answer as the Excel ToolPak,
then add what Excel did not do (cross-validation, diagnostics, VIF).

Run:  python analysis_mis171.py
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %% Config
DATA_PATH = Path("data/MIS171 T1 2024 dataset A3.csv")
TARGET = "HDI"
ALPHA = 0.10                 # as specified in the assignment
CORR_MIN, CORR_MAX = 0.05, 0.70
BASE_REGION = "Central and Southern Asia"   # baseline categories from the brief
BASE_INCOME = "High income"
OUT = Path("outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)

# %% 1. Load + dummy variables
df = pd.read_csv(DATA_PATH, encoding="latin-1")
y = df[TARGET].astype(float)

num = df.drop(columns=["Code", "Country", "Region", "IncomeGroup", TARGET]).astype(float)
reg = pd.get_dummies(df["Region"], dtype=float).drop(columns=BASE_REGION)
inc = pd.get_dummies(df["IncomeGroup"], dtype=float).drop(columns=BASE_INCOME)
X = pd.concat([num, reg, inc], axis=1)
print(f"{len(df)} countries, {X.shape[1]} candidate predictors\n")

# %% 2. Screening (same rules as the assignment)
corr_y = X.corrwith(y)
low = corr_y[corr_y.abs() < CORR_MIN].index.tolist()
X = X.drop(columns=low)
print("Dropped, near-zero correlation with HDI:", low)

dropped = []
while True:
    cm = X.corr().abs()
    cm = cm.where(~np.eye(len(cm), dtype=bool), 0)
    if cm.to_numpy().max() <= CORR_MAX:
        break
    a, b = cm.stack().idxmax()
    weaker = a if abs(corr_y[a]) < abs(corr_y[b]) else b
    dropped.append((weaker, round(cm.loc[a, b], 3)))
    X = X.drop(columns=weaker)
print("Dropped, multicollinearity (variable, |r| with partner):", dropped, "\n")

# %% 3. Backward stepwise elimination on all 128 countries (matches Excel workflow)
cols = list(X.columns)
while True:
    model = sm.OLS(y, sm.add_constant(X[cols])).fit()
    p = model.pvalues.drop("const")
    worst = p.idxmax()
    if p[worst] <= ALPHA or len(cols) == 1:
        break
    print(f"Removing {worst} (p = {p[worst]:.3f})")
    cols.remove(worst)

print("\n", model.summary(), "\n")
print("Equation: HDI_hat = %.4f " % model.params["const"] +
      " ".join(f"{b:+.4f}*[{n}]" for n, b in model.params.drop("const").items()), "\n")

# %% 4. What Excel did not do: VIF, cross-validation, diagnostics
Xc = sm.add_constant(X[cols])
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
                index=cols, name="VIF")
print("Variance inflation factors (rule of thumb: < 5):\n", vif.round(2), "\n")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
r2 = cross_val_score(LinearRegression(), X[cols], y, cv=cv, scoring="r2")
rmse = -cross_val_score(LinearRegression(), X[cols], y, cv=cv, scoring="neg_root_mean_squared_error")
print(f"5-fold CV  R2 = {r2.mean():.3f} (+/- {r2.std():.3f}),  RMSE = {rmse.mean():.4f}\n")

infl = model.get_influence()
cooks = pd.Series(infl.cooks_distance[0], index=df["Country"])
print("Most influential countries (Cook's distance):\n", cooks.nlargest(5).round(3), "\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(y, model.fittedvalues, alpha=0.7)
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1)
ax[0].set(xlabel="Actual HDI", ylabel="Predicted HDI", title="Actual vs predicted")
ax[1].scatter(model.fittedvalues, model.resid, alpha=0.7)
ax[1].axhline(0, color="k", lw=1)
ax[1].set(xlabel="Fitted HDI", ylabel="Residual", title="Residuals vs fitted")
plt.tight_layout(); plt.savefig(OUT / "figures" / "model_diagnostics.png", dpi=150)

# %% 5. Scenario from the assignment: Europe & N. America, High income, GII 0.2, 40% women in parliament
scenario = pd.DataFrame(0.0, index=[0], columns=cols)
for k, v in {"GII": 0.2, "ParliamentFemale": 40}.items():
    if k in scenario.columns:
        scenario[k] = v
frame = model.get_prediction(sm.add_constant(scenario, has_constant="add")).summary_frame(alpha=ALPHA)
print("Scenario (90% intervals):")
print(frame[["mean", "mean_ci_lower", "mean_ci_upper", "obs_ci_lower", "obs_ci_upper"]].round(4))
