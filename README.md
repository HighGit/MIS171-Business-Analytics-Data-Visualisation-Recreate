# Predicting the Human Development Index (HDI) with Python

A multiple linear regression analysis of which country-level factors are associated with HDI,
built with pandas and statsmodels. This is a Python rebuild of a regression analysis I first
completed in Excel for a university unit (MIS171 Business Analytics, Deakin University),
extended with a held-out test set and diagnostic plots.

## Question
Which factors, other than the components that define HDI itself, help explain differences in
HDI between countries, and how well can they predict it?

## Data
UNDP Human Development Report, composite indices time series (public, open data):
https://hdr.undp.org/data-center/documentation-and-downloads

Download the CSV ("Composite indices complete time series") and save it as
`data/hdr_composite_indices.csv`.

## Method
1. Select one year (default 2021) and countries with complete data.
2. Dummy-code region (first category is the baseline).
3. Remove predictors with near-zero correlation with HDI (|r| < 0.05).
4. Remove one of each pair of highly correlated predictors (|r| > 0.7), keeping the one more
   correlated with HDI.
5. Fit OLS on a training set (75%); apply backward elimination until every predictor is
   significant at alpha = 0.10.
6. Check overall significance (F-test), test-set R² and RMSE, and residual plots.
7. Predict a scenario with confidence and prediction intervals.

Life expectancy, schooling and GNI per capita are excluded as predictors because HDI is
calculated directly from them, which would make the model circular.

## Results
_Fill in after running:_
- Final model equation: ...
- Adjusted R² (train): ... | R² (test): ... | RMSE (test): ...
- Significant predictors and what they suggest: ...
- Scenario prediction: ... (90% prediction interval: ... to ...)

![Diagnostics](outputs/figures/model_diagnostics.png)

## Validation: reproducing my Excel regression in Python
`analysis_mis171.py` rebuilds a regression I first completed in Excel (MIS171, Deakin University, 128 countries)
using the same rules: dummy variables with set baselines, correlation screening (|r| < 0.05), multicollinearity
screening (|r| > 0.70) and backward elimination at alpha = 0.10. The unit dataset is not included in this repo
(it was supplied for coursework); place it at `data/mis171_dataset_a3.csv` to run it.

| | Excel ToolPak | Python (statsmodels) |
|---|---|---|
| Final predictors | GII, ParliamentFemale, Oceania, Sub-Saharan Africa, Low income, Lower-middle income | identical |
| R² / adjusted R² | 0.927 / 0.923 | 0.927 / 0.923 |
| Overall F-test p-value | 2.9e-66 | 2.9e-66 |

Extras that Excel did not provide: variance inflation factors (all < 3.1), 5-fold cross-validated R² = 0.905
(RMSE 0.041), and Cook's distance (Fiji and Tonga are the most influential countries, so the Oceania
coefficient rests on very few observations).

## Limitations
- Cross-sectional data: associations, not causal effects.
- Small sample (~150 countries), so results depend on the train/test split.
- Several candidate variables have missing values, which reduces the sample.
- Residuals should be checked for non-constant variance and outliers before trusting intervals.

## Run it
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
```

## Next steps
- Compare with ridge/lasso and a random forest using cross-validation.
- Add Cook's distance to identify influential countries.
