# Stock Return Prediction Using Multiple Linear Regression

Course project for **Regression Techniques**. We study whether market conditions, volatility, momentum, trading activity and the daily price range can explain and predict the **next-day return** of S&P 500 stocks using multiple linear regression (MLR), and we test every classical regression assumption along the way.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange) ![Report](https://img.shields.io/badge/Report-LaTeX%20%7C%20PDF-green)

---

## Table of contents
1. [Project overview](#project-overview)
2. [Key findings](#key-findings)
3. [Dataset](#dataset)
4. [Method](#method)
5. [Results](#results)
6. [Repository structure](#repository-structure)
7. [How to run](#how-to-run)
8. [Limitations and future work](#limitations-and-future-work)
9. [Team](#team)
10. [References](#references)

---

## Project overview

**Research question:** can today's return, market return, 20-day volatility, 20-day momentum, volume change and high-low spread predict a stock's return on the *next* trading day?

**What the project covers**
- Data cleaning and exploratory data analysis (EDA)
- Feature engineering without look-ahead bias (features use day *t*, target is day *t+1*)
- Multiple linear regression (OLS) with a **chronological** train/test split
- Hypothesis tests, confidence intervals and robust (HC3) standard errors
- Multicollinearity (VIF), outliers and influential points (leverage, Cook's distance)
- Heteroscedasticity (Breusch-Pagan), autocorrelation (Durbin-Watson, Ljung-Box) and normality checks
- Model selection (AIC, BIC) and out-of-sample evaluation against naive baselines
- Confidence and prediction intervals

---

## Key findings

- **Almost nothing is explained.** In-sample R² is about **0.115%**; out-of-sample R² is **-0.0012**.
- **Volatility (positive) and momentum (negative) are the significant predictors** under HC3 robust errors, but their economic effect is tiny (about +0.036 and -0.032 percentage points for a one-standard-deviation move, against a daily standard deviation of roughly 1.5 points).
- **The model does not beat naive baselines** (RMSE 0.014697 versus 0.014688 for the training-mean forecast).
- **Directional accuracy (51.35%) is below the 52.5% obtained by always predicting "up".**
- **Diagnostics:** negligible multicollinearity (max VIF 1.50), strong heteroscedasticity, extremely fat-tailed residuals (kurtosis about 69), no autocorrelation (Durbin-Watson 1.99).
- **The results are consistent with weak-form market efficiency:** simple price and volume features do not reliably predict next-day returns.

---

## Dataset

| Item | Value |
|---|---|
| File | `all_stocks_5yr.csv` (Kaggle: "S&P 500 stock data") |
| Stocks | 505 S&P 500 companies |
| Period | 8 Feb 2013 to 7 Feb 2018 |
| Rows | 619,040 raw, 619,028 after cleaning, 608,423 in the modelling sample |
| Columns | `date`, `open`, `high`, `low`, `close`, `volume`, `Name` (ticker) |

The dataset is **not included in this repository** (or place it in `data/`). Download it from Kaggle by searching for "S&P 500 stock data".

> The file contains **no S&P 500 index series**. The market return is therefore an **equal-weighted average of the 505 stocks' daily returns**. This is a proxy, not the index.

---

## Method

### Features (all use information up to day *t*)

| Variable | Definition | Meaning |
|---|---|---|
| `next_return` | next observed daily return | **Target** |
| `daily_return` | `close_t / close_(t-1) - 1` | Today's return |
| `market_return` | equal-weighted mean of `daily_return` across stocks on day *t* | Market conditions |
| `volatility_20d` | std. dev. of the last 20 daily returns | Volatility |
| `momentum_20d` | `close_t / close_(t-20) - 1` | Momentum (about 1 month) |
| `volume_change` | `volume_t / volume_(t-1) - 1` | Trading-activity change |
| `high_low_spread` | `(high - low) / close` | Intraday price range |

### Model

```
next_return = b0 + b1*daily_return + b2*market_return + b3*volatility_20d
              + b4*momentum_20d + b5*volume_change + b6*high_low_spread + error
```

Pooled OLS with one common intercept (no stock fixed effects).

### Train / test split (chronological, never shuffled)

| Set | Period | Observations |
|---|---|---|
| Train | 2013-03-11 to 2017-02-10 | 483,866 |
| Test | 2017-02-13 to 2018-02-06 | 124,557 |

The earliest 80% of distinct dates are used for training and the latest 20% for testing.

### Baselines
Predict zero every day, and predict the training-mean return every day.

---

## Results

### Predictive performance (test set)

| Forecast | MAE | RMSE |
|---|---|---|
| Linear regression | 0.009429 | 0.014697 |
| Baseline: zero | 0.009431 | 0.014696 |
| Baseline: training mean | 0.009414 | 0.014688 |

Test R² = **-0.0012**, directional accuracy = **51.35%**.

### Coefficients (full training set, classical OLS)

| Variable | Coefficient | p-value |
|---|---|---|
| `daily_return` | -0.00141 | 0.410 |
| `market_return` | 0.00605 | 0.063 |
| `volatility_20d` | 0.04773 | < 0.001 |
| `momentum_20d` | -0.00469 | < 0.001 |
| `volume_change` | -0.00005 | 0.225 |
| `high_low_spread` | 0.00859 | < 0.001 |

Overall F-test: F = 92.51, p < 0.001. On a 50,000-row sample with HC3 robust errors, `volatility_20d` and `momentum_20d` remain significant (`daily_return` marginally).

### Diagnostics

| Check | Result |
|---|---|
| Multicollinearity (VIF) | all below 1.51 |
| Influential points | 2,790 of 50,000 above Cook's 4/n cut-off; max Cook's D = 0.333 (CHK, March 2016) |
| Heteroscedasticity (Breusch-Pagan) | LM = 1710.8, p < 0.001 (variance not constant) |
| Autocorrelation | Durbin-Watson = 1.994; Ljung-Box p = 0.346 (lag 5), 0.488 (lag 10) |
| Normality | residual kurtosis about 69; Jarque-Bera p < 0.001 |

The full report also adds date-clustered standard errors, under which no predictor is significant at the 5% level.

---

## Repository structure

```
.
├── README.md
├── stock_return_prediction_using_multiple_linear_regression.ipynb   # full analysis
├── data/
│   └── all_stocks_5yr.csv                  # download separately (see Dataset)
├── report/
│   ├── Stock_Return_Prediction_Report.pdf  # final report
│   ├── Stock_Return_Prediction_Report.tex  # LaTeX source
│   └── figs/                               # figures used in the report
└── requirements.txt
```

Adjust the tree to match what you actually upload.

---

## How to run

### Option A: Google Colab
1. Upload `all_stocks_5yr.csv` to Google Drive.
2. Open the notebook in Colab and mount Drive.
3. Set the data path in the loading cell, then run all cells.

### Option B: locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Put `all_stocks_5yr.csv` in `data/` and change the notebook's data path to `data/all_stocks_5yr.csv`.

### requirements.txt
```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
statsmodels
jupyter
```

### Build the report (optional)
Upload the `report/` folder to [Overleaf](https://www.overleaf.com) and compile `Stock_Return_Prediction_Report.tex`, or run `pdflatex` twice locally.

---

## Limitations and future work

- The market return is an equal-weighted proxy, not the S&P 500 index.
- A single common intercept; stock and date fixed effects would be a natural improvement.
- Diagnostics use a 50,000-row sample; there is one chronological split covering about a year of test data.
- Only linear relationships; regularised (Ridge, LASSO) and non-linear models could be compared against the same baselines.
- Further robustness checks: weighted least squares, Newey-West errors, Fama-MacBeth regressions.
- No transaction costs; some extreme returns may reflect data issues.

---

## Team

| Name | Roll No. | Contribution |
|---|---|---|
| TODO | TODO | TODO |
| TODO | TODO | TODO |
| TODO | TODO | TODO |

**Guide:** TODO  |  **Institution:** TODO

---

## References

- Fama, E.F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2).
- Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1).
- Petersen, M.A. (2009). Estimating standard errors in finance panel data sets. *Review of Financial Studies*, 22(1).
- Montgomery, Peck and Vining (2012). *Introduction to Linear Regression Analysis*, 5th ed. Wiley.

---

## License

TODO: choose a license (for example MIT) and add a `LICENSE` file. This project is for educational purposes only and is **not financial advice**.
