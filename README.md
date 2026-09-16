# Customer Click Prediction

Predicting whether a user will click on a digital ad, using logistic regression and a bit of statistical rigor to pick the right features before modeling.

## Overview

Digital ad campaigns generate a lot of user-level data, but not all of it actually explains why someone clicks. This project works through that question end-to-end: clean the data, statistically test which features matter, and build a classification model that predicts click behavior with strong accuracy.

The dataset (`advertising.csv`) contains 1,000 user records with browsing behavior, demographics, and ad interaction details.

## Workflow

**1. Data Cleaning**
- Loaded and inspected the dataset — no missing values or duplicates found
- Converted `Timestamp` to datetime and extracted `Hour` as a new feature

**2. Exploratory Data Analysis**
- Confirmed the target variable (`Clicked on Ad`) is perfectly balanced (50/50), so no resampling was needed

**3. Feature Selection**
- **ANOVA test** on continuous variables against the target:
  - `Daily Time Spent on Site`, `Age`, `Area Income`, and `Daily Internet Usage` were all statistically significant
  - `Hour` was not correlated and was dropped
- **Chi-square test** on categorical variables (`Ad Topic Line`, `City`, `Country`, `Male`):
  - None showed a significant relationship with the target, so all were excluded from the model

**4. Preprocessing**
- Applied `StandardScaler` to the selected continuous features (chosen over MinMaxScaler due to sensitivity to outliers, and because it pairs well with logistic regression)
- Split the data 80/20 into training and test sets

**5. Modeling**
- Trained a **Logistic Regression** model on the four selected features
- Evaluated using accuracy, precision, recall, and F1-score

## Results

| Metric | Score |
|---|---|
| Accuracy | **95%** |
| Precision (class 1) | 0.97 |
| Recall (class 1) | 0.94 |
| F1-score (class 1) | 0.95 |

The final model correctly classifies 95% of users in the test set, with strong, balanced precision and recall across both classes.

## Key Takeaway

A user's on-site behavior — time spent on the site, daily internet usage, age, and the income level of their area — is far more predictive of ad clicks than who they are demographically (gender) or where the ad was seen (city, country, topic). Sometimes the simplest, most interpretable model is also the most useful one.

## Tech Stack

- **Python**: pandas, NumPy
- **Visualization**: matplotlib, seaborn
- **Statistics**: SciPy (ANOVA, Chi-square)
- **Modeling**: scikit-learn (Logistic Regression, StandardScaler, train-test split)

## Repository Structure

```
├── customer_click_prediction.ipynb   # Full analysis and modeling notebook
├── advertising.csv                   # Dataset (add if not already present)
└── README.md
```

## Getting Started

```bash
pip install pandas numpy seaborn matplotlib scipy scikit-learn
jupyter notebook customer_click_prediction.ipynb
```

## Future Improvements

- Experiment with additional models (Random Forest, XGBoost) for comparison
- Try polynomial or interaction features among the significant predictors
- Deploy as a simple API or Streamlit app for real-time click prediction
