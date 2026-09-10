[README.md](https://github.com/user-attachments/files/32038817/README.md)
# Predicting Football Player Market Value: A Machine Learning Approach

An end-to-end machine learning project predicting professional football players' transfer market values (€) using historical performance statistics, age curves, and positional traits extracted from the **Transfermarkt** relational dataset.

---

## 📌 Project Overview & Motivation

Football clubs spend hundreds of millions of euros each transfer window, yet player valuations are frequently biased by media reputation, commercial hype, club status, and agent leverage. 

This project addresses two core research questions:
1. **Can we model player market value objectively** using exclusively verifiable career performance metrics, age, and position?
2. **Where are the market's inefficiencies?** Can a data-driven model surface high-output, undervalued scouting targets for clubs operating under financial constraints?

---

## 🏗️ Repository Architecture

```text
├── data/
│   ├── players.csv                  # Base player profile data (DOB, position, market value)
│   └── appearances.csv              # Match event data (goals, assists, minutes, cards)
├── notebooks/
│   └── football_market_value.ipynb  # End-to-end pipeline (EDA, preprocessing, modeling)
├── presentations/
│   └── Football_Market_Value_Presentation.pptx # Presentation slide deck
├── README.md                        # Project documentation
└── requirements.txt                 # Environment dependencies
```

---

## 📊 Dataset & Relational Preprocessing

The project joins two relational tables from Transfermarkt linked by `player_id`:

- **`players.csv`**: Contains player demographic and contractual data (`player_id`, `name`, `date_of_birth`, `position`, `market_value_in_eur`).
- **`appearances.csv`**: Granular match event logs aggregated per player across their career (`goals`, `assists`, `minutes_played`, `yellow_cards`).

### Feature Engineering & Data Hygiene
1. **Relational Aggregation**: Career performance totals were calculated via `groupby('player_id').agg(...)` summing goals, assists, minutes played, and yellow cards.
2. **Age Derivation**: Player age was derived from `date_of_birth` using a standard reference year (`2026 - date_of_birth.year`).
3. **Categorical Encoding**: One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`) applied to categorical `position` tags.
4. **Data Cleansing**: Dropped records with missing target values (`market_value_in_eur`), null ages, or incomplete match histories.
5. **Partitioning**: 80/20 train/test split with `random_state=42`.

---

## 🔍 Key Exploratory Data Analysis (EDA)

1. **Age vs. Valuation Curve**:
   - Market valuation exhibits an inverted U-curve, peaking in the mid-to-late 20s before sharply tapering off.
   - Older veterans exhibit wider valuation variance, confirming that age and market value share a strictly non-linear relationship.
2. **Performance Correlation**:
   - Direct goal output (`r = 0.26`), assists (`r = 0.27`), and total minutes played (`r = 0.23`) show the strongest positive linear correlations with market value.
   - Linear age correlation (`r = -0.22`) underrepresents its real impact due to non-linearity, motivating ensemble tree algorithms over simple linear models.

---

## 🤖 Model Benchmark & Evaluation

Four distinct regression models were benchmarked across identical held-out test data using Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and the Coefficient of Determination (**R²**):

| Model Architecture | MAE (€) | RMSE (€) | Test R² Score | 5-Fold Mean CV R² |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | €2,648,437.52 | €6,181,564.15 | 0.1582 | 0.1456 |
| **Random Forest Regressor (Initial)** | €1,485,390.72 | €4,510,615.83 | 0.5518 | 0.5142 |
| **Gradient Boosting Regressor** | €1,481,120.78 | €4,541,810.47 | 0.5455 | 0.5198 |
| **Support Vector Regressor (SVR - RBF)** | €1,510,879.38 | €5,956,129.95 | 0.2184 | 0.2027 |

*Ensemble tree algorithms (Random Forest & Gradient Boosting) explained ~55% of valuation variance—more than 3× the predictive power of linear baselines.*

---

## ⚙️ Overfitting Diagnosis & Hyperparameter Regularization

Initial training of the Random Forest model revealed severe overfitting:
- **Initial Training R²**: `0.9189`
- **Testing R²**: `0.5518`
- **Initial R² Performance Gap**: `0.3671` (High risk of memorization)

### Regularization Adjustments
The Random Forest model was re-architected with strict leaf and tree depth constraints:
```python
RandomForestRegressor(
    n_estimators=150,
    max_depth=8,              # Constrains unconstrained branch depth
    min_samples_split=12,     # Demands sufficient statistical support to split
    min_samples_leaf=6,       # Enforces group averaging at leaf terminations
    random_state=42,
    n_jobs=-1
)
```

- **Post-Tuning Training R²**: `0.6722`
- **Post-Tuning Testing R²**: `0.5518`
- **Final R² Generalization Gap**: `0.1204` *(Diagnosed as a healthy, well-regularized production model)*.

---

## 🎯 Practical Scouting Application: Model-Identified Undervalued Targets

By comparing predicted market valuations against current Transfermarkt listing prices on held-out test data, the model flagged high-arbitrage scouting candidates:

| Player Name | Actual Market Value (€) | Predicted Market Value (€) | Model Valuation Surplus (€) |
| :--- | :---: | :---: | :---: |
| **Konstantin Tyukavin** | €17,000,000.00 | €66,128,347.21 | **+€49,128,347.21** |
| **Ivan Oblyakov** | €10,000,000.00 | €58,127,013.99 | **+€48,127,013.99** |
| **Nazar Voloshyn** | €5,000,000.00 | €43,894,896.17 | **+€38,894,896.17** |
| **Lewis Ferguson** | €14,000,000.00 | €50,268,671.37 | **+€36,268,671.37** |
| **Jonathan David** | €30,000,000.00 | €66,099,810.57 | **+€36,099,810.57** |

> **Scouting Caveat**: Model predictions rely on aggregate career volume. Real-world recruitment workflows should treat these outputs as initial screening filters alongside tactical fit, contract length, wage demands, and injury history.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.9+ is installed.

### 2. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/your-username/football-market-value-prediction.git
cd football-market-value-prediction
pip install -r requirements.txt
```

### 3. Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### 4. Running the Pipeline
Execute the notebook or run your script:
```bash
jupyter notebook notebooks/football_market_value.ipynb
```

---

## 👤 Author & Acknowledgments

- **Author**: NGUYEN THE HUNG (Student ID: 23229586)
- **Data Source**: Transfermarkt Relational Football Dataset
