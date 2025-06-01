# Clalit Innovation – Data Science Home Assignment

## Project Structure

```
clalit_ds_project/
│
.
├── main.py
├── README.md
├── eda_clalit.ipynb
├── data/
│   └── raw/
│       └── Prediction home assignment data.csv
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── neural_net.pt
└── src/
    ├── preprocessing.py
    ├── modeling.py
    └── explainability.py
```

## How to Run
1. **Clone the Repo**
    ```
    git clone https://github.com/uriel0co/Clalit_DS.git
    ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt --user
   ```

3. **Add the dataset**  
   Place the `Prediction home assignment data.csv` file inside the `data/raw/` directory.

4. **Run the project:**
   ```
   python main.py
   ```

5. **Outputs:**  
   - EDA plots will be shown.
   - Model metrics will print to console.
   - Model weights are saved in `models/`.
   - SHAP explainability plots for XGBoost will be shown.

## What's inside?

- **eda_clalit.ipynb**: EDA Notebook.
- **main.py**: Entry point, glues everything together.
- **src/**: All code modules for preprocessing, modeling, explainability.
- **models/**: Trained model weights and encoders.
- **data/raw/**: The dataset.
- **README.md**: This file.

## Project Philosophy

- **Separation of concerns**: code is modular and maintainable.
- **Reproducibility**: All model weights are saved in `models/`.
- **Scalable**: Can be extended for bigger data and more models.
- **Explainable**: SHAP is used for model interpretability.

## Data Improvement Suggestions
- **Temporal Data:** Time series of measurements (for example, trends in blood pressure)
- **Lifestyle Data:** Diet, physical activity trackers, sleep quality (from wearables)
- **Socioeconomic Data:** Income, education, region
- **Clinical History:** More granular medication types
- **Text Data:** Doctor visit notes (NLP features)
- **Lab Results:** More detailed blood/urine test results
- **Data Quality:** Ensure no negative/implausible ages, harmonize categorical values
- **External Data:** Environmental exposures (air quality, pollution, neighborhood factors)

## Author

Uriel Cohen
