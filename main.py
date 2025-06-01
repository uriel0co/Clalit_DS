import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess
from src.modeling import train_and_evaluate_xgboost, train_and_evaluate_nn

def main():
    data_path = "data/raw/Prediction home assignment data.csv"
    df = pd.read_csv(data_path)
    # Preprocess data
    X, y, le, scaler, feature_names, class_names = preprocess(df)

    # Store feature names for downstream explainability
    ##feature_names = list(X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set model save paths
    model_paths = {
        "xgboost": os.path.join("models", "xgboost.pkl"),
        "neural_net": os.path.join("models", "neural_net.pt"),
    }

    # Train and evaluate XGBoost
    print("Training and evaluating XGBoost...")
    
    train_and_evaluate_xgboost(
        X_train, X_test, y_train, y_test, model_paths["xgboost"], feature_names, class_names
    )

    # Train and evaluate Neural Network
    print("Training and evaluating Neural Network...")
    train_and_evaluate_nn(
        X_train, X_test, y_train, y_test, model_paths["neural_net"]
    )

if __name__ == "__main__":
    main()