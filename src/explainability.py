import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_xgboost(X, model_path, feature_names, class_names, max_display=10):
    """
    Plots the mean SHAP values (signed, not absolute) for each feature and class
    on a single grouped bar plot, and prints a textual explanation of the results.

    Parameters:
    - X: Features (numpy array or DataFrame)
    - model_path: Path to the saved XGBoost model
    - feature_names: List of feature names
    - class_names: List of class names
    - max_display: Number of top features to display (default 10)
    """
    print("=== SHAP Explainability (Signed Mean, per Class) for XGBoost ===")
    # Load the trained model from file
    model = joblib.load(model_path)
    # Compute SHAP values using a sample of the data (first 200 rows)
    X_sample = X[:200] if len(X) > 200 else X
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap_vals = shap_values.values

    # If binary, expand dims for uniformity
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[..., np.newaxis]

    num_classes = shap_vals.shape[-1]
    num_features = shap_vals.shape[1]

    # Compute mean signed SHAP for each feature and class
    mean_signed_shap = np.mean(shap_vals, axis=0)  # shape: (features, classes)

    # Select top features by max absolute mean SHAP across classes
    feature_importance = np.max(np.abs(mean_signed_shap), axis=1)
    top_idx = np.argsort(feature_importance)[-max_display:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_mean_signed = mean_signed_shap[top_idx, :]  # shape: (max_display, num_classes)

    # Colors for each class
    cmap = plt.get_cmap("tab10")
    class_colors = [cmap(i) for i in range(num_classes)]

    # Grouped bar plot
    plt.figure(figsize=(12, max_display * 0.4 + 2))
    bar_width = 0.8 / num_classes
    indices = np.arange(max_display)

    for c in range(num_classes):
        plt.barh(
            indices + c * bar_width,
            top_mean_signed[:, c],
            height=bar_width,
            color=class_colors[c],
            label=f"{class_names[c]}",
            edgecolor='black'
        )

    plt.yticks(indices + bar_width * (num_classes - 1) / 2, top_features)
    plt.xlabel("Mean SHAP Value (Signed Impact)")
    plt.title("Top Feature SHAP Values by Class (Signed Mean)")
    plt.axvline(0, color='grey', linewidth=1, linestyle='--')
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()
