import os
import joblib
import numpy as np
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from src.explainability import explain_xgboost

import torch
import torch.nn as nn
import torch.optim as optim


# ## Model Training and Evaluation Choices Explained

# ### XGBoost

# - **Model Choice**: We use `XGBClassifier` with `use_label_encoder=False` and `eval_metric='mlogloss'`. XGBoost is robust for both categorical and
#  continuous data, handles missing values, and is generally strong for tabular classification.
#  The label encoder is disabled to avoid unnecessary warnings, and 'mlogloss' is a standard metric for multi-class problems during training.
  
# ### Neural Network (PyTorch)

# - **Architecture**: The neural network consists of two hidden layers (128 and 64 units) with ReLU activations and dropout for regularization.
#  This architecture balances model capacity and overfitting risk—128 and 64 are typical starting points for tabular data,
#  and dropouts of 0.3 and 0.2 help prevent overfitting without causing underfitting.
# - **Training**: The network uses the Adam optimizer (`lr=1e-3`), which is robust and requires little hyperparameter tuning.
#  The batch size (`32`) is a standard, efficient size for mini-batch training, and 30 epochs are a reasonable default for convergence on most
#  tabular datasets.
# - **Device Management**: The model automatically uses a GPU if available, ensuring efficient computation.

# ### Evaluation Metric

# - **F1 Score (macro-averaged)**: We report the macro-averaged F1 score, which computes the F1 score independently for each class
#  and then takes the average. This metric is chosen because:
#     - It gives equal weight to all classes, which is crucial in our imbalanced setting (e.g., upsampled autoimmune disorders).
#     - It balances precision and recall, making it sensitive to false positives and false negatives.
# - **Classification Report**: We also output the full classification report for a detailed breakdown by class.

# These choices ensure our models are robust, generalize well, and are evaluated fairly, especially given the class imbalance
#  and the nature of the prediction task.


# XGBoost
def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test, model_path, feature_names, class_names=None):
    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print("==== XGBoost F1 (macro) ====")
    print(f1)
    print(classification_report(y_test, y_pred))
    # Automatically run explainability after modeling
    print("==== XGBoost Explainability ====")
    explain_xgboost(X_test, model_path, feature_names, class_names=class_names)
    return f1

# PyTorch NN
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_and_evaluate_nn(X_train, X_test, y_train, y_test, model_path, epochs=30, batch_size=32, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.long)

    num_classes = len(np.unique(y_train.numpy()))
    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()
    f1 = f1_score(y_test, y_pred, average='macro')
    print("==== Neural Network (PyTorch) F1 (macro) ====")
    print(f1)
    print(classification_report(y_test, y_pred))
    return f1