import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, roc_curve
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np

input_file = "/Users/liyiman/coding/NodeFormer/data/fraud/contrast_特征表.xlsx"
df = pd.read_excel(input_file)

"""机器学习"""

def logistic_regression_classification(df):
    """
    Perform logistic regression classification on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create logistic regression model
    model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"lr_Accuracy: {acc:.4f}")
    print(f"lr_ROC-AUC: {roc_auc:.4f}")
    print(f"lr_Recall: {recall:.4f}")


    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()


def random_forest_classification(df):
    """
    Perform random forest classification on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create random forest model
    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"rf_Accuracy: {acc:.4f}")
    print(f"rf_ROC-AUC: {roc_auc:.4f}")
    print(f"rf_Recall: {recall:.4f}")

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()


def svm_classification(df):
    """
    Perform SVM classification on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create SVM model
    model = SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"svm_Accuracy: {acc:.4f}")
    print(f"svm_ROC-AUC: {roc_auc:.4f}")
    print(f"svm_Recall: {recall:.4f}")

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()


def xgboost_classification(df):
    """
    Perform XGBoost classification on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create XGBoost model
    model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print(f"xgb_Accuracy: {acc:.4f}")
    print(f"xgb_ROC-AUC: {roc_auc:.4f}")
    print(f"xgb_Recall: {recall:.4f}")

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()


logistic_regression_classification(df)
random_forest_classification(df)
svm_classification(df)
xgboost_classification(df)


"""深度学习"""
def ann_classification(df):
    # Set a manual seed for reproducibility
    torch.manual_seed(42)
    """
    Perform ANN classification using PyTorch on the input DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame where the first column is the ID, middle columns are features, 
                    and the last column is the label.

    Returns:
    None: Prints evaluation metrics and displays the ROC curve.
    """
    # Ensure features and labels are correctly selected
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, stratify=y_np, random_state=42)

    # Convert back to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Recompute sample weights for training set
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # # Calculate class weights for imbalanced data
    # class_counts = np.bincount(y.astype(int))
    # class_weights = 1.0 / class_counts
    # sample_weights = class_weights[y.astype(int)]

    # sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # # Split into training and testing datasets
    # dataset = TensorDataset(X_tensor, y_tensor)
    # train_size = int(0.7 * len(dataset))
    # test_size = len(dataset) - train_size

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Define a simple ANN model
    class ANNModel(nn.Module):
        def __init__(self, input_size):
            super(ANNModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x

    model = ANNModel(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(50):  # Number of epochs
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            y_true.extend(y_batch.tolist())
            y_scores.extend(y_pred.tolist())

    y_pred_binary = [1 if score > 0.5 else 0 for score in y_scores]

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_scores)
    recall = recall_score(y_true, y_pred_binary)

    print(f"ann_Accuracy: {acc:.4f}")
    print(f"ann_ROC-AUC: {roc_auc:.4f}")
    print(f"ann_Recall: {recall:.4f}")

    # # Plot ROC curve
    # fpr, tpr, _ = roc_curve(y_true, y_scores)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="best")
    # plt.show()

ann_classification(df)