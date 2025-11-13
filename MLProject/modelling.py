import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Load dataset
data = pd.read_csv("namadataset_preprocessing/StudentsPerformance_preprocessed.csv")
data["average_score"] = data[["math score","reading score","writing score"]].mean(axis=1)
threshold = data["average_score"].mean()
data["performance"] = data["average_score"].apply(lambda x: "high" if x>=threshold else "low")

X = data.drop(columns=["performance"])
y = data["performance"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

model = RandomForestClassifier(random_state=args.random_state)
model.fit(X_train, y_train)

# Logging ke MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
