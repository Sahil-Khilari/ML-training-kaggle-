import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def load_and_preprocess(data_path, target):

    df = pd.read_csv(data_path)

    # Drop customerID but store it for prediction output
    customer_ids = df["customerID"]
    df = df.drop(columns=["customerID"])

    # Fix TotalCharges column
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.fillna(0)

    # Encode target
    df[target] = LabelEncoder().fit_transform(df[target])

    X = df.drop(columns=[target])
    y = df[target]

    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", "passthrough", categorical)
        ]
    )

    # Convert categorical columns to numeric using pandas
    X = pd.get_dummies(X, columns=categorical)

    return X, y, customer_ids


def evaluate_model(y_true, y_pred, scores):

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, scores)

    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, roc, cm


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--C", nargs="+", type=float, required=True)

    args = parser.parse_args()

    X, y, customer_ids = load_and_preprocess(args.data, args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []
    predictions_df = None

    for c_val in args.C:

        model = SVC(
            kernel="linear",
            C=c_val,
            probability=True,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc, prec, rec, f1, roc, cm = evaluate_model(y_test, preds, probs)

        num_sv = model.n_support_.sum()

        print(f"\n===== Results for C = {c_val} =====")
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc:.4f}")
        print(f"Support Vectors: {num_sv}")

        results.append({
            "C": c_val,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": roc,
            "Support_Vectors": num_sv
        })

        if predictions_df is None:
            predictions_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": preds,
                "Score": probs
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv("svm_linear_results.csv", index=False)
    predictions_df.to_csv("test_predictions.csv", index=False)

    print("\n===== Summary Table =====")
    print(results_df)


if __name__ == "__main__":
    main()