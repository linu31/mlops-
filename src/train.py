"""Train, compare and register Logistic Regression and Random Forest."""
from pathlib import Path
import json, joblib, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from .preprocessing import ROOT, NUMERIC, build_preprocessor
from .feature_store import build_feature_store, load_features

MODEL_NAME = "ShrimpPondOxygenRisk"

def main():
    feature_path = ROOT/"feature_store"/"pond_night_features.csv"
    if not feature_path.exists(): build_feature_store()
    df = load_features(feature_path)
    X = df[NUMERIC]
    y = df["oxygen_risk"].map({"Normal":0, "High":1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced", random_state=42)
    }
    mlflow.set_experiment("Shrimp-Pond-Oxygen-Risk")
    results=[]
    best_name, best_pipeline, best_f1 = None, None, -1
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test,pred),
                "high_risk_precision": precision_score(y_test,pred,zero_division=0),
                "high_risk_recall": recall_score(y_test,pred,zero_division=0),
                "high_risk_f1": f1_score(y_test,pred,zero_division=0)
            }
            mlflow.log_param("model", name)
            mlflow.log_param("positive_class", "High")
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
    		pipe,
    		artifact_path="model",
    		skops_trusted_types=[
        		"numpy.dtype",
        		"sklearn.tree._tree.Tree"
    		]
	    )
            results.append({"model":name, **metrics})
            if metrics["high_risk_f1"] > best_f1:
                best_name, best_pipeline, best_f1 = name, pipe, metrics["high_risk_f1"]
    joblib.dump(best_pipeline, ROOT/"models"/"best_model.joblib")
    pd.DataFrame(results).to_csv(ROOT/"reports"/"model_comparison.csv", index=False)
    (ROOT/"reports"/"classification_report.txt").write_text(
        f"Selected model: {best_name}\\nSelection metric: High-risk F1-score\\n" +
        pd.DataFrame(results).to_string(index=False), encoding="utf-8")
    # Register the selected model as a separate MLflow run.
    with mlflow.start_run(run_name=f"Register-{best_name}"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selected_high_risk_f1", best_f1)
        model_info = mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="selected_model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        try:
            mlflow.register_model(model_info.model_uri, MODEL_NAME)
        except Exception as exc:
            print(f"Registry step warning: {exc}")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Selected model: {best_name}; saved models/best_model.joblib")
    print(f"MLflow registry model name: {MODEL_NAME}")

if __name__ == "__main__":
    main()
