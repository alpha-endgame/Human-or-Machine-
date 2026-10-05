import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from text_features import FEATURE_COLUMNS, texts_to_dataframe


SUPPLIED_FEATURE_MAP = {
    "word_count": "word_count",
    "character_count": "character_count",
    "sentence_count": "sentence_count",
    "avg_sentence_length": "average_sentence_length",
    "flesch_reading_ease": "flesch_reading_ease",
    "gunning_fog_index": "gunning_fog_index",
    "punctuation_ratio": "punctuation_ratio",
}


def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    required = {"text_content", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    model_df = df[["text_content", "label"]].copy()
    model_df["_source_index"] = model_df.index

    model_df = model_df.dropna(subset=["text_content", "label"])
    model_df["text_content"] = model_df["text_content"].astype(str).str.strip()
    model_df = model_df[model_df["text_content"].str.len() > 0].copy()
    model_df["label"] = model_df["label"].astype(int)

    unexpected = sorted(set(model_df["label"].unique()) - {0, 1})
    if unexpected:
        raise ValueError(f"Expected labels 0 and 1. Found: {unexpected}")

    label_counts = model_df.groupby("text_content")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index

    if len(conflicting_texts) > 0:
        model_df = model_df[
            ~model_df["text_content"].isin(conflicting_texts)
        ].copy()

    before = len(model_df)
    model_df = model_df.drop_duplicates(
        subset=["text_content"],
        keep="first",
    ).reset_index(drop=True)

    print(f"Rows loaded: {len(df)}")
    print(f"Conflicting exact texts removed: {len(conflicting_texts)}")
    print(f"Duplicate exact texts removed: {before - len(model_df)}")
    print(f"Rows used for modelling: {len(model_df)}")
    print("Class distribution:")
    print(model_df["label"].value_counts().sort_index())

    return df, model_df


def run_feature_integrity_audit(df, model_df, recomputed_features, output_path):
    missing = [
        source_name
        for source_name in SUPPLIED_FEATURE_MAP
        if source_name not in df.columns
    ]

    if missing:
        print("\nFeature-integrity audit skipped.")
        print("Missing supplied columns:", missing)
        return None

    source_indices = model_df["_source_index"].to_numpy()

    supplied = (
        df.loc[source_indices, list(SUPPLIED_FEATURE_MAP.keys())]
        .reset_index(drop=True)
        .rename(columns=SUPPLIED_FEATURE_MAP)
    )

    rows = []

    for feature in FEATURE_COLUMNS:
        supplied_values = pd.to_numeric(supplied[feature], errors="coerce")
        recomputed_values = pd.to_numeric(
            recomputed_features[feature],
            errors="coerce",
        )

        valid = supplied_values.notna() & recomputed_values.notna()

        if valid.sum() >= 2:
            correlation = supplied_values[valid].corr(recomputed_values[valid])
            mae = (
                supplied_values[valid] - recomputed_values[valid]
            ).abs().mean()
            exact_rate = np.isclose(
                supplied_values[valid],
                recomputed_values[valid],
                rtol=1e-6,
                atol=1e-8,
            ).mean()
        else:
            correlation = np.nan
            mae = np.nan
            exact_rate = np.nan

        rows.append(
            {
                "feature": feature,
                "pearson_correlation": correlation,
                "mean_absolute_error": mae,
                "exact_match_rate": exact_rate,
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)

    print("\nFeature-integrity audit:")
    print(report.to_string(index=False))

    weak = report[
        report["pearson_correlation"].abs() < 0.20
    ]["feature"].tolist()

    if weak:
        print("\nWarning:")
        print("These supplied features have weak agreement with values")
        print("recomputed from the raw text:")
        print(weak)

    count_features = ["word_count", "character_count", "sentence_count"]
    print("\nNon-integer values in supplied literal-count columns:")
    for feature in count_features:
        values = pd.to_numeric(supplied[feature], errors="coerce").dropna()
        count = int((~np.isclose(values % 1, 0)).sum())
        print(f"{feature}: {count}")

    return report


def build_parameter_grid(quick=False):
    if quick:
        return {
            "n_estimators": [100],
            "max_depth": [4, 8],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt"],
        }

    return {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8, 12],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5],
    }


def train_model(csv_path, model_path, metrics_path, audit_path, quick=False):
    df, model_df = load_and_clean_data(csv_path)

    X = texts_to_dataframe(model_df["text_content"])
    y = model_df["label"].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    run_feature_integrity_audit(
        df=df,
        model_df=model_df,
        recomputed_features=X,
        output_path=audit_path,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    base_model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        bootstrap=True,
        n_jobs=-1,
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=build_parameter_grid(quick=quick),
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True,
        refit=True,
        verbose=1,
    )

    print("\nTraining Random Forest...")
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    train_f1 = f1_score(y_train, train_predictions, zero_division=0)
    test_f1 = f1_score(y_test, test_predictions, zero_division=0)
    test_precision = precision_score(y_test, test_predictions, zero_division=0)
    test_recall = recall_score(y_test, test_predictions, zero_division=0)

    cv_results = pd.DataFrame(grid.cv_results_)
    best_row = cv_results.loc[grid.best_index_]
    cv_train_f1 = float(best_row["mean_train_score"])
    cv_validation_f1 = float(best_row["mean_test_score"])

    matrix = confusion_matrix(y_test, test_predictions)

    metrics = {
        "rows_used": int(len(model_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_parameters": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "cv_train_f1": cv_train_f1,
        "cv_validation_f1": cv_validation_f1,
        "cv_f1_gap": cv_train_f1 - cv_validation_f1,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "accuracy_gap": float(train_accuracy - test_accuracy),
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "f1_gap": float(train_f1 - test_f1),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            test_predictions,
            target_names=["Human", "AI"],
            zero_division=0,
            output_dict=True,
        ),
    }

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {0: "Human", 1: "AI"},
        "feature_definition": (
            "7 linguistic features computed directly from raw text"
        ),
    }

    joblib.dump(bundle, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nBest parameters:")
    print(grid.best_params_)

    print(f"\nBest 5-fold CV F1: {grid.best_score_:.4f}")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Accuracy gap: {(train_accuracy - test_accuracy):.2%}")
    print(f"Training F1: {train_f1:.2%}")
    print(f"Test F1: {test_f1:.2%}")
    print(f"F1 gap: {(train_f1 - test_f1):.2%}")
    print(f"Test precision: {test_precision:.2%}")
    print(f"Test recall: {test_recall:.2%}")

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Human", "AI"],
            zero_division=0,
        )
    )

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Feature audit saved to: {audit_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Human vs Machine Random Forest model."
    )
    parser.add_argument(
        "--data",
        default="ai_detection_dataset.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--model",
        default="random_forest_ai_detector_regularized.pkl",
        help="Path for the saved model bundle.",
    )
    parser.add_argument(
        "--metrics",
        default="training_metrics.json",
        help="Path for the saved metrics JSON file.",
    )
    parser.add_argument(
        "--audit",
        default="feature_integrity_report.csv",
        help="Path for the feature-integrity report.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller grid for a fast test run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_model(
        csv_path=Path(args.data),
        model_path=Path(args.model),
        metrics_path=Path(args.metrics),
        audit_path=Path(args.audit),
        quick=args.quick,
    )
