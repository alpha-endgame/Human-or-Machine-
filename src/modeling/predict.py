import argparse
from pathlib import Path

import joblib
import pandas as pd

from text_features import FEATURE_COLUMNS, extract_text_features


def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run train.py first."
        )

    bundle = joblib.load(model_path)

    if "model" not in bundle:
        raise ValueError("The model bundle does not contain a model.")

    return bundle


def predict_text(text, bundle):
    if text is None or not str(text).strip():
        raise ValueError("Text cannot be empty.")

    feature_values = extract_text_features(text)
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)

    X = pd.DataFrame(
        [[feature_values[column] for column in feature_columns]],
        columns=feature_columns,
    )

    model = bundle["model"]
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    class_probability = dict(zip(model.classes_, probabilities))

    human_probability = float(class_probability.get(0, 0.0) * 100)
    ai_probability = float(class_probability.get(1, 0.0) * 100)

    label = (
        "AI Generated"
        if prediction == 1
        else "Human Written"
    )

    return {
        "prediction": label,
        "ai_probability": round(ai_probability, 2),
        "human_probability": round(human_probability, 2),
        "confidence": round(max(ai_probability, human_probability), 2),
        "features": feature_values,
    }


def get_input_text(args):
    if args.text:
        return args.text

    if args.file:
        return Path(args.file).read_text(encoding="utf-8")

    print("Paste text below.")
    print("Press Ctrl+D on Linux/macOS or Ctrl+Z then Enter on Windows when done:")
    return "\n".join(iter(input, ""))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict whether text is Human or AI generated."
    )
    parser.add_argument(
        "--model",
        default="random_forest_ai_detector_regularized.pkl",
        help="Path to the trained model file.",
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text",
        help="Text to classify.",
    )
    input_group.add_argument(
        "--file",
        help="Path to a text file to classify.",
    )

    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Print the seven extracted features.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    bundle = load_model(Path(args.model))
    text = get_input_text(args)
    result = predict_text(text, bundle)

    print("\nPrediction:", result["prediction"])
    print(f"AI probability: {result['ai_probability']:.2f}%")
    print(f"Human probability: {result['human_probability']:.2f}%")
    print(f"Confidence: {result['confidence']:.2f}%")

    if args.show_features:
        print("\nExtracted features:")
        for name, value in result["features"].items():
            print(f"{name}: {value}")
