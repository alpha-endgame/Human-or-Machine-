from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from text_features import FEATURE_COLUMNS, extract_text_features


MODEL_PATH = Path("random_forest_ai_detector_regularized.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run train.py first to create "
            "random_forest_ai_detector_regularized.pkl."
        )

    bundle = joblib.load(MODEL_PATH)

    if "model" not in bundle:
        raise ValueError("The model bundle does not contain a model.")

    return bundle


MODEL_BUNDLE = load_model()
MODEL = MODEL_BUNDLE["model"]
MODEL_FEATURES = MODEL_BUNDLE.get("feature_columns", FEATURE_COLUMNS)


def predict_text(text):
    if text is None or not str(text).strip():
        return "Enter text to analyze", 0.0, 0.0, 0.0

    feature_dict = extract_text_features(text)

    values = pd.DataFrame(
        [[feature_dict[column] for column in MODEL_FEATURES]],
        columns=MODEL_FEATURES,
    )

    prediction = int(MODEL.predict(values)[0])
    probabilities = MODEL.predict_proba(values)[0]
    class_probability = dict(zip(MODEL.classes_, probabilities))

    human_probability = class_probability.get(0, 0.0) * 100
    ai_probability = class_probability.get(1, 0.0) * 100

    result = "AI Generated" if prediction == 1 else "Human Written"
    confidence = max(human_probability, ai_probability)

    return (
        result,
        round(ai_probability, 2),
        round(human_probability, 2),
        round(confidence, 2),
    )


CSS = """
body,
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.20), transparent 32%),
        linear-gradient(135deg, #071426 0%, #0a1d36 48%, #0b2340 100%) !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    min-height: 100vh !important;
    padding-top: 28px !important;
}

#dashboard_header {
    padding: 20px 22px;
    margin-bottom: 16px;
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 16px;
    background: linear-gradient(
        135deg,
        rgba(15, 42, 75, 0.95),
        rgba(10, 31, 57, 0.95)
    );
}

#dashboard_header h1,
#dashboard_header p,
#dashboard_header strong {
    color: #f8fafc !important;
}

.input-card,
.result-card {
    border: 1px solid rgba(96, 165, 250, 0.24) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    background: #0e2746 !important;
}

#text_input textarea,
.result-card input {
    background: #091a30 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(96, 165, 250, 0.28) !important;
    border-radius: 10px !important;
}

#analyze_btn {
    width: 155px !important;
    min-width: 155px !important;
    max-width: 155px !important;
    background: #16a34a !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
}

#dashboard_note {
    margin-top: 16px;
    padding: 12px 15px;
    background: #0a1f39;
    border: 1px solid rgba(96, 165, 250, 0.20);
    border-radius: 12px;
    color: #94a3b8 !important;
}
"""


THEME = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="slate",
    radius_size="md",
)


with gr.Blocks(
    css=CSS,
    theme=THEME,
    title="Human or Machine?",
) as demo:

    with gr.Column(elem_id="dashboard_header"):
        gr.Markdown(
            """
# Human or Machine?
**Random Forest AI vs Human Text Detector**

Paste a text sample below to estimate whether it is more likely
human-written or AI-generated.
"""
        )

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, elem_classes="input-card"):
            gr.Markdown("### Text to Analyze")

            text_input = gr.Textbox(
                placeholder="Paste or type the text you want to analyze...",
                lines=14,
                show_label=False,
                elem_id="text_input",
            )

            analyze_button = gr.Button(
                "Analyze Text",
                variant="primary",
                elem_id="analyze_btn",
            )

        with gr.Column(scale=1, elem_classes="result-card"):
            gr.Markdown("### Analysis Result")

            prediction_output = gr.Textbox(
                label="Prediction",
                interactive=False,
            )

            ai_output = gr.Number(
                label="AI Probability (%)",
                precision=2,
                interactive=False,
            )

            human_output = gr.Number(
                label="Human Probability (%)",
                precision=2,
                interactive=False,
            )

            confidence_output = gr.Number(
                label="Model Confidence (%)",
                precision=2,
                interactive=False,
            )

    gr.Markdown(
        """
**Note:** This is a statistical estimate from the trained model.
It is not absolute proof of authorship.
""",
        elem_id="dashboard_note",
    )

    analyze_button.click(
        fn=predict_text,
        inputs=text_input,
        outputs=[
            prediction_output,
            ai_output,
            human_output,
            confidence_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
