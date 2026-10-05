# app.py

This file starts the Gradio dashboard for the Human vs Machine project.

The dashboard lets a user paste text and receive:

- Human or AI prediction
- AI probability
- Human probability
- Model confidence

## Required files

Keep these files in the same folder:

- `app.py`
- `text_features.py`
- `random_forest_ai_detector_regularized.pkl`

Create the model file first by running `train.py`.

## Install packages

```bash
pip install gradio pandas scikit-learn joblib
```

## Start the dashboard

```bash
python app.py
```

Gradio will print a local address in the terminal.

Open that address in a web browser.

It is usually similar to:

```text
http://127.0.0.1:7860
```

## How the dashboard works

1. The user pastes text.
2. The same seven text features used during training are calculated.
3. The saved Random Forest model makes a prediction.
4. The dashboard shows the predicted class and probabilities.

## Model file

By default, `app.py` looks for:

```text
random_forest_ai_detector_regularized.pkl
```

The model file should be in the same folder as `app.py`.

## Important note

The dashboard result is a statistical estimate. It is not absolute proof of authorship.
