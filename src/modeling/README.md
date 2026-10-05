# train.py and predict.py

This README explains the two main Python scripts used in the Human vs Machine project.

- `train.py` is used to train and evaluate the Random Forest model.
- `predict.py` is used to load the trained model and classify new text as Human or AI.

---

# 1. train.py

## Purpose

`train.py` trains the Random Forest classifier using the Human vs Machine dataset.

It also checks the dataset, calculates text features, tunes the model, evaluates performance, and saves the final trained model.

## What train.py Does

The script performs the following steps:

1. Loads the training dataset.
2. Checks that the required columns exist.
3. Removes empty text samples.
4. Checks for duplicate text.
5. Removes conflicting duplicate samples when the same text has different labels.
6. Recalculates the seven linguistic features directly from the raw text.
7. Runs the feature-integrity audit.
8. Splits the dataset into training and testing data.
9. Uses GridSearchCV to tune the Random Forest model.
10. Uses 5-fold cross-validation.
11. Evaluates the final model.
12. Saves the trained model.
13. Saves the performance metrics.
14. Saves the feature-integrity report.

## Required Dataset Columns

The training dataset must contain:

```text
text_content
label
```

The project uses:

```text
0 = Human
1 = AI
```

## Features Used for Training

The final model uses seven features calculated directly from `text_content`:

1. Word count
2. Character count
3. Sentence count
4. Average sentence length
5. Flesch Reading Ease
6. Gunning Fog Index
7. Punctuation ratio

These features are calculated using `text_features.py`.

## Required Files

Keep the following files in the same project folder:

```text
train.py
text_features.py
ai_detection_dataset.csv
```

## Install Required Packages

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train.py
```

By default, the script looks for:

```text
ai_detection_dataset.csv
```

## Use a Different Dataset

```bash
python train.py --data path/to/dataset.csv
```

## Quick Test Training

To run a smaller grid search and check that the code works:

```bash
python train.py --quick
```

The quick option is useful for testing the script.

For the final project model, use:

```bash
python train.py
```

## Files Created After Training

The script creates:

```text
random_forest_ai_detector_regularized.pkl
training_metrics.json
feature_integrity_report.csv
```

### random_forest_ai_detector_regularized.pkl

This is the trained Random Forest model.

It is used by:

```text
predict.py
app.py
```

### training_metrics.json

This file stores the model evaluation results.

These include:

- Best model parameters
- Cross-validation F1-score
- Training accuracy
- Test accuracy
- Accuracy gap
- Training F1-score
- Test F1-score
- Precision
- Recall
- Confusion matrix
- Classification report

### feature_integrity_report.csv

This file stores the comparison between the supplied dataset features and the same features recalculated from the raw text.

---

# 2. predict.py

## Purpose

`predict.py` uses the trained model to classify new text.

It can predict whether a text sample is more likely to be:

```text
Human Written
```

or:

```text
AI Generated
```

## How predict.py Works

The script:

1. Loads the saved Random Forest model.
2. Receives new text from the user.
3. Calculates the same seven features used during training.
4. Passes the features to the trained model.
5. Produces a Human or AI prediction.
6. Displays the class probabilities and confidence.

Using the same feature extraction during training and prediction helps keep the model pipeline consistent.

## Required Files

Keep these files in the same folder:

```text
predict.py
text_features.py
random_forest_ai_detector_regularized.pkl
```

The model file must be created first by running `train.py`.

## Predict Text Directly

Run:

```bash
python predict.py --text "Paste the text you want to test here."
```

Example:

```bash
python predict.py --text "Artificial intelligence is increasingly used in many industries."
```

## Predict a Text File

You can also classify text stored in a `.txt` file:

```bash
python predict.py --file sample.txt
```

## Show Extracted Features

To display the seven features calculated from the input text:

```bash
python predict.py --text "Your text here." --show-features
```

The script will show:

- Word count
- Character count
- Sentence count
- Average sentence length
- Flesch Reading Ease
- Gunning Fog Index
- Punctuation ratio

## Prediction Output

A normal prediction contains:

```text
Prediction
AI probability
Human probability
Confidence
```

Example:

```text
Prediction: Human Written
AI probability: 38.20%
Human probability: 61.80%
Confidence: 61.80%
```

## Use a Different Model File

If the trained model has a different name or location:

```bash
python predict.py --model path/to/model.pkl --text "Text to test"
```

---

# Recommended Workflow

The normal project workflow is:

```text
Training Dataset
      ↓
   train.py
      ↓
Trained Model
      ↓
   predict.py
      ↓
Human or AI Prediction
```

Run training first:

```bash
python train.py
```

Then make predictions:

```bash
python predict.py --text "Text to test"
```

---

# Important Note

The model output is a statistical estimate.

A Human or AI prediction should not be treated as absolute proof of authorship.

The quality of the prediction depends on the quality of the training data, the features used, and how similar the new text is to the data used during model development.
