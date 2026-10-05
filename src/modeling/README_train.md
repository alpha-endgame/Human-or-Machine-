# train.py

This file trains the Random Forest model used in the Human vs Machine project.

## What it does

1. Loads the CSV dataset.
2. Checks that `text_content` and `label` exist.
3. Removes empty text, conflicting exact duplicates, and exact duplicate text.
4. Recalculates the seven text features from the raw text.
5. Compares the supplied CSV features with the recalculated features when those columns are available.
6. Splits the data into 80% training data and 20% test data.
7. Uses 5-fold GridSearchCV to tune the Random Forest.
8. Prints the training and test metrics.
9. Saves the trained model.
10. Saves the metrics and the feature-integrity report.

## Required files

Keep these files in the same folder:

- `train.py`
- `text_features.py`
- `ai_detection_dataset.csv`

## Install packages

```bash
pip install pandas numpy scikit-learn joblib
```

## Train the full model

```bash
python train.py
```

The default input file is:

```text
ai_detection_dataset.csv
```

The default model output is:

```text
random_forest_ai_detector_regularized.pkl
```

The script also creates:

```text
training_metrics.json
feature_integrity_report.csv
```

## Use a different dataset path

```bash
python train.py --data path/to/your_dataset.csv
```

## Quick test run

The full grid search tests many Random Forest settings and can take time.

Use this command when you only want to check that the code works:

```bash
python train.py --quick
```

Do not use the quick run as the final project result. Use the normal command for the final training run.

## Label meaning

The project uses:

```text
0 = Human
1 = AI
```

## Main features

The final model uses seven features calculated directly from the text:

- Word count
- Character count
- Sentence count
- Average sentence length
- Flesch Reading Ease
- Gunning Fog Index
- Punctuation ratio

The model does not use the supplied CSV feature values as final inputs.
