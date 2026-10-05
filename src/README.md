# Human vs Machine Python Project

This folder contains the Python version of the notebook project.

## Files

- `train.py` - trains and evaluates the Random Forest model.
- `predict.py` - makes predictions from the command line.
- `app.py` - starts the Gradio dashboard.
- `text_features.py` - contains the shared seven-feature extraction code.
- `README_train.md` - instructions for training.
- `README_predict.md` - instructions for predictions.
- `README_app.md` - instructions for the dashboard.
- `requirements.txt` - Python package list.

## Basic order

First install the packages:

```bash
pip install -r requirements.txt
```

Then train the model:

```bash
python train.py
```

Then use either:

```bash
python predict.py --text "Text to test"
```

or:

```bash
python app.py
```

The training CSV should contain:

```text
text_content
label
```

The project uses:

```text
0 = Human
1 = AI
```
