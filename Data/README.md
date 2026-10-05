# Human vs Machine Datasets

This project uses two datasets:

1. `ai_detection_dataset.csv` - the main training dataset.
2. `human_vs_machine_testing_dataset_20.csv` - a small separate testing dataset.

The project uses the following labels:

- `0` = Human
- `1` = AI

---

## 1. Training Dataset

**File:** `ai_detection_dataset.csv`

This is the main dataset used to train and evaluate the Random Forest model.

### Size

- Total records: **1,367**
- Human samples: **684**
- AI samples: **683**
- Total columns: **17**

The classes are almost perfectly balanced.

### Main Columns

The dataset contains:

- `text_content` - the text sample
- `content_type` - type or category of the text
- `label` - Human or AI class

It also contains several supplied numerical and linguistic features:

- `word_count`
- `character_count`
- `sentence_count`
- `lexical_diversity`
- `avg_sentence_length`
- `avg_word_length`
- `punctuation_ratio`
- `flesch_reading_ease`
- `gunning_fog_index`
- `grammar_errors`
- `passive_voice_ratio`
- `predictability_score`
- `burstiness`
- `sentiment_score`

### How the Final Model Uses This Dataset

The final project does not directly trust all of the supplied numerical feature columns.

Instead, the model recalculates seven features directly from `text_content`:

1. Word count
2. Character count
3. Sentence count
4. Average sentence length
5. Flesch Reading Ease
6. Gunning Fog Index
7. Punctuation ratio

This was done because the feature-integrity analysis found that some supplied feature values did not closely match values calculated from the actual text.

The training script also checks for duplicate text and conflicting labels before training.

---

## 2. Testing Dataset

**File:** `human_vs_machine_testing_dataset_20.csv`

This is a small separate dataset created for testing the trained model.

### Size

- Total records: **20**
- Human-style samples: **10**
- AI-style samples: **10**
- Total columns: **3**

### Columns

- `sample_id` - unique number for each sample
- `text_content` - the text to classify
- `label` - expected class

### Purpose

This dataset can be used to check how the trained model behaves on text that was not part of the main training file.

It should not be added to the training dataset when it is being used as a test set.

The 20 samples are balanced so that both classes are represented equally.

### Important Limitation

The testing samples are synthetic style examples created for project and software testing.

They are useful for checking that the prediction pipeline works, but they should not be treated as a strong real-world benchmark of AI authorship detection.

A stronger final evaluation would use:

- real human-written samples from different authors;
- AI-generated samples from different AI systems;
- different topics and writing styles;
- samples that were not used during model development.

---

## Dataset Roles

| Dataset | Main Purpose | Records | Human | AI |
|---|---|---:|---:|---:|
| `ai_detection_dataset.csv` | Training and model development | 1,367 | 684 | 683 |
| `human_vs_machine_testing_dataset_20.csv` | Separate functional testing | 20 | 10 | 10 |

---

## Recommended Use

Train the model with:

```bash
python train.py --data ai_detection_dataset.csv
```

After training, use the separate testing dataset to evaluate predictions without retraining the model on those samples.

Do not combine the testing dataset with the training dataset before evaluation, because this would weaken the independence of the test.
