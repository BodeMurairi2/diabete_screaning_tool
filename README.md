# Diabetes Screening Tool

A machine learning model that screens adults for diabetes from simple health and lifestyle questions — no blood test needed.

## Data

- **Source:** [CDC Behavioral Risk Factor Surveillance System (BRFSS) 2023](https://www.cdc.gov/brfss/annual_data/annual_2023.html)
- **Size:** 433,323 survey respondents (432,339 after removing unknown targets)
- **Codebook:** [codebook23_llcp-v2-508](https://www.cdc.gov/brfss/annual_data/2023/zip/codebook23_llcp-v2-508.zip)

The dataset is downloaded automatically by the notebook and is not stored in this repository.

## Target

`diabetes` (BRFSS `DIABETE4`) — multiclass:

| Value | Meaning |
|---|---|
| 1 | Diabetes |
| 3 | No diabetes |
| 4 | At risk: prediabetes, or diabetes only during pregnancy (gestational) |

Respondents with diabetes only during pregnancy (original code 2) are grouped with prediabetes: they do not currently have diabetes, but gestational diabetes is a strong risk factor for type 2 diabetes.

## Features

| Type | Features |
|---|---|
| Numeric | bmi, age, height, weight, poor physical/mental health days, drinks per week, drink occasions per day, exercise minutes, vigorous exercise minutes, strength training per week |
| Ordinal | general health, age group, education, income, smoking status, cholesterol check |
| Binary | high blood pressure, high cholesterol, stroke, heart disease, kidney disease, physical activity, heavy drinker, health insurance, can't afford doctor, difficulty walking, sex |

## Project structure

```
├── analysis.ipynb           # download, cleaning, encoding, training, evaluation
├── data/                    # raw XPT + train/val/test splits (not tracked)
├── runs/                    # TensorBoard logs (not tracked)
└── README.md
```

## Getting started

```bash
pip install numpy pandas matplotlib scikit-learn tensorboard tensorboardX pyarrow
jupyter notebook analysis.ipynb
```

Run the notebook top to bottom, locally or on Kaggle (turn on **Internet** in the Kaggle notebook settings). It will:

1. Download and extract the BRFSS 2023 data into `data/`
2. Select and rename the features
3. Replace BRFSS missing codes (7, 9, 77, 99, …) with NaN and rescale numeric values
4. Encode binary and ordinal features
5. Split the data 60 / 20 / 20 into train, validation, and test (stratified)
6. Train and evaluate the models

## Viewing experiments

Every run is logged to TensorBoard:

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006.

## Results

Baseline: logistic regression (`class_weight="balanced"`), validation set.

| Metric | Score |
|---|---|
| ROC-AUC (one-vs-rest, macro) | 0.772 |
| Balanced accuracy | 0.514 |
| Macro F1 | 0.289 |
| Diabetes recall | 0.605 |

## Disclaimer

This tool is for educational and research purposes only. It does not replace a medical diagnosis — anyone concerned about diabetes should see a healthcare professional and get a blood test.
