# diabetes_nn

Single-layer neural network in pure C that classifies diabetes risk
using the Pima Indians Diabetes dataset.

## Get the dataset

Download `diabetes.csv` from:
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Place it at:  data/diabetes.csv

The file should have 769 rows (1 header + 768 samples) and 9 columns:
  Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
  BMI, DiabetesPedigreeFunction, Age, Outcome

## Build & run

    make run

## Tune

Edit the three #define lines at the top of src/main.c:

    #define EPOCHS      1000    /* try 500 – 2000          */
    #define LR          0.01    /* try 0.001, 0.01, 0.1    */
    #define TEST_RATIO  0.20    /* fraction held out        */

## Expected output

    [data] Loaded 768 samples, 8 features
    [data] Train: 614 samples | Test: 154 samples

    --- Training  (lr=0.0100, epochs=1000, samples=614) ---
      Epoch  100 | loss: 0.5321 | acc: 75.2%
      ...
      Epoch 1000 | loss: 0.4712 | acc: 79.5%

    --- Final results ---
      Train  loss: 0.4712   acc: 79.5%
      Test   loss: 0.4889   acc: 77.3%

    --- Confusion matrix (test set) ---
                 Predicted 1  Predicted 0
      Actual 1        38 (TP)     17 (FN)
      Actual 0        18 (FP)     81 (TN)
      Precision: 0.68  |  Recall: 0.69

    --- Learned weights ---
      W[Pregnancies    ] = +0.1823
      W[Glucose        ] = +0.8741    <-- highest: most predictive feature
      W[Blood pressure ] = -0.0921
      W[Skin thickness ] = +0.0342
      W[Insulin        ] = -0.0215
      W[BMI            ] = +0.6103    <-- second highest
      W[Pedigree func. ] = +0.2891
      W[Age            ] = +0.3012

## Project structure

    diabetes_nn/
    ├── data/
    │   └── diabetes.csv        <-- you add this
    ├── src/
    │   ├── main.c              -- entry point & hyperparams
    │   ├── data.h / data.c     -- CSV loading, normalise, split
    │   ├── nn.h   / nn.c       -- forward, backward, train, eval
    └── Makefile

## Day 3 bonus challenges

- Add a hidden layer (true 2-layer net) — compare accuracy vs single layer
- Try mini-batch SGD (batch_size=32) instead of pure online SGD
- Implement L2 regularisation: add lambda * W[i] to the weight gradient
- Swap min-max normalisation for z-score and compare convergence speed
