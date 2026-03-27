#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data.h"
#include "nn.h"

/* hyperparameters*/
#define EPOCHS      3000
#define LR          0.001
#define TEST_RATIO  0.20

static const char *FEATURE_NAMES[N_FEATURES] = {
    "Pregnancies",
    "Glucose",
    "Blood pressure",
    "Skin thickness",
    "Insulin",
    "BMI",
    "Pedigree func.",
    "Age"
};

/* Print a simple confusion matrix from test predictions. */
static void print_confusion(NeuralNet *net, Dataset *ds)
{
    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (int i = 0; i < ds->n_samples; i++) {
        int pred  = (nn_forward(net, ds->X[i]) >= 0.5) ? 1 : 0;
        int truth = (int)ds->y[i];
        if      (pred == 1 && truth == 1) tp++;
        else if (pred == 1 && truth == 0) fp++;
        else if (pred == 0 && truth == 0) tn++;
        else                              fn++;
    }
    printf("\n--- Confusion matrix (test set) ---\n");
    printf("             Predicted 1  Predicted 0\n");
    printf("  Actual 1      %4d (TP)   %4d (FN)\n", tp, fn);
    printf("  Actual 0      %4d (FP)   %4d (TN)\n", fp, tn);
    double precision = (tp + fp) ? (double)tp / (tp + fp) : 0.0;
    double recall    = (tp + fn) ? (double)tp / (tp + fn) : 0.0;
    printf("  Precision: %.2f  |  Recall: %.2f\n", precision, recall);
}

static void print_f1(NeuralNet *net, Dataset *ds)
{
    int tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < ds->n_samples; i++) {
        int pred  = (nn_forward(net, ds->X[i]) >= 0.5) ? 1 : 0;
        int truth = (int)ds->y[i];
        if      (pred == 1 && truth == 1) tp++;
        else if (pred == 1 && truth == 0) fp++;
        else if (pred == 0 && truth == 1) fn++;
    }
    double precision = (tp + fp) ? (double)tp / (tp + fp) : 0.0;
    double recall    = (tp + fn) ? (double)tp / (tp + fn) : 0.0;
    double f1        = (precision + recall > 0)
                       ? 2.0 * precision * recall / (precision + recall)
                       : 0.0;
    printf("  F1 Score:  %.4f\n", f1);
}

int main(void)
{
    srand((unsigned)time(NULL));   /* seed once here for reproducibility */


    /* 1. Load */
    static Dataset full;           /* static: keeps large arrays off the stack */
    if (load_csv("data/diabetes.csv", &full) != 0) {
        fprintf(stderr, "Could not load data/diabetes.csv\n");
        return 1;
    }


    /* 2. Pre-process */
    double mins[N_FEATURES], maxs[N_FEATURES];
    normalize_minmax(&full, mins, maxs);        /* [0,1] per feature */
    shuffle_dataset(&full);                      /* randomise before split */


    /* 3. Split */
    static Dataset train, test;
    train_test_split(&full, &train, &test, TEST_RATIO);


    /* 4. Train */
    NeuralNet net;
    nn_init(&net, LR);
    nn_train(&net, &train, EPOCHS);


    /* 5. Evaluate */
    printf("\n--- Final results ---\n");
    printf("  Train  loss: %.4f   acc: %.1f%%\n",
           nn_loss(&net, &train), nn_accuracy(&net, &train) * 100.0);
    printf("  Test   loss: %.4f   acc: %.1f%%\n",
           nn_loss(&net, &test),  nn_accuracy(&net, &test)  * 100.0);

    print_confusion(&net, &test);
    print_f1(&net, &test);


    /* 6. Inspect learned weights (feature importance proxy) */
    /* Row-wise L2 norm = overall importance of each feature across all hidden neurons */
    printf("\n--- Feature importance (L2 norm across hidden layer) ---\n");
    for (int k = 0; k < N_FEATURES; k++) {
        double norm = 0.0;
        for (int j = 0; j < HIDDEN; j++)
            norm += net.W1[j][k] * net.W1[j][k];
        norm = sqrt(norm);
        printf("  %-16s  %.4f\n", FEATURE_NAMES[k], norm);
    }


    /* 7. Save the trained model weights to a file */
    nn_save(&net, "model_weights_best.bin", mins, maxs);
                                                     // model_weights.bin is a binary file that will contain the trained weights of the neural network. This allows you to save the state of the model after training, and later load it back for inference or further training without having to retrain from scratch.
                                                     // model_weights.bin is the most dynamic outcome and model_weights_best.bin is the outcome that had 84% accuracy.

    return 0;
}
