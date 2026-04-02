#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data.h"
#include "nn.h"

static const char *FEATURE_NAMES[N_FEATURES] = {
    "Pregnancies       (e.g. 2)",
    "Glucose           (e.g. 120)",
    "Blood pressure    (e.g. 70)",
    "Skin thickness    (e.g. 30)",
    "Insulin           (e.g. 80)",
    "BMI               (e.g. 32.0)",
    "Diabetes pedigree (e.g. 0.4)",
    "Age               (e.g. 25)"
};

int main()
{
    NeuralNet net;
    double mins[N_FEATURES], maxs[N_FEATURES];

    nn_load(&net, "model_weights.bin", mins, maxs);

    /* input loop */
    double x[N_FEATURES];
    for (int i = 0; i < N_FEATURES; i++) {
        printf("  %s: ", FEATURE_NAMES[i]);
        scanf("%lf", &x[i]);
    }

    /* normalize using saved min/max — no dataset needed */
    for (int j = 0; j < N_FEATURES; j++) {
        double range = (maxs[j] - mins[j] < 1e-9) ? 1.0 : maxs[j] - mins[j];
        x[j] = (x[j] - mins[j]) / range;
    }

    double prob = nn_forward(&net, x);
    printf("Diabetes probability: %.1f%%\n", prob * 100.0);
    printf("Prediction: %s\n", prob >= 0.5 ? "DIABETIC" : "NOT DIABETIC");
    return 0;
}