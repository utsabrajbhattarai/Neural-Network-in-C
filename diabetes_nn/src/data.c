#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "data.h"

int load_csv(const char *path, Dataset *ds)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return -1;
    }

    char line[512];
    ds->n_samples = 0;

    /* Skip the header row */
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }

    while (fgets(line, sizeof(line), f) && ds->n_samples < MAX_SAMPLES) {
        char *tok = strtok(line, ",\n\r");
        int col = 0;
        while (tok && col < N_FEATURES + 1) {
            if (col < N_FEATURES)
                ds->X[ds->n_samples][col] = atof(tok);
            else
                ds->y[ds->n_samples] = atof(tok);
            col++;
            tok = strtok(NULL, ",\n\r");
        }
        if (col == N_FEATURES + 1)
            ds->n_samples++;
    }
    fclose(f);
    printf("[data] Loaded %d samples, %d features\n", ds->n_samples, N_FEATURES);
    return 0;
}

void normalize_minmax(Dataset *ds)
{
    for (int j = 0; j < N_FEATURES; j++) {
        double mn = ds->X[0][j], mx = ds->X[0][j];
        for (int i = 1; i < ds->n_samples; i++) {
            if (ds->X[i][j] < mn) mn = ds->X[i][j];
            if (ds->X[i][j] > mx) mx = ds->X[i][j];
        }
        double range = (mx - mn < 1e-9) ? 1.0 : mx - mn;
        for (int i = 0; i < ds->n_samples; i++)
            ds->X[i][j] = (ds->X[i][j] - mn) / range;
    }
}

void shuffle_dataset(Dataset *ds)
{
    /* Fisher-Yates shuffle — call srand once before training, not here,
       so callers control reproducibility via main(). */
    for (int i = ds->n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        /* Swap feature rows */
        double tmp_x[N_FEATURES];
        memcpy(tmp_x,      ds->X[i], N_FEATURES * sizeof(double));
        memcpy(ds->X[i],   ds->X[j], N_FEATURES * sizeof(double));
        memcpy(ds->X[j],   tmp_x,    N_FEATURES * sizeof(double));
        /* Swap labels */
        double tmp_y  = ds->y[i];
        ds->y[i]      = ds->y[j];
        ds->y[j]      = tmp_y;
    }
}

void train_test_split(Dataset *full, Dataset *train, Dataset *test,
                      double test_ratio)
{
    int n_test  = (int)(full->n_samples * test_ratio);
    int n_train = full->n_samples - n_test;

    train->n_samples = n_train;
    test->n_samples  = n_test;

    for (int i = 0; i < n_train; i++) {
        memcpy(train->X[i], full->X[i], N_FEATURES * sizeof(double));
        train->y[i] = full->y[i];
    }
    for (int i = 0; i < n_test; i++) {
        memcpy(test->X[i], full->X[n_train + i], N_FEATURES * sizeof(double));
        test->y[i] = full->y[n_train + i];
    }
    printf("[data] Train: %d samples | Test: %d samples\n",
           n_train, n_test);
}
