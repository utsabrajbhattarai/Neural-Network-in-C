#ifndef DATA_H
#define DATA_H

#define MAX_SAMPLES 800
#define N_FEATURES  8

/* All 768 samples + labels live in flat static arrays — no malloc needed. */
typedef struct {
    double X[MAX_SAMPLES][N_FEATURES];
    double y[MAX_SAMPLES];          /* 0.0 = no diabetes, 1.0 = diabetes */
    int    n_samples;
} Dataset;

/* Load diabetes.csv (expects header row, then 8 feature cols + 1 label col). */
int  load_csv(const char *path, Dataset *ds);

/* Min-max normalise each feature column to [0, 1]. Must call before split. */
void normalize_minmax(Dataset *ds,
                      double mins[N_FEATURES], double maxs[N_FEATURES]);

/* Fisher-Yates in-place shuffle. Call before split for unbiased test set. */
void shuffle_dataset(Dataset *ds);

/* Copy first (1-test_ratio) fraction into *train, remainder into *test. */
void train_test_split(Dataset *full, Dataset *train, Dataset *test,
                      double test_ratio);

#endif /* DATA_H */
