#ifndef NN_H
#define NN_H

#include "data.h"

/*
 * Single-layer neural network (a.k.a. logistic regression with backprop).
 *
 * Forward:   z    = W · x + b
 *            y_hat = sigmoid(z)
 *
 * Loss:      L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]   (Binary Cross-Entropy)
 *
 * Backward:  dL/dz = ŷ − y   ← the beautiful BCE+sigmoid shortcut
 *            dL/dW = dz · x
 *            dL/db = dz
 *
 * Update:    W -= lr * dL/dW
 *            b -= lr * dL/db
 */

#define HIDDEN 4 // no. of neurons in the hidden layer

/* Clip gradients to prevent explosion */
#define CLIP 1.0


typedef struct {
    // hidden layer
    double W1[HIDDEN][N_FEATURES];
    double b1[HIDDEN];
    // output layer
    double W2[HIDDEN];
    double b2;
    double z1[HIDDEN];   /* pre-activation hidden layer (for backprop) */
    double hidden[HIDDEN]; /* cache for backprop */
    double lr;            /* learning rate */
} NeuralNet;

void   nn_init    (NeuralNet *net, double lr);
double nn_forward (NeuralNet *net, double x[N_FEATURES]);
void   nn_backward(NeuralNet *net, double x[N_FEATURES],
                   double y_hat, double y_true);
void   nn_train   (NeuralNet *net, Dataset *ds, int epochs);
double nn_accuracy(NeuralNet *net, Dataset *ds);
double nn_loss    (NeuralNet *net, Dataset *ds);
void nn_save(NeuralNet *net, const char *path,
             double mins[N_FEATURES], double maxs[N_FEATURES]);
void nn_load(NeuralNet *net, const char *path,
             double mins[N_FEATURES], double maxs[N_FEATURES]);

#endif /* NN_H */
