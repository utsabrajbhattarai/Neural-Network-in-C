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
typedef struct {
    double W[N_FEATURES]; /* one weight per input feature */
    double b;             /* bias term */
    double lr;            /* learning rate */
} NeuralNet;

void   nn_init    (NeuralNet *net, double lr);
double nn_forward (NeuralNet *net, double x[N_FEATURES]);
void   nn_backward(NeuralNet *net, double x[N_FEATURES],
                   double y_hat, double y_true);
void   nn_train   (NeuralNet *net, Dataset *ds, int epochs);
double nn_accuracy(NeuralNet *net, Dataset *ds);
double nn_loss    (NeuralNet *net, Dataset *ds);

#endif /* NN_H */
