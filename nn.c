#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nn.h"

/* ------------------------------------------------------------------ helpers */

static double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

/* ------------------------------------------------------------------ public API */

void nn_init(NeuralNet *net, double lr)
{
    net->lr = lr;
    net->b  = 0.0;
    /*
     * Xavier-style init: scale by 1/sqrt(N_FEATURES).
     * Small random weights break symmetry; zero init would keep all
     * weights identical forever (no symmetry breaking to worry about
     * in a single-layer net, but good habit).
     */
    double scale = 1.0 / sqrt((double)N_FEATURES);
    for (int i = 0; i < N_FEATURES; i++)
        net->W[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale;
}

double nn_forward(NeuralNet *net, double x[N_FEATURES])
{
    double z = net->b;
    for (int i = 0; i < N_FEATURES; i++)
        z += net->W[i] * x[i];
    return sigmoid(z);
}

void nn_backward(NeuralNet *net, double x[N_FEATURES],
                 double y_hat, double y_true)
{
    /*
     * dL/dz = y_hat - y_true   (BCE loss + sigmoid — derivation in nn.h)
     * Then dL/dW[i] = dz * x[i],   dL/db = dz
     */
    double dz = y_hat - y_true;
    for (int i = 0; i < N_FEATURES; i++)
        net->W[i] -= net->lr * dz * x[i];
    net->b -= net->lr * dz;
}

void nn_train(NeuralNet *net, Dataset *ds, int epochs)
{
    printf("\n--- Training  (lr=%.4f, epochs=%d, samples=%d) ---\n",
           net->lr, epochs, ds->n_samples);

    for (int e = 0; e < epochs; e++) {
        /* Online (stochastic) gradient descent: one update per sample */
        for (int i = 0; i < ds->n_samples; i++) {
            double y_hat = nn_forward(net, ds->X[i]);
            nn_backward(net, ds->X[i], y_hat, ds->y[i]);
        }

        if ((e + 1) % 100 == 0) {
            printf("  Epoch %4d | loss: %.4f | acc: %.1f%%\n",
                   e + 1,
                   nn_loss(net, ds),
                   nn_accuracy(net, ds) * 100.0);
        }
    }
}

double nn_accuracy(NeuralNet *net, Dataset *ds)
{
    int correct = 0;
    for (int i = 0; i < ds->n_samples; i++) {
        int pred = (nn_forward(net, ds->X[i]) >= 0.5) ? 1 : 0;
        if (pred == (int)ds->y[i])
            correct++;
    }
    return (double)correct / ds->n_samples;
}

double nn_loss(NeuralNet *net, Dataset *ds)
{
    double total = 0.0;
    for (int i = 0; i < ds->n_samples; i++) {
        double yh = nn_forward(net, ds->X[i]);
        double y  = ds->y[i];
        /* Clip to avoid log(0) */
        total -= y * log(yh + 1e-15) + (1.0 - y) * log(1.0 - yh + 1e-15);
    }
    return total / ds->n_samples;
}
