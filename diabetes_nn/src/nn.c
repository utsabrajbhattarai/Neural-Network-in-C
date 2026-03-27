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

static double relu(double z)
{
    return z > 0.0 ? z : 0.01 * z;  // leaky — never fully dies
}

static double relu_deriv(double z)
{
    return z > 0.0 ? 1.0 : 0.01;    // always small gradient, never zero
}

/* ------------------------------------------------------------------ public API */

void nn_init(NeuralNet *net, double lr)
{
    net->lr = lr;
    net->b2  = 0.0;
    /*
     * Xavier-style init: scale by 1/sqrt(N_FEATURES).
     * Small random weights break symmetry; zero init would keep all
     * weights identical forever.
     */
    double scale1 = sqrt(2.0 / (double)N_FEATURES);  // He init for ReLU
    double scale2 = sqrt(2.0 / (double)HIDDEN);  // He init for output layer (even though it's sigmoid, not ReLU, this is a common choice)

    for (int i = 0; i < HIDDEN; i++)
        {
            net->b1[i] = 0.0;
            net->W2[i] = scale2 * 2.0 * ((double)rand() / RAND_MAX - 0.5);
            for (int j = 0; j < N_FEATURES; j++)
                net->W1[i][j] = scale1 * 2.0 * ((double)rand() / RAND_MAX - 0.5);
        }
}

double nn_forward(NeuralNet *net, double x[N_FEATURES])
{ 
    /* Stage 1: input -> hidden (ReLU) */
    for (int j = 0; j < HIDDEN; j++) {
        double z = net->b1[j];
        for (int k = 0; k < N_FEATURES; k++)
            z += net->W1[j][k] * x[k];
        net->hidden[j] = relu(z);
        net->z1[j] = z;   /* save pre-activation for finding derivatives later in backprop */
    }

    /* Stage 2: hidden -> output (sigmoid) */
    double z2 = net->b2;
    for (int j = 0; j < HIDDEN; j++)
        z2 += net->W2[j] * net->hidden[j];
    return sigmoid(z2);
}

void nn_backward(NeuralNet *net, double x[N_FEATURES],
                 double y_hat, double y_true)
{
    /*
    * OUTPUT LAYER GRADIENT: (Think of L likke a cost function we want to minimize)
    * dL/dz2 = y_hat - y_true        (derived output - actual output)
    * dL/dW2[j] = dz2 * hidden[j]   (hidden is the input neuron to this layer)
    * dL/db2    = dz2
    *
    * HIDDEN LAYER GRADIENT (chain rule):
    * dL/dhidden[j] = dz2 * W2[j]           (error flowing back through W2)
    * dL/dz1[j]     = dh * relu'(z1[j])     (relu' = 1 if z1>0, else 0)
    * dL/dW1[j][k]  = dz1 * x[k]            (x is the input to this layer)
    * dL/db1[j]     = dz1
    */

    /* Output gradient */
    double dz2 = y_hat - y_true;
    /* Clip gradients to prevent explosion */
    dz2 = dz2 > CLIP ? CLIP : (dz2 < -CLIP ? -CLIP : dz2);

    /* Update W2 and b2 */
    for (int j = 0; j < HIDDEN; j++)
        net->W2[j] -= net->lr * dz2 * net->hidden[j];
    net->b2 -= net->lr * dz2;

    /* Hidden layer gradient */
    for (int j = 0; j < HIDDEN; j++) {
        double dh  = dz2 * net->W2[j];
        double dz1 = dh * relu_deriv(net->z1[j]);
        dz1 = dz1 > CLIP ? CLIP : (dz1 < -CLIP ? -CLIP : dz1);


        for (int k = 0; k < N_FEATURES; k++)
            net->W1[j][k] -= net->lr * dz1 * x[k];
        net->b1[j] -= net->lr * dz1;
    }
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
