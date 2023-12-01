#ifndef ACTIVATION_FUNS
#define ACTIVATION_FUNS

#include <math.h>
#include "matrices.h"


/* Get random integer between `min` and `max`, including both.
   Use `state` in rand_r function. */
int get_random_int(int min, int max, unsigned int *state);

/* Compute mean squared error */
double mse(Matrix* mat1, Matrix* mat2);

/* Generate number from normal random distribution */
double generate_normal_random(double mean, double variance);

/* Generate number from uniform random distribution */
double generate_uniform(double min_val, double max_val);

/* Apply sigmoid activation function */
void sigmoid(const Matrix *in, const Matrix *out);

/* Apply derivation of sigmoid function */
void sigmoid_der(const Matrix *in, const Matrix *out);

/* Apply ReLU activation function */
void ReLU(const Matrix *in, const Matrix *out);

/* Apply derivation of ReLU*/
void ReLU_der(const Matrix *in, const Matrix *out);

/* Apply softmax activation function */
void softmax(const Matrix *in, const Matrix *out);

/* Apply derivation of softmax */
void softmax_der(const Matrix *in, const Matrix *out);

#endif
