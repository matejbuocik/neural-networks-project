#ifndef ACTIVATION_FUNS
#define ACTIVATION_FUNS

#include <math.h>
#include "matrices.h"


int get_random_int(int min, int max);

double mse(Matrix* mat1, Matrix* mat2);

double generate_normal_random(double mean, double variance);

double generate_uniform(double min_val, double max_val);

void sigmoid(const Matrix *in, const Matrix *out);

void sigmoid_der(const Matrix *in, const Matrix *out);

void ReLU(const Matrix *in, const Matrix *out);

void ReLU_der(const Matrix *in, const Matrix *out);

void softmax(const Matrix *in, const Matrix *out);

void softmax_der(const Matrix *in, const Matrix *out);

#endif
