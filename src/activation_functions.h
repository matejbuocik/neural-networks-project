#ifndef ACTIVATION_FUNS
#define ACTIVATION_FUNS

#include <math.h>
#include "matrices.h"


void sigmoid(const Matrix *in, const Matrix *out);

void sigmoid_der(const Matrix *in, const Matrix *out);

void ReLU(const Matrix *in, const Matrix *out);

void ReLU_der(const Matrix *in, const Matrix *out);

#endif
