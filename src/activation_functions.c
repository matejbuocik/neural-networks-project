#include "activation_functions.h"
#include <assert.h>


double sigmoid_scal(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void sigmoid(const Matrix *in, const Matrix *out) {
    apply_func_mat(in, out, sigmoid_scal, false);
}

double sigmoid_der_scal(double x) {
    double s = sigmoid_scal(x);
    return s * (1.0 - s);
}

void sigmoid_der(const Matrix *in, const Matrix *out) {
    apply_func_mat(in, out, sigmoid_der_scal, true);
}

double ReLU_scal(double x) {
    if (x > 0) {
        return x;
    } else {
        return 0.0;
    }
}

void ReLU(const Matrix *in, const Matrix *out) {
    apply_func_mat(in, out, ReLU_scal, false);
}

double ReLU_der_scal(double x) {
    if (x > 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}


void ReLU_der(const Matrix *in, const Matrix *out) {
    apply_func_mat(in, out, ReLU_der_scal, true);
}

void softmax(const Matrix *in, const Matrix *out) {
    assert(in->cols == out->cols && in->rows == out->rows);

    double suma = 0.0;
    for (int i = 0; i < in->cols; i++) {
        suma += exp(in->data[0][i]);
    }

    for (int i = 0; i < out->cols; i++) {
        out->data[0][i] = exp(in->data[0][i]) / suma;
    }
}

void softmax_der(const Matrix *in, const Matrix *out) {

}
