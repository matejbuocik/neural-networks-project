#include "activation_functions.h"


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
        return 0;
    }
}

void ReLU(const Matrix *in, const Matrix *out) {
    apply_func_mat(in, out, ReLU_scal, false);
}

double ReLU_der_scal(double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}


void ReLU_der(const Matrix *in, const Matrix *out) {
    apply_func_mat(in, out, ReLU_der_scal, true);
}