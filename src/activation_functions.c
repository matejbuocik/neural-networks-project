#include "activation_functions.h"
#include <assert.h>


int get_random_int(int min, int max, unsigned int *state) {
    int range = max - min + 1;  // include min and max
    int random_int = min + rand_r(state) % range;

    return random_int;
}

double mse(Matrix* mat1, Matrix* mat2) {
    Matrix *sub = sub_mat(mat1, mat2);
    apply_func_mat(sub, sub, fabs, false);

    double sum = sum_mat(sub);
    free_mat(sub);

    return sum;
}

double generate_normal_random(double mean, double variance) {
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#References
    double u1 = ((double)rand() / RAND_MAX);
    double u2 = ((double)rand() / RAND_MAX);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return mean + sqrt(variance) * z0;
}

double generate_uniform(double min_val, double max_val) {
    return ((double)rand() / RAND_MAX) * (max_val - min_val) + min_val;
}

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
    printf("Not implemented: args[%p, %p]\n", in, out);
}
