#include "activation_functions.h"


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_der(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double ReLU(double x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

double ReLU_der(double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}
