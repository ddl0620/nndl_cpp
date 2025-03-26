#include "math_lib.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_prime(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}