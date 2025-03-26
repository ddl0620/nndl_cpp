#include <iostream>
#include <string>
#include <vector>
#include "math_lib/math_lib.h"
#include "loader/data_loader.h"
#include "network/network.h"

int main() {
    std::vector<int>sizes = {784, 30, 10};
    Network net(sizes);
    auto training_data = load_training_data();
    auto test_data = load_test_data();
    net.SGD(training_data, test_data, 30, 10, 3.0);
}
    