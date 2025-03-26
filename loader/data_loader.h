#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

std::vector<std::pair<std::vector<double>, std::vector<double>>> load_training_data();
std::vector<std::pair<std::vector<double>, std::vector<double>>> load_test_data();
std::vector<std::pair<std::vector<double>, std::vector<double>>> load_validation_data();

#endif