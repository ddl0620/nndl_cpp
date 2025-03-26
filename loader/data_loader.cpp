#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

std::vector<std::pair<std::vector<double>, std::vector<double>>> load_training_data() {
    const std::string filename = "data/mnist_train.csv";
    std::vector<std::pair<std::vector<double>, std::vector<double>>> training_dataset;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> input(784);
        std::vector<double> output(10, 0.0);
        int label;
        char comma;

        ss >> label;
        output[label] = 1.0;

        for (int i = 0; i < 784; i++) {
            double pixel;
            ss >> comma >> pixel;
            input[i] = pixel;
        }

        training_dataset.emplace_back(input, output);
    }

    file.close();
    return training_dataset;

}


std::vector<std::pair<std::vector<double>, std::vector<double>>> load_test_data() {
    const std::string filename = "data/mnist_test.csv";
    std::vector<std::pair<std::vector<double>, std::vector<double>>> test_dataset;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> input(784);
        std::vector<double> output(10, 0.0);
        int label;
        char comma;

        ss >> label;
        output[label] = 1.0;

        for (int i = 0; i < 784; i++) {
            double pixel;
            ss >> comma >> pixel;
            input[i] = pixel;
        }

        test_dataset.emplace_back(input, output);
    }

    file.close();
    return test_dataset;
}

std::vector<std::pair<std::vector<double>, std::vector<double>>> load_validation_data() {
    const std::string filename = "data/mnist_valid.csv";
    std::vector<std::pair<std::vector<double>, std::vector<double>>> validation_dataset;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> input(784);
        std::vector<double> output(10, 0.0);
        int label;
        char comma;

        ss >> label;
        output[label] = 1.0;

        for (int i = 0; i < 784; i++) {
            double pixel;
            ss >> comma >> pixel;
            input[i] = pixel;
        }

        validation_dataset.emplace_back(input, output);
    }

    file.close();
    return validation_dataset;


}
