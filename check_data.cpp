#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "loader/data_loader.h"

using namespace std;

void test_read_raw(const string& file_name) {
    ifstream file(file_name);

    if (!file) {
        cerr << "Error: File not found" << endl;
        exit(1);
    }
    
    string line;
    if (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> image;
        int label;
        
        getline(ss, value, ',');
        label = stoi(value);

        while(getline(ss, value, ',')) {
            image.push_back(stod(value));
        }

        if (image.size() != 784) {
            cerr << "Invalid image!" << endl;
            exit(2);
        }

        cout << "Label: " << label << endl;

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                cout << (image[i * 28 + j] > 0.5 ? '#' : '.');
            }
            cout << endl;
        }
    }
    else {
        cerr << "Empty csv file!" << endl;
        exit(1);
    }
}

void test_data_processed() {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> training_dataset = load_validation_data();
    std::pair<std::vector<double>, std::vector<double>> sample_pair = training_dataset[0];
    std::vector<double> input = sample_pair.first;
    std::vector<double> output = sample_pair.second;

    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }

    std::cout << std::endl;

    int cnt = 1;
    for (int i = 0; i < 784; i++) {
        if (i == 28 * cnt) {
            cnt++;
            std::cout << std::endl;
            std::cout << (input[i] > 0.5 ? "#" : ".");
        }
        else {
            std::cout << (input[i] > 0.5 ? "#" : ".");
        }
    }
    std::cout << std::endl;
}


int main() {
    string file_name = "data/mnist_train.csv";
    test_read_raw(file_name);
    return 0;
}