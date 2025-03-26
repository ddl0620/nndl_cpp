#include "network.h"
#include "../math_lib/math_lib.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <ctime>


Network::Network(std::vector<int> sizes) : sizes(sizes) {
    num_layers = sizes.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    biases.resize(num_layers - 1);
    for (int i = 1; i < num_layers; i++) {
        biases[i - 1].resize(sizes[i]);
        for (double &b : biases[i - 1]) {
            b = dist(gen);
        }
    }

    weights.resize(num_layers - 1);
    for (int i = 0; i < num_layers - 1; i++) {
        weights[i].resize(sizes[i + 1], std::vector<double>(sizes[i]));
        for (int j = 0; j < sizes[i + 1]; j++) {
            for (int k = 0; k < sizes[i]; k++) {
                weights[i][j][k] = dist(gen);
            }
        }
    }

}

std::vector<double> Network::feedForward(const std::vector<double>& input) {
    std::vector<double> activation = input;
    for (size_t layer = 0; layer < weights.size(); layer++) {
        std::vector<double> z((size_t) weights[layer].size());
        for (size_t j = 0; j < (size_t) weights[layer].size(); j++) {
            double sum = biases[layer][j];
            for (size_t k = 0; k < weights[layer][j].size(); k++) {
                sum += weights[layer][j][k] * activation[k];
            }
            z[j] = sigmoid(sum);
        }
        activation = z;
    }
    return activation;
}

void Network::SGD(std::vector<std::pair<std::vector<double>, std::vector<double>>> &training_data, std::vector<std::pair<std::vector<double>, std::vector<double>>> &test_data,int epochs, int mini_batch_size, double eta)
{
    int n = training_data.size();
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(training_data.begin(), training_data.end(), gen);
        for (size_t k = 0; k < n; k += mini_batch_size) {
            std::vector<std::pair<std::vector<double>, std::vector<double>>> mini_batch(training_data.begin() + k, 
            training_data.begin() + std::min(n, static_cast<int>(k + mini_batch_size)));
            update_mini_batch(mini_batch, eta);
        }
        std::cout << "Epoch " << epoch + 1 << " complete.\n";

        int n_test = test_data.size();
        std::cout << evaluate(test_data) << "/" << n_test << std::endl;
        
    }

}

void Network::update_mini_batch(std::vector<std::pair<std::vector<double>, std::vector<double>>> &mini_batch, double eta) {
    std::vector<std::vector<double>> nabla_b = biases;
    std::vector<std::vector<std::vector<double>>> nabla_w = weights;

    for (size_t i = 0; i < biases.size(); i++) {
        std::fill(nabla_b[i].begin(), nabla_b[i].end(), 0);
    }
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights[i].size(); j++) {
            std::fill(nabla_w[i][j].begin(), nabla_w[i][j].end(), 0);
        }
    }

    for (auto &data : mini_batch) {
        auto [delta_nabla_b, delta_nabla_w] = backprop(data.first, data.second);

        for (size_t i = 0; i < nabla_b.size(); i++) {
            for (size_t j = 0; j < nabla_b[i].size(); j++) {
                nabla_b[i][j] += delta_nabla_b[i][j];
            }
        }
        for (size_t i = 0; i < nabla_w.size(); i++) {
            for (size_t j = 0; j < nabla_w[i].size(); j++) {
                for (size_t k = 0; k < nabla_w[i][j].size(); k++) {
                    nabla_w[i][j][k] += delta_nabla_w[i][j][k];
                }
            }
        }
    }

    double learning_factor = eta / mini_batch.size();
    for (size_t i = 0; i < biases.size(); i++) {
        for (size_t j = 0; j < biases[i].size(); j++) {
            biases[i][j] -= learning_factor * nabla_b[i][j];
        }
    }
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights[i].size(); j++) {
            for (size_t k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] -= learning_factor * nabla_w[i][j][k];
            }
        }
    }
    
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> Network::backprop(const std::vector<double> &x, const std::vector<double> &y)
{
    // Initialize gradients with correct sizes
    std::vector<std::vector<double>> nabla_b(biases.size());
    std::vector<std::vector<std::vector<double>>> nabla_w(weights.size());
    for (size_t i = 0; i < biases.size(); i++) {
        nabla_b[i] = std::vector<double>(biases[i].size(), 0.0);
    }
    for (size_t i = 0; i < weights.size(); i++) {
        nabla_w[i] = std::vector<std::vector<double>>(weights[i].size(), std::vector<double>(weights[i][0].size(), 0.0));
    }

    for (size_t i = 0; i < biases.size(); i++) {
        std::fill(nabla_b[i].begin(), nabla_b[i].end(), 0);
    }
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights[i].size(); j++) {
            std::fill(nabla_w[i][j].begin(), nabla_w[i][j].end(), 0);
        }
    }

    std::vector<std::vector<double>> activations = {x};
    std::vector<std::vector<double>> zs;

    //forward pass. store vectors of weighted inputs (2D) and vector of activations (2D)
    std::vector<double> activation = x;
    for (size_t layer = 0; layer < weights.size(); layer++) {
        std::vector<double> z(weights[layer].size());
        for (size_t j = 0; j < weights[layer].size(); j++) {
            double sum = biases[layer][j];
            for (size_t k = 0; k < weights[layer][j].size(); k++) {
                sum += weights[layer][j][k] * activation[k];
            }
            z[j] = sum;
        }
        zs.push_back(z);
        for (double& val : z) {
            val = sigmoid(val);
        }
        activations.push_back(z);
        activation = z;
    }

    //backward pass: 
    //1. Calculate the error delta in the outermost layer L

    /*last layer*/
    std::vector<double> delta(activations.back().size());
    for (size_t i = 0; i < delta.size(); i++) {
        //Backpropagation equation #1: partial derivative of cost fn w.r.t activations in final layer 
        //multiply by sigmoid prime of weighted input in the final laayer
        delta[i] = (activations.back()[i] - y[i]) * sigmoid_prime(zs.back()[i]);
    }

    //Backpropagation equation #3, applied for the last layer
    nabla_b.back() = delta;

    //Backpropagation equation #4, applied for last layer
    for (size_t i = 0; i < weights.back().size(); i++) {
        for (size_t j = 0; j < weights.back()[i].size(); j++) {
            nabla_w.back()[i][j] = delta[i] * activations[activations.size() - 2][j];
        }
    }

    /*backward pass to prev layers*/
    for (int l = weights.size() - 2; l >= 0; l--) {
        
        //calculate delta layer l using BP2

        //sigmoid prime vector of layer l for BP2
        std::vector<double> sp(zs[l].size());
        for (size_t i = 0; i < sp.size(); i++) {
            sp[i] = sigmoid_prime(zs[l][i]);
        }

        std::vector<double> new_delta(weights[l].size(), 0.0);
        for (size_t j = 0; j < weights[l].size(); j++) {
            for (size_t i = 0; i < weights[l + 1].size(); i++) {
                new_delta[j] += weights[l + 1][i][j] * delta[i];
            }
            new_delta[j] *= sp[j];
        }

        delta = new_delta;
        nabla_b[l] = delta;

        for (size_t i = 0; i < weights[l].size(); i++) {
            for (size_t j = 0; j < weights[l][i].size(); j++) {
                nabla_w[l][i][j] = delta[i] * activations[l][j];
            }
        }
    }


    return {nabla_b, nabla_w};
}

int Network::evaluate(const std::vector<std::pair<std::vector<double>, std::vector<double>>> &test_data)
{
    int correct = 0;
    for (const auto& [x, y] : test_data) {
        std::vector<double> output = feedForward(x);
        auto max_it = std::max_element(output.begin(), output.end());
        int predicted = std::distance(output.begin(), max_it);

        auto max_target_it = std::max_element(y.begin(), y.end());
        int target = std::distance(y.begin(), max_target_it);

        if (predicted == target) {
            ++correct;
        }
    }
    return correct;
}

void Network::print_net() {
    std::cout << "Network Structure:\n";
    
    std::cout << "Biases:\n";
    for (size_t i = 0; i < biases.size(); ++i) {
        std::cout << "Layer " << i << ":\n";
        for (const double &b : biases[i]) {
            std::cout << b << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nWeights:\n";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << "Layer " << i << ":\n";
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                std::cout << weights[i][j][k] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
