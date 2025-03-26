#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

class Network {
    private:
        int num_layers;
        std::vector<int> sizes;
        std::vector<std::vector<double>> biases;
        std::vector<std::vector<std::vector<double>>> weights;
    
    public:
        Network(std::vector<int> sizes); //done
        std::vector<double> feedForward(const std::vector<double>& input); //done
        void SGD(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_data, std::vector<std::pair<std::vector<double>, std::vector<double>>>& test_data, int epochs, int mini_batch_size, double eta);
        void update_mini_batch(std::vector<std::pair<std::vector<double>, std::vector<double>>>& mini_batch, double eta); //done
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> backprop(const std::vector<double>& x, const std::vector<double>& y); //done
        int evaluate(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& test_data);
        void print_net(); //done
    };


#endif