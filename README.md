# MNIST digits recognition from scratch with C++

A simple neural network built entirely from scratch in C++ using only the standard library—no NumPy, no TensorFlow, no Eigen. Just pure C++ and math.

## Features
- A fully implemented neural network from scratch in C++
- Custom-built math library featuring sigmoid and sigmoid prime functions
- Efficient backpropagation and stochastic gradient descent (SGD) for training
- Seamless support for the MNIST handwritten digit recognition dataset
- Pure C++ solution using only the standard library, with no external dependencies

## Project Structure
```
.
├── data/                    # Folder for MNIST dataset (CSV format)
├── loader/                  # Folder for MNIST dataset loader
├── math_lib/                # Custom math library
├── network/                 # Neural network implementation
├── main.cpp                 # Main entry point of the program
├── Makefile                 # Build system
├── README.md                # Project documentation
```

## Installation & Usage
### 1. Clone the repository
```sh
git clone https://github.com/ddl0620/nndl_cpp.git
cd nndl_cpp
```

### 1.1 Preparing the MNIST Dataset
This project uses the MNIST dataset in CSV format. If you don’t have the dataset, use the provided Python script to convert the `.pkl.gz` version:
```sh
python scripts/convert_csv.py
```

### 2. Build the project
Ensure you have **g++** installed, then run:
```sh
make
```

### 3. Run the program
```sh
./nndl_cpp
```

## Excluding Data from Git
The following files are ignored in version control using `.gitignore`:
```
data/
data/mnist_train.csv
data/mnist_valid.csv
data/mnist_test.csv
```

