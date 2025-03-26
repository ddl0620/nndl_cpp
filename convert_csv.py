import pickle
import gzip
import pandas as pd

# Load the MNIST dataset
with gzip.open("data/mnist.pkl.gz", "rb") as f:
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

# Function to save dataset in CSV format
def save_csv(filename, images, labels):
    df = pd.DataFrame(images)
    df.insert(0, "label", labels)  # Insert labels as the first column
    df.to_csv(filename, index=False, header=False)

# Save training, validation, and test sets as separate CSV files
save_csv("data/mnist_train.csv", training_data[0], training_data[1])
save_csv("data/mnist_valid.csv", validation_data[0], validation_data[1])
save_csv("data/mnist_test.csv", test_data[0], test_data[1])

print("MNIST dataset saved as CSV files!")
