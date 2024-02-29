# Simple XOR Neural Network in C

## Description

This project implements a simple neural network in C that can learn the XOR function. The XOR (exclusive OR) problem is a classic problem in neural network literature, demonstrating a case where linear classifiers fail, and non-linear solutions are required. This neural network uses a single hidden layer and is trained using backpropagation.

## Features

- Sigmoid activation function for non-linearity.
- Initialization of weights with random values.
- Implementation of the forward pass and backpropagation to adjust weights based on errors.
- Use of a simple shuffle function to randomize the order of training data for each epoch.

## Requirements

- C compiler (e.g., GCC, Clang)
- Standard C libraries (`stdio.h`, `stdlib.h`, `math.h`)

## Usage

1. Clone this repository to your local machine.
2. Compile the source code using a C compiler. For example, using GCC:
   ```sh
   gcc -o xor_nn xor_nn.c -lm
   ```
   Note: The `-lm` flag is necessary to link the math library.
3. Run the compiled executable:
   ```sh
   ./xor_nn
   ```

## Implementation Details

### Activation Function:
The network uses the sigmoid function for activation, providing a smooth gradient for backpropagation.

### Weight Initialization:
Weights are initialized to random values using `init_weights` function, which generates numbers between 0 and 1.

### Training Data:
Hardcoded to represent the XOR truth table.

### Learning Rate:
Set to 0.1, adjustable based on training needs.

### Epochs:
The network trains for 10,000 epochs, which can be adjusted based on convergence requirements.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your improvements.

## License
This project is open-source and available under the MIT License.

