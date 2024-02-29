#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Simple nn that can learn xor

// Activation function for sigmoid
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of sigmoid
double dSigmoid(double x) {
    return x * (1 - x);
}

// Initialize random values (0,1) for weights, to be adjusted
double init_weights() {
    return ((double) rand()) / ((double) RAND_MAX);
}

// Shuffle the data, randomizing indexes
void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

int main(void) {
    // Learning rate
    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    // Two-dim matrix of weights
    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = {
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 1.0f}
    };

    double training_outputs[numTrainingSets][numOutputs] = {
            {0.0f},
            {1.0f},
            {1.0f},
            {0.0f}
    };

    // Setting random values for each of the elements
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weights();
        }
    }

    // Setting random values for each of the elements in the second layer
    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weights();
        }
    }

    // Setting random values for bias
    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weights();
    }

    // Setting random values for bias
    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weights();
    }

    int trainingSetOrder[] = {0, 1, 2, 3};

    int numberOfEpochs = 10000;

    // Train the neural network for the number of epochs
    for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
        shuffle(trainingSetOrder, numTrainingSets);

        for (int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];

            // Forward pass

            // Compute hidden input layer activation
            for (int j = 0; j < numHiddenNodes; j++) {
                // Add bias
                double activation = hiddenLayerBias[j];

                // Adding activation for inputs/weights for the first layer
                for (int k = 0; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }

                hiddenLayer[j] = sigmoid(activation);
            }

            // Compute hidden output layer activation
            for (int j = 0; j < numOutputs; j++) {
                // Add bias
                double activation = outputLayerBias[j];

                // Adding activation for inputs/weights for the first layer
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }

                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g   Output: %g  Predicted Output: %g \n",
                   training_inputs[i][0], training_inputs[i][1],
                   outputLayer[0], training_outputs[i][0]);

            // Backpropagation

            // Compute change in output weights
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double error = (training_outputs[i][j] - outputLayer[j]);
                // train... is an actual value, where outL... is a predicted value
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            // Compute change in hidden weights
            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            // Apply change in output weights
            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            // Apply change in hidden weights
            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    // Print final weights after done training
    printf("\nFinal Hidden Weights:\n");
    for (int j = 0; j < numHiddenNodes; j++) {
        for (int k = 0; k < numInputs; k++) {
            printf("%f ", hiddenWeights[k][j]);
        }
        printf("\n");
    }

    printf("\nFinal Output Weights:\n");
    for (int j = 0; j < numOutputs; j++) {
        for (int k = 0; k < numHiddenNodes; k++) {
            printf("%f", outputWeights[k][j]);
        }
        printf("\n");
    }

    printf("\nFinal Hidden Biases:\n");
    for (int j = 0; j < numHiddenNodes; j++) {
        printf("%f", hiddenLayerBias[j]);
    }
    printf("\n");

    printf("\nFinal Output Biases:\n");
    for (int j = 0; j < numOutputs; j++) {
        printf("%f", outputLayerBias[j]);
    }
    printf("\n");

    return 0;
}
