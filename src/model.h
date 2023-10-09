#include "matrices.h"


typedef struct {
    int input_size;
    int output_size;
    int num_hidden_layers;
    int* hidden_layer_sizes;
    Matrix weights[]; // Array of weight matrices
    Matrix inner_potentials[]; // Array of inner potentials
    Matrix neuron_outputs[]; // Array of neuron outputs
    Matrix error_derivatives[]; // Array of derivatives of the error with respect to neuron outputs
    double** activation_derivatives; // Array of derivatives of activation functions
    Matrix* weight_derivatives; // Array of weight derivatives
    double (*activation_functions[])(double); // Activation function pointers
    double (*activation_derivatives[])(double); // Activation derivative function pointers
} MLP;

// Function to create an MLP
MLP create_mlp(int input_size, int num_hidden_layers, int* hidden_layer_sizes,
               double (*activation_functions[])(double), double (*activation_derivatives[])(double));

// Function to free memory used by the MLP
void free_mlp(MLP* mlp);

// Function to initialize weights randomly
void initialize_weights(MLP* mlp);

// Function to forward pass (compute neuron outputs)
void forward_pass(MLP* mlp, Matrix input);

// Function to compute derivatives during forward pass
void compute_derivatives(MLP* mlp, Matrix target_output);

// Function to update weights using stochastic gradient descent
void update_weights(MLP* mlp, double learning_rate);

// Function to train the MLP using stochastic gradient descent
void train(MLP* mlp, Matrix* input_data, Matrix* target_data, double learning_rate, int num_epochs, int batch_size, double (*error_function)(Matrix, Matrix));
