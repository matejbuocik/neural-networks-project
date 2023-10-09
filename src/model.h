#include "matrices.h"


typedef double (*func_ptr)(double);


typedef struct {
    int input_size;
    int output_size;
    int num_hidden_layers;
    Matrix* weights;
    Matrix* inner_potentials;
    Matrix* neuron_outputs;
    Matrix* error_derivatives;
    Matrix* activation_derivatives;
    Matrix* weight_derivatives;
    func_ptr* activation_functions;
    func_ptr* activation_funs_der;
} MLP;

// Function to create an MLP
MLP create_mlp(int input_size, int output_size, int num_hidden_layers, int hidden_layer_sizes[],
               double (*activation_functions[])(double), double (*activation_derivatives[])(double));

// Function to free memory used by the MLP
void free_mlp(MLP* mlp);

// Function to initialize weights randomly
void initialize_weights(MLP* mlp, int seed);

// Function to forward pass (compute neuron outputs)
void forward_pass(MLP* mlp, Matrix input);

// Function to compute derivatives during forward pass
void compute_derivatives(MLP* mlp, Matrix target_output);

// Function to update weights using stochastic gradient descent
void update_weights(MLP* mlp, double learning_rate);

// Function to train the MLP using stochastic gradient descent
void train(MLP* mlp, Matrix* input_data, Matrix* target_data, double learning_rate, int num_epochs, int batch_size, double (*error_function)(Matrix, Matrix));
