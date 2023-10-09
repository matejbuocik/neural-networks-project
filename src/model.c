#include "model.h"


// Function to create an MLP
MLP create_mlp(int input_size, int output_size, int num_hidden_layers, int hidden_layer_sizes[],
               double (*activation_functions[])(double), double (*activation_funs_der[])(double)) {
    MLP mlp;

    mlp.input_size = input_size;
    mlp.num_hidden_layers = num_hidden_layers;
    // allocate memory for arrays
    mlp.weights = (Matrix*)malloc((num_hidden_layers + 1) * sizeof(Matrix));
    mlp.inner_potentials = (Matrix*)malloc((num_hidden_layers + 1) * sizeof(Matrix));
    mlp.neuron_outputs = (Matrix*)malloc((num_hidden_layers + 1) * sizeof(Matrix));
    mlp.error_derivatives = (Matrix*)malloc((num_hidden_layers + 1) * sizeof(Matrix));
    mlp.activation_derivatives = (Matrix*)malloc((num_hidden_layers + 1) * sizeof(Matrix));
    mlp.weight_derivatives = (Matrix*)malloc((num_hidden_layers + 1) * sizeof(Matrix));
    mlp.activation_functions = (func_ptr*)malloc((num_hidden_layers + 1) * sizeof(func_ptr));
    mlp.activation_funs_der = (func_ptr*)malloc((num_hidden_layers + 1) * sizeof(func_ptr));

    // initialize allocated arrays
    for (int i = 0; i < num_hidden_layers + 1; i++) {
        int rows = ((i - 1) < 0) ? input_size : hidden_layer_sizes[i - 1];
        int cols = (i >= num_hidden_layers) ? output_size : hidden_layer_sizes[i];
    
        mlp.weights[i] = create_mat(rows, cols);
        mlp.weight_derivatives[i] = create_mat(rows, cols);
    
        mlp.inner_potentials[i] = create_mat(1, cols);
        mlp.neuron_outputs[i] = create_mat(1, cols);
        mlp.error_derivatives[i] = create_mat(1, cols);
        mlp.activation_derivatives[i] = create_mat(1, cols);

        mlp.activation_functions[i] = activation_functions[i];
        mlp.activation_funs_der[i] = activation_funs_der[i];
    }

    return mlp;
}

// Function to free memory used by the MLP
void free_mlp(MLP* mlp) {
    for (int i = 0; i < mlp->num_hidden_layers + 1; i++) {  
        free_mat(mlp->weights + i);
        free_mat(mlp->weight_derivatives + i);

        free_mat(mlp->inner_potentials + i);
        free_mat(mlp->neuron_outputs + i);
        free_mat(mlp->error_derivatives + i);
        free_mat(mlp->activation_derivatives + i);
    }

    free(mlp->weights);
    free(mlp->inner_potentials);
    free(mlp->neuron_outputs);
    free(mlp->error_derivatives);
    free(mlp->activation_derivatives);
    free(mlp->weight_derivatives);
    free(mlp->activation_functions);
    free(mlp->activation_funs_der);
}

// Function to initialize weights randomly
void initialize_weights(MLP* mlp, int seed) {
    
}

// Function to forward pass (compute neuron outputs)
void forward_pass(MLP* mlp, Matrix input);

// Function to compute derivatives during forward pass
void compute_derivatives(MLP* mlp, Matrix target_output);

// Function to update weights using stochastic gradient descent
void update_weights(MLP* mlp, double learning_rate);

// Function to train the MLP using stochastic gradient descent
void train(MLP* mlp, Matrix* input_data, Matrix* target_data, double learning_rate, int num_epochs, int batch_size, double (*error_function)(Matrix, Matrix));
