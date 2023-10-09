#include "model.h"
#include <stdbool.h>


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
    for (int i = 0; i <= num_hidden_layers; i++) {
        int rows = ((i - 1) < 0) ? input_size : hidden_layer_sizes[i - 1];
        int cols = (i >= num_hidden_layers) ? output_size : hidden_layer_sizes[i];
    
        mlp.weights[i] = create_mat(rows, cols);
        mlp.weight_derivatives[i] = create_mat(rows, cols);
    
        mlp.inner_potentials[i] = create_mat(1, cols);
        mlp.neuron_outputs[i] = create_mat(1, cols);
        mlp.error_derivatives[i] = create_mat(cols, 1);
        mlp.activation_derivatives[i] = create_mat(cols, 1);

        mlp.activation_functions[i] = activation_functions[i];
        mlp.activation_funs_der[i] = activation_funs_der[i];
    }

    return mlp;
}

// Function to free memory used by the MLP
void free_mlp(MLP* mlp) {
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {  
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
void initialize_weights(MLP* mlp, int seed, double max_val, double min_val) {
    srand(seed);

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        for (int j = 0; j < mlp->weights[i].rows; j++) {
            for (int k = 0; k < mlp->weights[i].cols; k++) {
                double random_val = ((double)rand() / RAND_MAX) * (max_val - min_val) + min_val;
                set_element(mlp->weights + i, j, k, random_val);
            }
        }
    }
}

// Function to forward pass (compute neuron outputs)
Matrix forward_pass(MLP* mlp, Matrix input, bool prep_back) {
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        // in first iteration consider input as previous layer
        Matrix* prev_layer = ((i - 1) < 0) ? &input : mlp->neuron_outputs + i - 1;

        mult_mat_with_out(prev_layer, mlp->weights + i, mlp->inner_potentials + i);
        apply_to_mat_with_out(mlp->inner_potentials + i, mlp->neuron_outputs + i, mlp->activation_functions[i], false);
        
        if (prep_back)
            apply_to_mat_with_out(mlp->inner_potentials + i, mlp->activation_derivatives + i, mlp->activation_funs_der[i], true);
    }

    return mlp->neuron_outputs[mlp->num_hidden_layers];
}

// Function to compute derivatives during forward pass
void compute_derivatives(MLP* mlp, Matrix target_output) {
    
}

void set_derivatives_to_zero(MLP* mlp);

// Function to update weights using stochastic gradient descent
void update_weights(MLP* mlp, double learning_rate);

// Function to train the MLP using stochastic gradient descent
void train(MLP* mlp, Matrix* input_data, Matrix* target_data, double learning_rate, int num_epochs, int batch_size);
