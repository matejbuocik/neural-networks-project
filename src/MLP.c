#include "MLP.h"
#include <stdbool.h>
#include <time.h>
#include <math.h>


// Function to create an MLP
MLP create_mlp(int input_size, int output_size, int num_hidden_layers, int hidden_layer_sizes[],
               func_ptr activation_functions[], func_ptr activation_funs_der[]) {
    MLP mlp;

    mlp.input_size = input_size;
    mlp.input = NULL;
    mlp.num_hidden_layers = num_hidden_layers;
    // allocate memory for arrays
    mlp.weights = (Matrix**)malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.inner_potentials = (Matrix**)malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.neuron_outputs = (Matrix**)malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.error_derivatives = (Matrix**)malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.activation_derivatives = (Matrix**)malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.weight_derivatives = (Matrix**)malloc((num_hidden_layers + 1) * sizeof(Matrix*));
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
        free_mat(mlp->weights[i]);
        free_mat(mlp->weight_derivatives[i]);

        free_mat(mlp->inner_potentials[i]);
        free_mat(mlp->neuron_outputs[i]);
        free_mat(mlp->error_derivatives[i]);
        free_mat(mlp->activation_derivatives[i]);
    }

    free(mlp->weights);
    free(mlp->inner_potentials);
    free(mlp->neuron_outputs);
    free(mlp->error_derivatives);
    free(mlp->activation_derivatives);
    free(mlp->weight_derivatives);
    free(mlp->activation_functions);
    free(mlp->activation_funs_der);

    if (mlp->input != NULL) {
        free_mat(mlp->input);
    }
}

// Function to initialize weights randomly
void initialize_weights(MLP* mlp, int seed, double max_val, double min_val) {
    srand(seed);

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        for (int j = 0; j < mlp->weights[i]->rows; j++) {
            for (int k = 0; k < mlp->weights[i]->cols; k++) {
                double random_val = ((double)rand() / RAND_MAX) * (max_val - min_val) + min_val;
                set_element(mlp->weights[i], j, k, random_val);
            }
        }
    }
}

// Function to forward pass (compute neuron outputs)
Matrix forward_pass(MLP* mlp, Matrix *input) {
    Matrix *input_copy = copy_mat(input);
    mlp->input = input_copy;

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        // in first iteration consider input as previous layer
        Matrix *prev_layer = ((i - 1) < 0) ? mlp->input : mlp->neuron_outputs[i - 1];

        mult_mat_with_out(prev_layer, mlp->weights[i], mlp->inner_potentials[i]);
        apply_to_mat_with_out(mlp->inner_potentials[i],
                              mlp->neuron_outputs[i],
                              mlp->activation_functions[i], false);
    }

    return *(mlp->neuron_outputs[mlp->num_hidden_layers]);
}

// Function to compute derivatives during backward pass
void compute_derivatives(MLP* mlp, Matrix *target_output) {
    // apply the derivative of activation function to inner potentials (maybe does not need to be stored)
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        apply_to_mat_with_out(mlp->inner_potentials[i],
                                mlp->activation_derivatives[i],
                                mlp->activation_funs_der[i], true);
    }

    // compute derivatives of the error function with respect to the neuron outputs of the last layer
    // the result is transposed, because of the way the derivatives with regards to the neuron outputs
    // in other layers are computed (multiplication with the same weight matrices as in forward pass
    // just from the other side)
    Matrix *deriv_last = sub_mat(mlp->neuron_outputs[mlp->num_hidden_layers], target_output);
    Matrix *deriv_last_T = transpose_mat(deriv_last);
    free_mat(deriv_last);

    mult_with_out(deriv_last_T, mlp->activation_derivatives[mlp->num_hidden_layers],
                  mlp->error_derivatives[mlp->num_hidden_layers]);
    
    free_mat(deriv_last_T);
    // computing derivatives of the error function with respect to all the other neuron outputs
    for (int i = mlp->num_hidden_layers - 1; i >= 0; i--) {
        mult_mat_with_out(mlp->weights[i + 1], mlp->error_derivatives[i + 1], mlp->error_derivatives[i]);
        mult_with_out(mlp->error_derivatives[i], mlp->activation_derivatives[i], mlp->error_derivatives[i]);
    }

    // computing derivatives of the error function with respect to all the weights
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        for (int i = 0; i < mlp->weight_derivatives[k]->rows; i++) {
            for (int j = 0; j < mlp->weight_derivatives[k]->cols; j++) {
                double grad = get_element(mlp->weight_derivatives[k], i, j);
                // compute derivative
                Matrix* neuron_vals = ((k - 1) < 0) ? mlp->input : mlp->neuron_outputs[k - 1];
                grad += get_element(mlp->error_derivatives[k], j, 0) * get_element(neuron_vals, 0, i);
                set_element(mlp->weight_derivatives[k], i, j, grad);
            }
        }
    }
}

void set_derivatives_to_zero(MLP* mlp) {
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        mult_scal_with_out(mlp->weight_derivatives[k], 0.0, mlp->weight_derivatives[k]);
    }
}

// Function to update weights using stochastic gradient descent
void update_weights(MLP* mlp, double learning_rate) {
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        mult_scal_with_out(mlp->weight_derivatives[k], learning_rate, mlp->weight_derivatives[k]);
        sub_mat_with_out(mlp->weights[k], mlp->weight_derivatives[k], mlp->weights[k]);
    }

    set_derivatives_to_zero(mlp);
}

int _get_random_int(int min, int max) {
    int range = max - min + 1;  // include min and max
    int random_int = rand() % range + min;

    return random_int;
}

// Function to train the MLP using stochastic gradient descent
void train(MLP* mlp, int num_samples, Matrix *input_data[], Matrix *target_data[], double learning_rate, int num_batches, int batch_size) {
    for (int batch = 0; batch < num_batches; batch++) {
        for (int i = 0; i < batch_size; i++) {
            int data_i = _get_random_int(0, num_samples - 1);

            forward_pass(mlp, input_data[data_i]);
            compute_derivatives(mlp, target_data[data_i]);
        }

        update_weights(mlp, learning_rate);
    }
}

double _mse(Matrix* mat1, Matrix* mat2) {
    Matrix *sub = sub_mat(mat1, mat2);
    apply_to_mat_with_out(sub, sub, fabs, false);

    double sum = sum_mat(sub);
    free_mat(sub);

    return sum;
}

double test(MLP* mlp, int num_samples, Matrix *input_data[], Matrix *target_data[], double (*metric_fun)(Matrix*, Matrix*)) {
    if (metric_fun == NULL) {
        metric_fun = &_mse;  // default metric function
    }
    double res = 0.0;

    for (int i = 0; i < num_samples; i++) {
        Matrix computed_out = forward_pass(mlp, input_data[i]);
        res += metric_fun(&computed_out, target_data[i]);

        printf("%f, %f\n", get_element(&computed_out, 0, 0), get_element(target_data[i], 0, 0));
    }

    return res;
}