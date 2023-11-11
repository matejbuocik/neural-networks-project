#include "MLP.h"
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "parse_csv.h"


MLP create_mlp(int input_size, int output_size, int num_hidden_layers, int hidden_layer_sizes[],
               func_ptr activation_functions[], func_ptr activation_funs_der[]) {//CHANGE
    MLP mlp;

    mlp.num_hidden_layers = num_hidden_layers;
    mlp.activation_functions = activation_functions;
    mlp.activation_funs_der = activation_funs_der;

    // allocate memory for arrays
    mlp.weights                 = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.inner_potentials        = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.neuron_outputs          = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));

    mlp.error_derivatives       = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.activation_derivatives  = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.weight_derivatives      = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));

    // initialize allocated arrays
    int plus_one_output_col = 1;
    for (int i = 0; i <= num_hidden_layers; i++) {
        int rows = (i == 0) ? input_size : hidden_layer_sizes[i - 1];  // neurons in previous layer
        int cols = hidden_layer_sizes[i];  // neurons in the next layer
        if (i == num_hidden_layers) {
            cols = output_size;
            plus_one_output_col = 0;  // Do not add 1 to the output vector
        }

        rows += 1;  // Add 1 for bias

        mlp.weights[i] = create_mat(rows, cols);
        mlp.weight_derivatives[i] = create_mat(rows, cols);

        mlp.inner_potentials[i] = create_mat(1, cols);
        mlp.neuron_outputs[i] = create_mat(1, cols + plus_one_output_col);  // First one will always be one (input for bias)
        mlp.error_derivatives[i] = create_mat(cols, 1);
        mlp.activation_derivatives[i] = create_mat(cols, 1);
    }

    return mlp;
}

void free_mlp(MLP* mlp) {
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        free_mat(mlp->weights[i]);
        free_mat(mlp->weight_derivatives[i]);

        free_mat(mlp->inner_potentials[i]);
        free_mat(mlp->neuron_outputs[i]);
        
        free_mat(mlp->error_derivatives[i]);
        printf("here %d\n", i);
        free_mat(mlp->activation_derivatives[i]);
        
    }

    free(mlp->weights);
    free(mlp->inner_potentials);
    free(mlp->neuron_outputs);

    free(mlp->error_derivatives);
    free(mlp->activation_derivatives);
    free(mlp->weight_derivatives);
}

void initialize_weights(MLP* mlp, int seed, double max_val, double min_val) {
    // TODO smart initialization (normal He for ReLU, normal Glorot for softmax)
    // TODO normal distribution function
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

Matrix *forward_pass(MLP *mlp, Matrix *input) {
    Matrix *prev_layer = input;

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        multiply_mat(prev_layer, mlp->weights[i], mlp->inner_potentials[i]);
        mlp->activation_functions[i](mlp->inner_potentials[i], mlp->neuron_outputs[i]);

        prev_layer = mlp->neuron_outputs[i];
    }

    return mlp->neuron_outputs[mlp->num_hidden_layers];
}

void backpropagate(MLP *mlp, Matrix *input, Matrix *target_output) {
    // TODO special case for output layer softmax

    // compute derivatives of the error function with respect to the neuron outputs of the last layer
    // the result is transposed, because of the way the derivatives with regards to the neuron outputs
    // in other layers are computed (multiplication with the same weight matrices as in forward pass
    // just from the other side)
    // TODO do not allocate new matrices, everything needs to be there in the beginning
    Matrix *deriv_last = sub_mat(mlp->neuron_outputs[mlp->num_hidden_layers], target_output);
    Matrix *deriv_last_T = transpose_mat(deriv_last);
    free_mat(deriv_last);

    mlp->activation_funs_der[mlp->num_hidden_layers](mlp->inner_potentials[mlp->num_hidden_layers],
                                                     mlp->activation_derivatives[mlp->num_hidden_layers]);
    elem_multiply_mat(deriv_last_T, mlp->activation_derivatives[mlp->num_hidden_layers],
                      mlp->error_derivatives[mlp->num_hidden_layers]);

    free_mat(deriv_last_T);
    // computing derivatives of the error function with respect to all the other neuron outputs
    for (int i = mlp->num_hidden_layers - 1; i >= 0; i--) {
        multiply_mat(mlp->weights[i + 1], mlp->error_derivatives[i + 1], mlp->error_derivatives[i]);
        mlp->activation_funs_der[i](mlp->inner_potentials[i], mlp->activation_derivatives[i]);
        elem_multiply_mat(mlp->error_derivatives[i], mlp->activation_derivatives[i], mlp->error_derivatives[i]);
    }

    // computing derivatives of the error function with respect to all weights
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        for (int i = 0; i < mlp->weight_derivatives[k]->rows; i++) {
            for (int j = 0; j < mlp->weight_derivatives[k]->cols; j++) {
                double grad = get_element(mlp->weight_derivatives[k], i, j);
                // compute derivative
                Matrix* neuron_vals = (k == 0) ? input : mlp->neuron_outputs[k - 1];
                grad += get_element(mlp->error_derivatives[k], j, 0) * get_element(neuron_vals, 0, i);
                set_element(mlp->weight_derivatives[k], i, j, grad);
            }
        }
    }
}

void set_derivatives_to_zero(MLP *mlp) {
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        multiply_scalar_mat(mlp->weight_derivatives[k], 0.0, mlp->weight_derivatives[k]);
    }
}

void gradient_descent(MLP *mlp, double learning_rate) {
    // TODO use better techniques (adaptive learning rate, momentum, ...)
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        multiply_scalar_mat(mlp->weight_derivatives[k], learning_rate, mlp->weight_derivatives[k]);
        subtract_mat(mlp->weights[k], mlp->weight_derivatives[k], mlp->weights[k]);
    }

    set_derivatives_to_zero(mlp);
}

int _get_random_int(int min, int max) {
    int range = max - min + 1;  // include min and max
    int random_int = min + rand() % range;

    return random_int;
}

void train(MLP* mlp, int num_samples, Matrix *input_data[], Matrix *target_data[], double learning_rate, int num_batches, int batch_size) {
    // TODO use classification
    // input_data[0] must be 1
    for (int batch = 0; batch < num_batches; batch++) {
        for (int i = 0; i < batch_size; i++) {
            // TODO spustit na batch_size procesoroch naraz
            int data_i = _get_random_int(0, num_samples - 1);

            forward_pass(mlp, input_data[data_i]);
            backpropagate(mlp, input_data[data_i], target_data[data_i]);
        }

        gradient_descent(mlp, learning_rate);

        // print_matrices(mlp->weights, mlp->num_hidden_layers);
    }
}

double _mse(Matrix* mat1, Matrix* mat2) {
    Matrix *sub = sub_mat(mat1, mat2);
    apply_func_mat(sub, sub, fabs, false);

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
        Matrix *computed_out = forward_pass(mlp, input_data[i]);

        print_matrices(&computed_out, 1);

        res += metric_fun(computed_out, target_data[i]);

        printf("%f, %f\n", get_element(computed_out, 0, 0), get_element(target_data[i], 0, 0));
    }

    print_matrices(mlp->weights, mlp->num_hidden_layers);

    return res;
}
