#include "MLP.h"
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "parse_csv.h"
#include "activation_functions.h"


MLP create_mlp(int input_size, int output_size, int num_hidden_layers, int hidden_layer_sizes[],
               func_ptr activation_functions[], func_ptr activation_funs_der[]) {//CHANGE
    MLP mlp;

    mlp.num_hidden_layers = num_hidden_layers;
    mlp.activation_functions = activation_functions;
    mlp.activation_funs_der = activation_funs_der;

    // allocate memory for arrays
    mlp.layers_sizes     = (int *) malloc((num_hidden_layers + 2) * sizeof(int));

    mlp.weights                 = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.inner_potentials        = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.neuron_outputs          = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    // the error_derivatives[-1] is added for extensions of the MLP with adding preciding layers
    // in such case this filed is needed for the backpropagation on the whole ensemble
    mlp.error_derivatives       = (Matrix**) malloc((num_hidden_layers + 2) * sizeof(Matrix*));
    mlp.error_derivatives = mlp.error_derivatives + 1;
    mlp.error_derivatives[-1] = create_mat(input_size, 1);
    mlp.activation_derivatives  = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.weight_derivatives      = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));

    // initialize allocated arrays
    int i;
    int plus_one_output_col = 1;
    for (i = 0; i <= num_hidden_layers; i++) {
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

        mlp.layers_sizes[i + 1] = hidden_layer_sizes[i];
    }

    mlp.layers_sizes[0] = input_size;
    mlp.layers_sizes[i] = output_size; // i == num_hidden_layers + 1

    return mlp;
}

void free_mlp(MLP* mlp) {
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        free_mat(mlp->weights[i]);
        free_mat(mlp->weight_derivatives[i]);

        free_mat(mlp->inner_potentials[i]);
        free_mat(mlp->neuron_outputs[i]);

        free_mat(mlp->error_derivatives[i]);
        free_mat(mlp->activation_derivatives[i]);

    }

    free(mlp->layers_sizes);

    free(mlp->weights);
    free(mlp->inner_potentials);
    free(mlp->neuron_outputs);

    free(mlp->error_derivatives);
    free(mlp->activation_derivatives);
    free(mlp->weight_derivatives);
}

void initialize_weights(MLP* mlp, int seed) {
    srand(seed);

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        int input_size = mlp->layers_sizes[i];
        int output_size = mlp->layers_sizes[i + 1];

        double (*generator)(double, double);
        double arg1, arg2;

        if (mlp->activation_functions[i] == &ReLU) {
            // normal He
            generator = generate_normal_random;
            arg1 = 0.0;
            arg2 = 4.0 / (double)(input_size + output_size);
        } else if (mlp->activation_functions[i] == &softmax || mlp->activation_functions[i] == &sigmoid) {
            // normal Glorot
            generator = generate_normal_random;
            arg1 = 0.0;
            arg2 = 2.0 / (double)(input_size + output_size);
        }

        for (int j = 0; j < mlp->weights[i]->rows; j++) {
            for (int k = 0; k < mlp->weights[i]->cols; k++) {
                double random_val = generator(arg1, arg2);
                set_element(mlp->weights[i], j, k, random_val);
            }
        }
    }
}

Matrix *forward_pass(MLP *mlp, Matrix *input) {
    Matrix *prev_layer = input;

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        multiply_mat(prev_layer, mlp->weights[i], mlp->inner_potentials[i], false);
        mlp->activation_functions[i](mlp->inner_potentials[i], mlp->neuron_outputs[i]);

        prev_layer = mlp->neuron_outputs[i];
// print_matrices(&prev_layer, 1);
    }

    return mlp->neuron_outputs[mlp->num_hidden_layers];
}

void backpropagate(MLP *mlp, Matrix *input, Matrix *target_output) {
    // compute derivatives of the error function with respect to the neuron outputs of the last layer
    // the result is transposed, because of the way the derivatives with regards to the neuron outputs
    // in other layers are computed (multiplication with the same weight matrices as in forward pass
    // just from the other side)
    // Matrix *deriv_last = sub_mat(mlp->neuron_outputs[mlp->num_hidden_layers], target_output);
    // Matrix *deriv_last_T = transpose_mat(deriv_last);
    // free_mat(deriv_last);

    // mlp->activation_funs_der[mlp->num_hidden_layers](mlp->inner_potentials[mlp->num_hidden_layers],
    //                                                  mlp->activation_derivatives[mlp->num_hidden_layers]);
    // elem_multiply_mat(deriv_last_T, mlp->activation_derivatives[mlp->num_hidden_layers],
    //                   mlp->error_derivatives[mlp->num_hidden_layers]);

    // free_mat(deriv_last_T);

    int last = mlp->num_hidden_layers;

    double suma_posledna_vrstva_e_na_vnutorny_potencial = 0.0;
    for (int i = 0; i < mlp->inner_potentials[last]->cols; i++) {
        suma_posledna_vrstva_e_na_vnutorny_potencial += exp(mlp->inner_potentials[last]->data[0][i]);
    }

    for (int i = 0; i < target_output->cols; i++) {
        if (target_output->data[0][i] == 0) {
            mlp->error_derivatives[last]->data[i][0] = 0;
            continue;
        }

        // Compute dE_k/dy_j * softmax derivation for the last layer
        // mlp->error_derivatives[last]->data[i][0] = target_output->data[0][i] / mlp->neuron_outputs[last]->data[0][i];  // dE_k/dy_j
        // mlp->error_derivatives[last]->data[i][0] *= mlp->neuron_outputs[last]->data[0][i] * (1 - mlp->neuron_outputs[last]->data[0][i]);
        mlp->error_derivatives[last]->data[i][0] = (1 - mlp->neuron_outputs[last]->data[0][i]);  // it's the same wtf

        // Next to last layer
        for (int j = 0; j < mlp->error_derivatives[last - 1]->rows; j++) {
            double suma_s_nasobenim_vahy = 0.0;
            for (int k = 0; k < mlp->inner_potentials[last]->cols; k++) {
                suma_s_nasobenim_vahy += exp(mlp->inner_potentials[last]->data[0][k]) * mlp->weights[last]->data[j+1][k];
            }

            mlp->error_derivatives[last - 1]->data[j][0] = mlp->weights[last]->data[j+1][i] - suma_s_nasobenim_vahy / suma_posledna_vrstva_e_na_vnutorny_potencial;
        }
    }

    mlp->activation_funs_der[last - 1](mlp->inner_potentials[last - 1], mlp->activation_derivatives[last - 1]);
    elem_multiply_mat(mlp->error_derivatives[last - 1], mlp->activation_derivatives[last - 1], mlp->error_derivatives[last - 1]);

    for (int i = last - 2; i >= 0; i--) {
        multiply_mat(mlp->weights[i + 1], mlp->error_derivatives[i + 1], mlp->error_derivatives[i], true);
        mlp->activation_funs_der[i](mlp->inner_potentials[i], mlp->activation_derivatives[i]);
        elem_multiply_mat(mlp->error_derivatives[i], mlp->activation_derivatives[i], mlp->error_derivatives[i]);
    }

    // For preceding layers extensions
    multiply_mat(mlp->weights[0], mlp->error_derivatives[0], mlp->error_derivatives[-1], true);

    // computing derivatives of the error function with respect to all weights
    for (int k = 0; k <= last; k++) {
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

void multiply_derivatives_by(MLP* mlp, double factor) {
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        multiply_scalar_mat(mlp->weight_derivatives[k], factor, mlp->weight_derivatives[k]);
    }
}

void gradient_descent(MLP *mlp, double learning_rate, int batch_size, double aplha) {
    // TODO use better techniques (adaptive learning rate, momentum, ...)
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        multiply_scalar_mat(mlp->weight_derivatives[k], -learning_rate / batch_size, mlp->weight_derivatives[k]);
        subtract_mat(mlp->weights[k], mlp->weight_derivatives[k], mlp->weights[k]);
    }
// print_matrices(mlp->weights, mlp->num_hidden_layers + 1);
// print_matrices(mlp->weight_derivatives, mlp->num_hidden_layers + 1);
    multiply_derivatives_by(mlp, aplha);
}

void train(MLP* mlp, int num_samples, Matrix *input_data[], Matrix *target_data[],
           double learning_rate, int num_batches, int batch_size, double alpha) {
    // input_data[0] must be 1
    multiply_derivatives_by(mlp, 0.0);

    for (int batch = 0; batch < num_batches; batch++) {
        for (int i = 0; i < batch_size; i++) {
            // TODO spustit na batch_size procesoroch naraz
            int data_i = get_random_int(0, num_samples - 1);

            forward_pass(mlp, input_data[data_i]);
            backpropagate(mlp, input_data[data_i], target_data[data_i]);
// print_matrices(mlp->weight_derivatives, mlp->num_hidden_layers + 1);
        }

        gradient_descent(mlp, learning_rate, batch_size, alpha);

// print_matrices(mlp->weights, mlp->num_hidden_layers + 1);
    }
}

double test(MLP* mlp, int num_samples, Matrix *input_data[], Matrix *target_data[], double (*metric_fun)(Matrix*, Matrix*)) {
    if (metric_fun == NULL) {
        metric_fun = &mse;  // default metric function
    }
    double res = 0.0;

    int hits = 0;

    for (int i = 0; i < num_samples; i++) {
        Matrix *computed_out = forward_pass(mlp, input_data[i]);

        // print_matrices(&computed_out, 1);

        // res += metric_fun(computed_out, target_data[i]);

        double max_comp = computed_out->data[0][0];
        int max_index = 0;
        int target_index = 0;
        for (int j = 0; j < target_data[i]->cols; j++) {
            if (computed_out->data[0][j] > max_comp) {
                max_index = j;
                max_comp = computed_out->data[0][j];
            }

            if (target_data[i]->data[0][j] == 1) {
                target_index = j;
            }
        }

        if (max_index == target_index) {
            hits++;
        } else {
            for (int j = 0; j < target_data[i]->cols; j++) {
                printf("%f, ", computed_out->data[0][j]);
            }

            printf("   Target: %d\n", target_index);
        }

        // printf("%f, %f\n", get_element(computed_out, 0, 0), get_element(target_data[i], 0, 0));
        // printf("%f, %f      %f, %f\n", computed_out->data[0][0], computed_out->data[0][1], target_data[i]->data[0][0], target_data[i]->data[0][1]);
    }

    printf("%d / %d\n", hits, num_samples);

    // print_matrices(mlp->weights, mlp->num_hidden_layers + 1);

    return res;
}
