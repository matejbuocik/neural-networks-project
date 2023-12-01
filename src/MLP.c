#include "MLP.h"
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <omp.h>


MLP create_mlp(int input_size, int output_size, int num_hidden_layers, int hidden_layer_sizes[],
               func_ptr activation_functions[], func_ptr activation_funs_der[]) {//CHANGE
    MLP mlp;

    mlp.num_hidden_layers = num_hidden_layers;
    mlp.activation_functions = activation_functions;
    mlp.activation_funs_der = activation_funs_der;

    /* Shared arrays */
    mlp.weights                 = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.weight_deltas           = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));

    // Adam
    mlp.first_momentum          = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    mlp.second_momentum         = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));

    /* Initialize shared arrays */
    for (int i = 0; i <= num_hidden_layers; i++) {
        int rows = (i == 0) ? input_size : hidden_layer_sizes[i - 1];  // neurons in previous layer
        int cols = hidden_layer_sizes[i];  // neurons in the next layer
        if (i == num_hidden_layers) {
            cols = output_size;
        }

        rows += 1;  // Add 1 for bias

        mlp.weights[i] = create_mat(rows, cols);
        mlp.weight_deltas[i] = create_mat(rows, cols);

        // Adam
        mlp.first_momentum[i] = create_mat(rows, cols);
        mlp.second_momentum[i] = create_mat(rows, cols);
    }


    /* Thread specific arrays */
    mlp.inner_potentials        = (Matrix***) malloc(NUM_THREADS * sizeof(Matrix**));
    mlp.neuron_outputs          = (Matrix***) malloc(NUM_THREADS * sizeof(Matrix**));

    mlp.error_derivatives       = (Matrix***) malloc(NUM_THREADS * sizeof(Matrix**));

    mlp.activation_derivatives  = (Matrix***) malloc(NUM_THREADS * sizeof(Matrix**));
    mlp.weight_derivatives      = (Matrix***) malloc(NUM_THREADS * sizeof(Matrix**));

    for (int t = 0; t < NUM_THREADS; t++) {
        mlp.inner_potentials[t]        = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
        mlp.neuron_outputs[t]          = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));

        // the error_derivatives[-1] is added for extensions of the MLP with adding preceding layers
        // in such case this filed is needed for the backpropagation on the whole ensemble
        mlp.error_derivatives[t]       = (Matrix**) malloc((num_hidden_layers + 2) * sizeof(Matrix*));
        mlp.error_derivatives[t]       = mlp.error_derivatives[t] + 1;
        mlp.error_derivatives[t][-1]   = create_mat(input_size, 1);

        mlp.activation_derivatives[t]  = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
        mlp.weight_derivatives[t]      = (Matrix**) malloc((num_hidden_layers + 1) * sizeof(Matrix*));
    }

    /* Initialize thread-specific arrays */
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

        for (int t = 0; t < NUM_THREADS; t++) {
            mlp.weight_derivatives[t][i] = create_mat(rows, cols);
            mlp.inner_potentials[t][i] = create_mat(1, cols);
            mlp.neuron_outputs[t][i] = create_mat(1, cols + plus_one_output_col);  // First one will always be one (input for bias)
            mlp.error_derivatives[t][i] = create_mat(cols, 1);
            mlp.activation_derivatives[t][i] = create_mat(cols, 1);
        }
    }

    return mlp;
}

void free_mlp(MLP* mlp) {
    /* Free shared arrays */
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        free_mat(mlp->weights[i]);
        free_mat(mlp->weight_deltas[i]);

        free_mat(mlp->first_momentum[i]);
        free_mat(mlp->second_momentum[i]);
    }

    free(mlp->weights);
    free(mlp->weight_deltas);
    free(mlp->first_momentum);
    free(mlp->second_momentum);


    /* Free thread-specific arrays*/
    for (int t = 0; t < NUM_THREADS; t++) {
        for (int i = 0; i <= mlp->num_hidden_layers; i++) {
            free_mat(mlp->weight_derivatives[t][i]);
            free_mat(mlp->inner_potentials[t][i]);
            free_mat(mlp->neuron_outputs[t][i]);
            free_mat(mlp->error_derivatives[t][i]);
            free_mat(mlp->activation_derivatives[t][i]);
        }

        free(mlp->inner_potentials[t]);
        free(mlp->neuron_outputs[t]);
        free(mlp->error_derivatives[t] - 1);  // :D
        free(mlp->activation_derivatives[t]);
        free(mlp->weight_derivatives[t]);
    }

    free(mlp->inner_potentials);
    free(mlp->neuron_outputs);
    free(mlp->error_derivatives);  // :D
    free(mlp->activation_derivatives);
    free(mlp->weight_derivatives);
}

void initialize_weights(MLP* mlp, int seed) {
    srand(seed);

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        int input_size = mlp->weights[i]->rows - 1;  // do not count bias input neuron
        int output_size = mlp->weights[i]->cols;

        double (*generator)(double, double) = NULL;
        double arg1 = 0.0;
        double arg2 = 0.0;

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
                mlp->weights[i]->data[j][k] = random_val;
            }
        }
    }
}

Matrix *forward_pass(MLP *mlp, Matrix *input, int thread) {
    Matrix *prev_layer = input;

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        multiply_mat(prev_layer, mlp->weights[i], mlp->inner_potentials[thread][i], false);
        mlp->activation_functions[i](mlp->inner_potentials[thread][i], mlp->neuron_outputs[thread][i]);

        prev_layer = mlp->neuron_outputs[thread][i];
    }

    return mlp->neuron_outputs[thread][mlp->num_hidden_layers];
}

/* Compute error function partial derivatives by weights */
void backpropagate(MLP *mlp, Matrix *input, Matrix *target_output, int thread) {
    /* USED WITH SIGMOID OUTPUT FUNCTION
    //compute derivatives of the error function with respect to the neuron outputs of the last layer
    //the result is transposed, because of the way the derivatives with regards to the neuron outputs
    //in other layers are computed (multiplication with the same weight matrices as in forward pass
    //just from the other side)
    Matrix *deriv_last = sub_mat(mlp->neuron_outputs[mlp->num_hidden_layers], target_output);
    Matrix *deriv_last_T = transpose_mat(deriv_last);
    free_mat(deriv_last
    mlp->activation_funs_der[mlp->num_hidden_layers](mlp->inner_potentials[mlp->num_hidden_layers],
                                                     mlp->activation_derivatives[mlp->num_hidden_layers]);
    elem_multiply_mat(deriv_last_T, mlp->activation_derivatives[mlp->num_hidden_layers],
                      mlp->error_derivatives[mlp->num_hidden_layers]
    free_mat(deriv_last_T);
    */

    int last = mlp->num_hidden_layers;

    double suma_posledna_vrstva_e_na_vnutorny_potencial = 0.0;
    for (int i = 0; i < mlp->inner_potentials[thread][last]->cols; i++) {
        suma_posledna_vrstva_e_na_vnutorny_potencial += exp(mlp->inner_potentials[thread][last]->data[0][i]);
    }

    for (int i = 0; i < target_output->cols; i++) {
        if (target_output->data[0][i] == 0) {
            mlp->error_derivatives[thread][last]->data[i][0] = 0;
            continue;
        }

        // Compute dE_k/dy_j * softmax derivation for the last layer
        // mlp->error_derivatives[last]->data[i][0] = target_output->data[0][i] / mlp->neuron_outputs[last]->data[0][i];  // dE_k/dy_j
        // mlp->error_derivatives[last]->data[i][0] *= mlp->neuron_outputs[last]->data[0][i] * (1 - mlp->neuron_outputs[last]->data[0][i]);
        mlp->error_derivatives[thread][last]->data[i][0] = (1 - mlp->neuron_outputs[thread][last]->data[0][i]);  // it's the same wtf

        // Next to last layer
        for (int j = 0; j < mlp->error_derivatives[thread][last - 1]->rows; j++) {
            double suma_s_nasobenim_vahy = 0.0;
            for (int k = 0; k < mlp->inner_potentials[thread][last]->cols; k++) {
                suma_s_nasobenim_vahy += exp(mlp->inner_potentials[thread][last]->data[0][k]) * mlp->weights[last]->data[j+1][k];
            }

            mlp->error_derivatives[thread][last - 1]->data[j][0] = mlp->weights[last]->data[j+1][i] - suma_s_nasobenim_vahy / suma_posledna_vrstva_e_na_vnutorny_potencial;
        }
    }

    mlp->activation_funs_der[last - 1](mlp->inner_potentials[thread][last - 1], mlp->activation_derivatives[thread][last - 1]);
    elem_multiply_mat(mlp->error_derivatives[thread][last - 1], mlp->activation_derivatives[thread][last - 1], mlp->error_derivatives[thread][last - 1]);

    for (int i = last - 2; i >= 0; i--) {
        multiply_mat(mlp->weights[i + 1], mlp->error_derivatives[thread][i + 1], mlp->error_derivatives[thread][i], true);
        mlp->activation_funs_der[i](mlp->inner_potentials[thread][i], mlp->activation_derivatives[thread][i]);
        elem_multiply_mat(mlp->error_derivatives[thread][i], mlp->activation_derivatives[thread][i], mlp->error_derivatives[thread][i]);
    }

    // For preceding layers extensions
    multiply_mat(mlp->weights[0], mlp->error_derivatives[thread][0], mlp->error_derivatives[thread][-1], true);

    // computing derivatives of the error function with respect to all weights
    for (int k = 0; k <= last; k++) {
        for (int i = 0; i < mlp->weight_derivatives[thread][k]->rows; i++) {
            for (int j = 0; j < mlp->weight_derivatives[thread][k]->cols; j++) {
                double grad = mlp->weight_derivatives[thread][k]->data[i][j];
                // compute derivative
                Matrix* neuron_vals = (k == 0) ? input : mlp->neuron_outputs[thread][k - 1];
                grad += mlp->error_derivatives[thread][k]->data[j][0] * neuron_vals->data[0][i];
                mlp->weight_derivatives[thread][k]->data[i][j] = grad;
            }
        }
    }
}

/* Update the weights */
void gradient_descent(MLP *mlp, double learning_rate, int batch_size, double alpha) {
    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        multiply_scalar_mat(mlp->weight_derivatives[0][k], -learning_rate / batch_size, mlp->weight_derivatives[0][k]);
        add_mat(mlp->weight_derivatives[0][k], mlp->weight_deltas[k], mlp->weight_deltas[k]);

        /* Update weights*/
        subtract_mat(mlp->weights[k], mlp->weight_deltas[k], mlp->weights[k]);

        /* Zero-out weight derivatives for the next batch */
        multiply_scalar_mat(mlp->weight_derivatives[0][k], 0, mlp->weight_derivatives[0][k]);
        /* Momentum */
        multiply_scalar_mat(mlp->weight_deltas[k], alpha, mlp->weight_deltas[k]);
    }
}

/* Update the weights using Adam algorithm */
void gradient_descent_adam(MLP *mlp, double learning_rate, int time_step, double beta1, double beta2) {
    // https://arxiv.org/abs/1412.6980
    double epsilon = 0.00000001;  // for corection of division by 0

    for (int k = 0; k <= mlp->num_hidden_layers; k++) {
        for (int i = 0; i < mlp->weights[k]->rows; i++) {
            for (int j = 0; j < mlp->weights[k]->cols; j++) {
                double der = mlp->weight_derivatives[0][k]->data[i][j];

                // Update biased first moment estimate
                double mean = beta1 * mlp->first_momentum[k]->data[i][j] + (1 - beta1) * der;
                mlp->first_momentum[k]->data[i][j] = mean;
                // Update biased second raw moment estimate
                double var = beta2 * mlp->second_momentum[k]->data[i][j] + (1 - beta2) * der * der;
                mlp->second_momentum[k]->data[i][j] = var;

                // Compute bias-corrected first moment estimate
                double mean_cor = mean / (1 -  pow(beta1, time_step));
                // Compute bias-corrected second raw moment estimate
                double var_cor = var / (1 - pow(beta2, time_step));

                // Update weights
                mlp->weights[k]->data[i][j] = mlp->weights[k]->data[i][j] + learning_rate * mean_cor / (sqrt(var_cor) + epsilon);
            }
        }

        /* Zero-out weight derivatives for the next batch */
        multiply_scalar_mat(mlp->weight_derivatives[0][k], 0, mlp->weight_derivatives[0][k]);
    }
}

/* Sum all threads into first one, zero out the other ones */
void sum_all_threads(MLP* mlp) {
    for (int t = 1; t < NUM_THREADS; t++) {
        for (int i = 0; i <= mlp->num_hidden_layers; i++) {
            add_mat(mlp->weight_derivatives[0][i], mlp->weight_derivatives[t][i], mlp->weight_derivatives[0][i]);
            multiply_scalar_mat(mlp->weight_derivatives[t][i], 0, mlp->weight_derivatives[t][i]);
        }
    }
}

void train(MLP* mlp, Samples *samples, double learning_rate, int num_batches, int batch_size, double alpha) {
    // input_data[0] must be 1
    int t = 1;
    double beta1 = 0.9;
    double beta2 = 0.999;

    omp_set_dynamic(0);     // Explicitly disable dynamic number of threads
    omp_set_num_threads(NUM_THREADS);

    unsigned int seeds[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        seeds[t] = 42 + t;
    }

    for (int batch = 0; batch < num_batches; batch++) {
        #pragma omp parallel for
        for (int i = 0; i < batch_size; i++) {
            int thread_num = omp_get_thread_num();
            int data_i = get_random_int(0, samples->num_samples - 1, &seeds[thread_num]);
            forward_pass(mlp, samples->inputs[data_i], thread_num);
            backpropagate(mlp, samples->inputs[data_i], samples->outputs[data_i], thread_num);
        }

        sum_all_threads(mlp);
        //gradient_descent(mlp, learning_rate, batch_size, alpha);
        gradient_descent_adam(mlp, learning_rate, t, beta1, beta2);
        t++;
    }
}

double test(MLP* mlp, Samples *samples, double (*metric_fun)(Matrix*, Matrix*)) {
    if (metric_fun == NULL) {
        metric_fun = &mse;  // default metric function
    }
    double res = 0.0;

    int hits = 0;

    for (int i = 0; i < samples->num_samples; i++) {
        Matrix *computed_out = forward_pass(mlp, samples->inputs[i], 0);

        // res += metric_fun(computed_out, target_data[i]);

        double max_computed = computed_out->data[0][0];
        int max_index = 0;
        int target_index = 0;
        for (int j = 0; j < samples->outputs[i]->cols; j++) {
            if (computed_out->data[0][j] > max_computed) {
                max_index = j;
                max_computed = computed_out->data[0][j];
            }

            if (samples->outputs[i]->data[0][j] == 1) {
                target_index = j;
            }
        }

        if (max_index == target_index) {
            hits++;
        } else {
            for (int j = 0; j < samples->outputs[i]->cols; j++) {
                printf("%f, ", computed_out->data[0][j]);
            }

            printf("   Target: %d\n", target_index);
        }
    }

    printf("%d / %d\n", hits, samples->num_samples);

    return res;
}
