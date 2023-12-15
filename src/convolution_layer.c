#include "convolution_layer.h"
#include "parse_csv.h"

/* CONVOLUTION LAYER IS WIP */

ConLayer create_con_layer(int input_width, int input_height, int num_feature_maps, int kernel_size, int pool_size,
                          func_ptr activation_function, func_ptr activation_fun_der, MLP* mlp) {//CHANGE
    ConLayer convl;

    int neurons_in_feat_map = (input_width - kernel_size + 1) * (input_height - kernel_size + 1);

    convl.neurons_in_feat_map = neurons_in_feat_map;
    convl.input_width = input_width;
    convl.input_height = input_height;
    convl.num_feature_maps = num_feature_maps;
    convl.kernel_size = kernel_size;
    convl.pool_size = pool_size;
    convl.activation_function = activation_function;
    convl.activation_fun_der = activation_fun_der;
    convl.mlp = mlp;

    // allocate memory for arrays
    convl.weights                 = (Matrix**) malloc((num_feature_maps) * sizeof(Matrix*));
    convl.inner_potentials        = (Matrix**) malloc((num_feature_maps) * sizeof(Matrix*));
    convl.neuron_outputs          = (Matrix**) malloc((num_feature_maps) * sizeof(Matrix*));
    convl.error_derivatives       = (Matrix**) malloc((num_feature_maps) * sizeof(Matrix*));
    convl.weight_derivatives      = (Matrix**) malloc((num_feature_maps) * sizeof(Matrix*));
    convl.weight_deltas           = (Matrix**) malloc((num_feature_maps) * sizeof(Matrix*));
    convl.first_momentum          = (Matrix**) malloc((num_feature_maps) * sizeof(Matrix*));
    convl.second_momentum         = (Matrix**) malloc((num_feature_maps) * sizeof(Matrix*));

    // initialize allocated arrays
    for (int i = 0; i < num_feature_maps; i++) {
        convl.weights[i] = create_mat(kernel_size, kernel_size);
        convl.weight_derivatives[i] = create_mat(kernel_size, kernel_size);
        convl.weight_deltas[i] = create_mat(kernel_size, kernel_size);
        convl.first_momentum[i] = create_mat(kernel_size, kernel_size);
        convl.second_momentum[i] = create_mat(kernel_size, kernel_size);

        convl.inner_potentials[i] = create_mat(1, neurons_in_feat_map);
        convl.neuron_outputs[i] = create_mat(1, neurons_in_feat_map);
        convl.error_derivatives[i] = create_mat(neurons_in_feat_map, 1);
    }

    convl.output = create_mat(1, neurons_in_feat_map / (pool_size * pool_size) * num_feature_maps + 1);
    convl.output->data[0][0] = 1;

    convl.potential = create_mat(1, neurons_in_feat_map / (pool_size * pool_size) * num_feature_maps);
    convl.error_der = create_mat(neurons_in_feat_map / (pool_size * pool_size) * num_feature_maps, 1);

    return convl;
}

void free_con_layer(ConLayer* conl) {
    for (int i = 0; i < conl->num_feature_maps; i++) {
        free_mat(conl->weights[i]);
        free_mat(conl->weight_derivatives[i]);
        free_mat(conl->weight_deltas[i]);
        free_mat(conl->first_momentum[i]);
        free_mat(conl->second_momentum[i]);

        free_mat(conl->inner_potentials[i]);
        free_mat(conl->neuron_outputs[i]);

        free_mat(conl->error_derivatives[i]);

    }

    free(conl->weights);
    free(conl->inner_potentials);
    free(conl->neuron_outputs);

    free(conl->error_derivatives);
    free(conl->weight_derivatives);
    free(conl->weight_deltas);
    free(conl->first_momentum);
    free(conl->second_momentum);

    free(conl->output);
    free(conl->potential);
    free(conl->error_der);
}


void init_weights(ConLayer* conl, int seed) {
    srand(seed);

    for (int i = 0; i < conl->num_feature_maps; i++) {
        for (int j = 0; j < conl->weights[i]->rows; j++) {
            for (int k = 0; k < conl->weights[i]->cols; k++) {
                double random_val = generate_normal_random(0.0, 1.0);
                conl->weights[i]->data[j][k] = random_val;
            }
        }
    }

    initialize_weights(conl->mlp, seed);
}


Matrix *fwd_pass(ConLayer *conl, Matrix *input) {
    // compute feature maps
    int f_map_w = conl->input_width - conl->kernel_size + 1;
    int f_map_h = conl->input_height - conl->kernel_size + 1;

    for (int map_i = 0; map_i < conl->num_feature_maps; map_i++) {
        for (int out_x = 0; out_x < f_map_w; out_x++) {
            for (int out_y = 0; out_y < f_map_h; out_y++) {
                double inner_potential = 0.0;

                for (int x_offset = 0; x_offset < conl->kernel_size; x_offset++) {
                    for (int y_offset = 0; y_offset < conl->kernel_size; y_offset++) {
                        int in_x = out_x + x_offset;
                        int in_y = out_y + y_offset;

                        int input_index = in_y * conl->input_width + in_x + 1; // ignore initial 1
                        double weight = conl->weights[map_i]->data[y_offset][x_offset];
                        double input_data = input->data[0][input_index];

                        inner_potential += weight * input_data;
                    }
                }

                conl->inner_potentials[map_i]->data[0][out_y * f_map_w + out_x] = inner_potential;
            }
        }

        conl->activation_function(conl->inner_potentials[map_i], conl->neuron_outputs[map_i]);
    }

    // max pooling
    int width = f_map_w / conl->pool_size;
    int height = f_map_h / conl->pool_size;

    for (int map_i = 0; map_i < conl->num_feature_maps; map_i++) {
        for (int out_x = 0; out_x < width; out_x++) {
            for (int out_y = 0; out_y < height; out_y++) {
                int out_index = map_i * width * height + out_y * width + out_x;

                // det initial max value
                int max_i = out_y * conl->pool_size * f_map_w + out_x * conl->pool_size;
                double max = conl->neuron_outputs[map_i]->data[0][max_i];
                conl->error_derivatives[map_i]->data[max_i][0] = out_index;

                for (int x_offset = 0; x_offset < conl->pool_size; x_offset++) {
                    for (int y_offset = 0; y_offset < conl->pool_size; y_offset++) {
                        int in_x = out_x * conl->pool_size + x_offset;
                        int in_y = out_y * conl->pool_size + y_offset;
                        int in_1d = in_y * f_map_w + in_x;
                        // -1 indicates, that this value was not the max
                        if (in_1d != max_i) { // do not overwrite the value set earlier
                            conl->error_derivatives[map_i]->data[in_1d][0] = -1;
                        }

                        if (conl->neuron_outputs[map_i]->data[0][in_1d] > max) {
                            // neuron value corrresponding to max_i was not he maximum,
                            // so the derivative will be 0, this is indicated by the -1 index
                            conl->error_derivatives[map_i]->data[max_i][0] = -1;
                            max_i = in_1d;
                            max = conl->neuron_outputs[map_i]->data[0][max_i];
                            conl->error_derivatives[map_i]->data[max_i][0] = out_index;
                        }

                    }
                }
                // TODO: max pool based on iner potentials, apply activation later
                conl->output->data[0][map_i * width * height + out_y * width + out_x + 1] = max;
                conl->potential->data[0][map_i * width * height + out_y * width + out_x] = conl->inner_potentials[map_i]->data[0][max_i];
            }
        }
    }


    return forward_pass(conl->mlp, conl->output);
}


void backprop(ConLayer *conl, Matrix *input, Matrix *target_output) {
    backpropagate(conl->mlp, conl->output, target_output);

//print_matrices(&(conl->mlp->error_derivatives[-1]), 1);

//print_matrices(&(conl->potential), 1);

    conl->activation_fun_der(conl->potential, conl->error_der);

//print_matrices(&(conl->error_der), 1);

    elem_multiply_mat(conl->error_der, conl->mlp->error_derivatives[-1], conl->error_der);

//print_matrices(&(conl->error_der), 1);

    for (int mask_i = 0; mask_i < conl->num_feature_maps; mask_i++) {
        // pull derivatives through the max pooling layer
        for (int i = 0; i < conl->neurons_in_feat_map; i++) {
            int index = conl->error_derivatives[mask_i]->data[i][0];
//print_matrices(&(conl->error_derivatives[mask_i]), 1);
            if (index == -1) {
                conl->error_derivatives[mask_i]->data[i][0] = 0;
            } else {
                conl->error_derivatives[mask_i]->data[i][0] = conl->error_der->data[index][0];
            }
        }

        // derivatives of the error function with respect to weights
        for (int off_x = 0; off_x < conl->kernel_size; off_x++) {
            for (int off_y = 0; off_y < conl->kernel_size; off_y++) {
                int width = conl->input_width - conl->kernel_size + 1;
                int height = conl->input_height - conl->kernel_size + 1;

                double der = 0.0;

                for (int err_x = 0; err_x < width; err_x++) {
                    for (int err_y = 0; err_y < height;  err_y++) {
                        int error_index = err_y * width + err_x;

                        int in_x = err_x + off_x;
                        int in_y = err_y + off_y;
                        // account for 1 at position 0 in the input data vector
                        int in_index = in_y * conl->input_width + in_x + 1;

                        der += conl->error_derivatives[mask_i]->data[error_index][0] * input->data[0][in_index];
                    }
                }

                conl->weight_derivatives[mask_i]->data[off_y][off_x] += der;
            }
        }
    }

//print_matrices(conl->weight_derivatives, conl->num_feature_maps);
}


void multiply_ders_by(ConLayer* conl, double factor) {
    for (int k = 0; k < conl->num_feature_maps; k++) {
        multiply_scalar_mat(conl->weight_derivatives[k], factor, conl->weight_derivatives[k]);
    }
}

void multiply_delts_by(ConLayer* conl, double factor) {
    for (int k = 0; k < conl->num_feature_maps; k++) {
        multiply_scalar_mat(conl->weight_deltas[k], factor, conl->weight_deltas[k]);
    }
}


void grad_des(ConLayer* conl, double learning_rate, int batch_size, double alpha) {
    // TODO use better techniques (adaptive learning rate, momentum, ...)
    // TODO: get rid of the * 10 with better techniques :D
    for (int k = 0; k < conl->num_feature_maps; k++) {
        multiply_scalar_mat(conl->weight_derivatives[k], -(learning_rate * 10) / batch_size, conl->weight_derivatives[k]);
        add_mat(conl->weight_derivatives[k], conl->weight_deltas[k], conl->weight_deltas[k]);

        subtract_mat(conl->weights[k], conl->weight_deltas[k], conl->weights[k]);
    }

    gradient_descent(conl->mlp, learning_rate, batch_size, alpha);

    multiply_ders_by(conl, 0.0);
    multiply_delts_by(conl, alpha);
}

void grad_des_adam(ConLayer* conl, double learning_rate, int time_step, double beta1, double beta2) {
    // https://arxiv.org/abs/1412.6980
    double epsilon = 0.00000001;  // for corection of division by 0

    // TODO: exchange multiple applications of the functions on matricies by one cycle
    for (int k = 0; k < conl->num_feature_maps; k++) {
        for (int i = 0; i < conl->weights[k]->rows; i++) {
            for (int j = 0; j < conl->weights[k]->cols; j++) {
                double der = conl->weight_derivatives[k]->data[i][j];

                // Update biased first moment estimate
                double mean = beta1 * conl->first_momentum[k]->data[i][j] + (1 - beta1) * der;
                conl->first_momentum[k]->data[i][j] = mean;
                // Update biased second raw moment estimate
                double var = beta2 * conl->second_momentum[k]->data[i][j] + (1 - beta2) * der * der;
                conl->second_momentum[k]->data[i][j] = var;

                // Compute bias-corrected first moment estimate
                double mean_cor = mean / (1 -  pow(beta1, time_step));
                // Compute bias-corrected second raw moment estimate
                double var_cor = var / (1 - pow(beta2, time_step));

                // Update weights
                conl->weights[k]->data[i][j] = conl->weights[k]->data[i][j] + learning_rate * mean_cor / (sqrt(var_cor) + epsilon);
            }
        }

    }

    multiply_ders_by(conl, 0);

    gradient_descent_adam(conl->mlp, learning_rate, time_step, beta1, beta2);
}


void train_con(ConLayer *conl, int num_samples, Matrix *input_data[], Matrix *target_data[],
           double learning_rate, int num_batches, int batch_size, double alpha) {
    // input_data[0] must be 1
    for (int k = 0; k < conl->num_feature_maps; k++) {
        multiply_scalar_mat(conl->weight_deltas[k], 0.0, conl->weight_deltas[k]);
        multiply_scalar_mat(conl->weight_derivatives[k], 0.0, conl->weight_derivatives[k]);
        multiply_scalar_mat(conl->first_momentum[k], 0.0, conl->first_momentum[k]);
        multiply_scalar_mat(conl->second_momentum[k], 0.0, conl->second_momentum[k]);
    }
    // init mlp as well
    // init to 0, TODO: do within allocation ?
    for (int k = 0; k <= conl->mlp->num_hidden_layers; k++) {
        multiply_scalar_mat(conl->mlp->weight_deltas[k], 0.0, conl->mlp->weight_deltas[k]);
        multiply_scalar_mat(conl->mlp->weight_derivatives[k], 0.0, conl->mlp->weight_derivatives[k]);
        multiply_scalar_mat(conl->mlp->first_momentum[k], 0.0, conl->mlp->first_momentum[k]);
        multiply_scalar_mat(conl->mlp->second_momentum[k], 0.0, conl->mlp->second_momentum[k]);
    }

    int t = 1;
    double beta1 = 0.9;
    double beta2 = 0.999;

    for (int batch = 0; batch < num_batches; batch++) {
        for (int i = 0; i < batch_size; i++) {
            // TODO spustit na batch_size procesoroch naraz
            int data_i = get_random_int(0, num_samples - 1);

            fwd_pass(conl, input_data[data_i]);
            backprop(conl, input_data[data_i], target_data[data_i]);
// print_matrices(mlp->weight_derivatives, mlp->num_hidden_layers + 1);
        }

        //grad_des(conl, learning_rate, batch_size, alpha);
        grad_des_adam(conl, learning_rate, t, beta1, beta2);

        t++;

// print_matrices(mlp->weights, mlp->num_hidden_layers + 1);
    }
}


double test_con(ConLayer *conl, int num_samples, Matrix *input_data[], Matrix *target_data[],
            double (*metric_fun)(Matrix*, Matrix*)) {
    if (metric_fun == NULL) {
        metric_fun = &mse;  // default metric function
    }
    double res = 0.0;

    int hits = 0;

    for (int i = 0; i < num_samples; i++) {
        Matrix *computed_out = fwd_pass(conl, input_data[i]);

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
