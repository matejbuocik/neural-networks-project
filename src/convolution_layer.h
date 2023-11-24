#ifndef CONVOLUTIONAL_LAYER
#define CONVOLUTIONAL_LAYER

# include "MLP.h"


typedef struct {
    /* Set on start */
    int neurons_in_feat_map;
    int input_width;
    int input_height;
    int num_feature_maps;
    int kernel_size; // square kernel assumed
    int pool_size;
    func_ptr activation_function;
    func_ptr activation_fun_der;

    /* Forward pass */
    Matrix** weights;                   /* Array of weight matrices */
    Matrix** inner_potentials;          /* Array of inner potential vectors */
    Matrix** neuron_outputs;            /* Array of neuron output vectors */
    Matrix* output;
    Matrix* potential;
    Matrix* error_der;

    /* Backpropagation */
    Matrix** error_derivatives;         /* Array of error function partial derivatives by neuron outputs (transponed) */
    Matrix** activation_derivatives;    /* Array of activation function derivatives vectors (transponed) */
    Matrix** weight_derivatives;        /* Array of error function partial derivatives by weights vectors */
    Matrix** weight_deltas;
    Matrix** first_momentum;
    Matrix** second_momentum;

    MLP* mlp;
} ConLayer;


ConLayer create_con_layer(int input_width, int input_height, int num_feature_maps, int kernel_size, int pool_size,
                          func_ptr activation_function, func_ptr activation_fun_der, MLP* mlp);

void free_con_layer(ConLayer* conl);

void init_weights(ConLayer* conl, int seed);

Matrix *fwd_pass(ConLayer *conl, Matrix *input);

void backprop(ConLayer *conl, Matrix *input, Matrix *target_output);

void multiply_ders_by(ConLayer* conl, double factor);

void grad_des(ConLayer* conl, double learning_rate, int batch_size, double alpha);

void train_con(ConLayer *conl, int num_samples, Matrix *input_data[], Matrix *target_data[],
           double learning_rate, int num_batches, int batch_size, double alpha);

double test_con(ConLayer *conl, int num_samples, Matrix *input_data[], Matrix *target_data[],
            double (*metric_fun)(Matrix*, Matrix*));

#endif
