#ifndef MULTI_LAYER_PERCEPTRON
#define MULTI_LAYER_PERCEPTRON

#include "matrices.h"
#include "activation_functions.h"


typedef void (*func_ptr)(const Matrix *, const Matrix *);

/*
    Multi-layer Perceptron.
    Each array is num_hidden_layers + 1 (output layer) long.
*/
typedef struct {
    /* Set on start */
    int num_hidden_layers;              /* Number of hidden layers */
    func_ptr* activation_functions;     /* Array of activation functions */
    func_ptr* activation_funs_der;      /* Array of derived activation functions */

    /* Forward pass */
    Matrix** weights;                   /* Array of weight matrices */
    Matrix** inner_potentials;          /* Array of inner potential vectors */
    Matrix** neuron_outputs;            /* Array of neuron output vectors */

    /* Backpropagation */
    Matrix** error_derivatives;         /* Array of error function partial derivatives by neuron outputs (transponed) */
    Matrix** activation_derivatives;    /* Array of activation function derivatives vectors (transponed) */
    Matrix** weight_derivatives;        /* Array of error function partial derivatives by weights vectors */

    Matrix** weight_deltas;             /* Momentum of weight derivatives */

    // Adam
    Matrix** first_momentum;
    Matrix** second_momentum;

} MLP;

/* Create a MLP */
MLP create_mlp(int input_size, int output_size, int num_hidden_layers, int hidden_layer_sizes[],
               func_ptr activation_functions[], func_ptr activation_funs_der[]);

/* Free memory used by MLP */
void free_mlp(MLP* mlp);

/* Initialize weights */
void initialize_weights(MLP* mlp, int seed);

/* Compute neuron outputs */
Matrix *forward_pass(MLP* mlp, Matrix *input);

/* Compute error function partial derivatives by weights */
void backpropagate(MLP* mlp, Matrix *input, Matrix *target_output);

/* Update the weights */
void gradient_descent(MLP *mlp, double learning_rate, int batch_size, double aplha);

/* Update the weights using Adam algorithm */
void gradient_descent_adam(MLP *mlp, double learning_rate, int time_step, double beta1, double beta2);

/* Train the MLP */
void train(MLP* mlp, int num_samples, Matrix *input_data[], Matrix *target_data[],
           double learning_rate, int num_batches, int batch_size, double alpha);

/* Test the model on `input_data` using `target_data` with `metric_fun` (if NULL, use mean square error)*/
double test(MLP* mlp, int num_samples, Matrix *input_data[], Matrix *target_data[],
            double (*metric_fun)(Matrix*, Matrix*));

#endif
