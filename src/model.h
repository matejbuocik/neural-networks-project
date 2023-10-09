#include "matrices.h"


typedef double (*func_ptr)(double);

/*
    Multi-layer Perceptron.
    Each array is num_hidden_layers + 1 (output layer) long.
*/
typedef struct {
    int input_size;                     /* Size of the input vector */
    int output_size;                    /* Size of the output vector */
    int num_hidden_layers;              /* Number of hidden layers */
    Matrix* weights;                    /* Array of weight matrices */
    Matrix* inner_potentials;           /* Array of inner potential vectors */
    Matrix* neuron_outputs;             /* Array of neuron output vectors */
    Matrix* error_derivatives;          /* Array of error function partial derivatives by neuron outputs vectors (transponed) */
    Matrix* activation_derivatives;     /* Array of activation function derivatives vectors (transponed) */
    Matrix* weight_derivatives;         /* Array of error function partial derivatives by weights vectors */
    func_ptr* activation_functions;     /* Array of activation functions */
    func_ptr* activation_funs_der;      /* Array of derived activation functions */
} MLP;

/* Create a MLP */
MLP create_mlp(int input_size, int output_size, int num_hidden_layers, int hidden_layer_sizes[],
               double (*activation_functions[])(double), double (*activation_derivatives[])(double));

/* Free memory used by MLP */
void free_mlp(MLP* mlp);

/* Initialize weights randomly */
void initialize_weights(MLP* mlp, int seed, double max_val, double min_val);

/* Forward pass (compute neuron outputs) */
Matrix forward_pass(MLP* mlp, Matrix input);

/* Compute derivatives during forward pass */
void compute_derivatives(MLP* mlp, Matrix target_output);

/* Set derivatives to zero */
void set_derivatives_to_zero(MLP* mlp);

/* Update weights using stochastic gradient descent */
void update_weights(MLP* mlp, double learning_rate);

/* Train the MLP using stochastic gradient descent */
void train(MLP* mlp, Matrix* input_data, Matrix* target_data, double learning_rate, int num_epochs, int batch_size, double (*error_function)(Matrix, Matrix));
