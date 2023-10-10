#include "parse_csv.h"
#include "MLP.h"
#include "activation_functions.h"


int main() {
    const char *path_inputs = "/home/janto/pv021-project/data/xor_vectors.csv";
    Matrix** inputs_array;
    int in_n = parse_csv_file(path_inputs, &inputs_array);

    const char *path_outputs = "/home/janto/pv021-project/data/xor_labels.csv";
    Matrix** outputs_array;  
    int out_n = parse_csv_file(path_outputs, &outputs_array);

    if (in_n != out_n) {
        printf("Input count is different from output count");
    }

    //print_matrices(inputs_array, in_n);
    //print_matrices(outputs_array, in_n);

    int hidden_layer_sizes[1] = {4};
    func_ptr activation_funs[2] = {&ReLU, &sigmoid};
    func_ptr activation_funs_der[2] = {&ReLU_der, &sigmoid_der};

    MLP mlp = create_mlp(inputs_array[0]->cols, outputs_array[0]->cols, 1, hidden_layer_sizes,
                         activation_funs, activation_funs_der);

    initialize_weights(&mlp, 42, -0.001, 0.001);

    train(&mlp, in_n, inputs_array, outputs_array, 0.1, 10000000, 10);

    double test_res = test(&mlp, in_n, inputs_array, outputs_array, NULL);

    printf("%f\n", test_res);

    // free input and output arrays
    for (int i = 0; i < in_n; i++) {
        free_mat(inputs_array[i]);
        free_mat(outputs_array[i]);
    }
    free(inputs_array);
    free(outputs_array);
}