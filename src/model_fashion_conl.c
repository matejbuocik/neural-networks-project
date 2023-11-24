#include "parse_csv.h"
#include "convolution_layer.h"
#include "activation_functions.h"
#include <unistd.h>
#include <stdlib.h>
#include <getopt.h>


extern char *optarg;

void print_help() {
    printf("CMLP\n");
    printf("   -r --rate\t\tLearning rate (default 0.01)\n");
    printf("   -n --num-batches\tNumber of batches (default 10000)\n");
    printf("   -s --batch-size\tSize of a batch (default 16)\n\n");

    printf("   -i --input-weights\tLoad weights from file\n");
    printf("   -o --output-weights\tSave weights to file\n\n");

    printf("   -h --help\t\tShow this help\n");
}

int main(int argc, char *argv[]) {
    // Set default values
    char *train_inputs_path = "data/fashion_mnist_train_vectors.csv";
    char *train_outputs_path = "data/fashion_mnist_train_labels.csv";
    char *test_inputs_path = "data/fashion_mnist_test_vectors.csv";
    char *test_outputs_path = "data/fashion_mnist_test_labels.csv";

    double learning_rate = 0.001;
    double alpha = 0.95;
    int num_batches = 37500;
    int batch_size = 16;


    // Parse args
    struct option longopts[] = {
        {"rate", 1, NULL, 'r'},
        {"momentum", 1, NULL, 'a'},
        {"num-batches", 1, NULL, 'n'},
        {"batch-size", 1, NULL, 's'},
        {"input-weights", 1, NULL, 'i'},
        {"output-weights", 1, NULL, 'o'},
        {"help", 0, NULL, 'h'},
        {0, 0, 0, 0}
    };
    int opt;
    char *endptr;
    while ((opt = getopt_long(argc, argv, "r:a:n:s:i:o:h", longopts, NULL)) != -1) {
        switch (opt) {
            case 'r':  // learning rate
                learning_rate = strtod(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "learning_rate: Parse error\n");
                    exit(1);
                }
                break;
            case 'a':  // alpha for momentum
                alpha = strtod(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "alpha: Parse error\n");
                    exit(1);
                }
                break;
            case 'n':  // number of batches
                num_batches = strtol(optarg, &endptr, 10);
                if (endptr == optarg) {
                    fprintf(stderr, "num_batches: Parse error\n");
                    exit(1);
                }
                break;
            case 's':  // batch size
                batch_size = strtol(optarg, &endptr, 10);
                if (endptr == optarg) {
                    fprintf(stderr, "batch_size: Parse error\n");
                    exit(1);
                }
                break;
            case 'h':  // help
                print_help();
                exit(0);
            case '?':
                // Invalid option or missing argument, print help
                print_help();
                exit(1);
        }
    }

    Matrix** test_inputs;
    int test_in_n = parse_csv_vectors(test_inputs_path, &test_inputs, 1);

    Matrix** test_outputs;
    int test_out_n = parse_classification_labels(test_outputs_path, 10, &test_outputs);

    if (test_in_n != test_out_n) {
        fprintf(stderr, "Input count is different than output count\n");
        exit(1);
    }

    //print_matrices(inputs_array, in_n);
    //print_matrices(outputs_array, in_n);

    int hidden_layer_sizes[2] = {256, 64};
    func_ptr activation_funs[3] = {&ReLU, &ReLU, &softmax};
    func_ptr activation_funs_der[3] = {&ReLU_der, &ReLU_der, &softmax_der};

    MLP mlp = create_mlp(16 * 12 * 12, test_outputs[0]->cols, 2, hidden_layer_sizes,
                         activation_funs, activation_funs_der);

    ConLayer conl = create_con_layer(28, 28, 16, 5, 2, ReLU, ReLU_der, &mlp);


    Matrix** train_inputs;
    int train_in_n = parse_csv_vectors(train_inputs_path, &train_inputs, 1);

    Matrix** train_outputs;
    int train_out_n = parse_classification_labels(train_outputs_path, 10, &train_outputs);

    if (train_in_n != train_out_n) {
        fprintf(stderr, "Input count is different than output count\n");
        exit(1);
    }

    init_weights(&conl, 42);

    train_con(&conl, train_in_n, train_inputs, train_outputs, learning_rate, num_batches, batch_size, alpha);

print_matrices(conl.weights, conl.num_feature_maps);

    // free train inputs and outputs
    for (int i = 0; i < train_in_n; i++) {
        free_mat(train_inputs[i]);
        free_mat(train_outputs[i]);
    }
    free(train_inputs);
    free(train_outputs);


    double test_res = test_con(&conl, test_in_n, test_inputs, test_outputs, NULL);
    printf("%f\n", test_res);


    // free model
    free_mlp(&mlp);

    free_con_layer(&conl);

    // free test inputs and outputs
    for (int i = 0; i < test_in_n; i++) {
        free_mat(test_inputs[i]);
        free_mat(test_outputs[i]);
    }
    free(test_inputs);
    free(test_outputs);
}
