#include "parse_csv.h"
#include "MLP.h"
#include "activation_functions.h"
#include <unistd.h>
#include <stdlib.h>
#include <getopt.h>

extern char *optarg;

void print_help() {
    printf("MLP\n");
    printf("   -v --vectors\t\tInput vectors file (default data/xor_vectors.csv)\n");
    printf("   -l --labels\t\tOutput labels file (default data/xor_labels.csv)\n");
    printf("   -r --rate\t\tLearning rate (default 1.0)\n");
    printf("   -n --num-batches\tNumber of batches (default 1000000)\n");
    printf("   -s --batch-size\tSize of a batch (default 1)\n");
    printf("\n   -h --help\t\tShow this help\n");
}

int main(int argc, char *argv[]) {
    // Set default values
    char *train_inputs_path = "data/fashion_mnist_train_vectors.csv";
    char *train_outputs_path = "data/fashion_mnist_train_labels.csv";
    char *test_inputs_path = "data/fashion_mnist_test_vectors.csv";
    char *test_outputs_path = "data/fashion_mnist_test_labels.csv";
    double learning_rate = 0.001;
    int num_batches = 10000;
    int batch_size = 32;

    // Parse args
    struct option longopts[] = {
        {"vectors", 1, NULL, 'v'},
        {"labels", 1, NULL, 'l'},
        {"rate", 1, NULL, 'r'},
        {"num-batches", 1, NULL, 'n'},
        {"batch-size", 1, NULL, 's'},
        {"help", 0, NULL, 'h'},
        {0, 0, 0, 0}
    };
    int opt;
    char *endptr;
    while ((opt = getopt_long(argc, argv, "v:l:r:n:s:h", longopts, NULL)) != -1) {
        switch (opt) {
            case 'v':  // vectors
                train_inputs_path = optarg;
                break;
            case 'l':  // labels
                train_outputs_path = optarg;
                break;
            case 'r':  // learning rate
                learning_rate = strtod(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "learning_rate: Parse error\n");
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

    Matrix** train_inputs;
    int train_in_n = parse_csv_vectors(train_inputs_path, &train_inputs, 1);

    Matrix** train_outputs;
    int train_out_n = parse_classification_labels(train_outputs_path, 10, &train_outputs);

    if (train_in_n != train_out_n) {
        fprintf(stderr, "Input count is different than output count\n");
        exit(1);
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

    int hidden_layer_sizes[3] = {64, 16, 10};
    func_ptr activation_funs[4] = {&ReLU, &ReLU, &ReLU, &softmax};
    func_ptr activation_funs_der[4] = {&ReLU_der, &ReLU_der, &ReLU_der, &softmax_der};

    MLP mlp = create_mlp(train_inputs[0]->cols - 1, train_outputs[0]->cols, 3, hidden_layer_sizes,
                         activation_funs, activation_funs_der);

    initialize_weights(&mlp, 42);

    train(&mlp, train_in_n, train_inputs, train_outputs, learning_rate, num_batches, batch_size);

    double test_res = test(&mlp, test_in_n, test_inputs, test_outputs, NULL);

    printf("%f\n", test_res);

    // free model
    free_mlp(&mlp);

    // free input and output arrays
    for (int i = 0; i < train_in_n; i++) {
        free_mat(train_inputs[i]);
        free_mat(train_outputs[i]);
    }
    free(train_inputs);
    free(train_outputs);

    for (int i = 0; i < test_in_n; i++) {
        free_mat(test_inputs[i]);
        free_mat(test_outputs[i]);
    }
    free(test_inputs);
    free(test_outputs);
}
