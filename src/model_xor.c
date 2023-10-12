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
    char *path_inputs = "data/xor_vectors.csv";
    char *path_outputs = "data/xor_labels.csv";
    double learning_rate = 1;
    int num_batches = 1000000;
    int batch_size = 1;

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
                path_inputs = optarg;
                break;
            case 'l':  // labels
                path_outputs = optarg;
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

    Matrix** inputs_array;
    int in_n = parse_csv_file(path_inputs, &inputs_array);

    Matrix** outputs_array;
    int out_n = parse_csv_file(path_outputs, &outputs_array);

    if (in_n != out_n) {
        fprintf(stderr, "Input count is different than output count\n");
        exit(1);
    }

    //print_matrices(inputs_array, in_n);
    //print_matrices(outputs_array, in_n);

    int hidden_layer_sizes[1] = {4};
    func_ptr activation_funs[2] = {&sigmoid, &sigmoid};
    func_ptr activation_funs_der[2] = {&sigmoid_der, &sigmoid_der};

    MLP mlp = create_mlp(inputs_array[0]->cols, outputs_array[0]->cols, 1, hidden_layer_sizes,
                         activation_funs, activation_funs_der);

    initialize_weights(&mlp, 42, -10, 10);

    train(&mlp, in_n, inputs_array, outputs_array, learning_rate, num_batches, batch_size);

    double test_res = test(&mlp, in_n, inputs_array, outputs_array, NULL);

    printf("%f\n", test_res);

    // free model
    free_mlp(&mlp);

    // free input and output arrays
    for (int i = 0; i < in_n; i++) {
        free_mat(inputs_array[i]);
        free_mat(outputs_array[i]);
    }
    free(inputs_array);
    free(outputs_array);
}
