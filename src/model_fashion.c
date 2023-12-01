#include "parse_csv.h"
#include "MLP.h"
#include "activation_functions.h"
#include <unistd.h>
#include <stdlib.h>
#include <getopt.h>

void save_weights(MLP *mlp, const char *filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("save_weights - Failed to open weights save file");
        exit(1);
    }

    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        for (int row = 0; row < mlp->weights[i]->rows; row++) {
            for (int col = 0; col < mlp->weights[i]->cols; col++) {
                fprintf(file, "%lf ", mlp->weights[i]->data[row][col]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
}

void load_weights(MLP *mlp, const char *filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("load_weights - Failed to open file");
        exit(1);
    }

    double weight;
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        for (int row = 0; row < mlp->weights[i]->rows; row++) {
            for (int col = 0; col < mlp->weights[i]->cols; col++) {
                fscanf(file, "%lf ", &weight);
                mlp->weights[i]->data[row][col] = weight;
            }
            fscanf(file, "\n");
        }
        fscanf(file, "\n");
    }
}

extern char *optarg;

void print_help() {
    printf("MLP\n");
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
    double alpha = 0.1;
    int num_batches = 2500;
    int batch_size = 256;

    char *input_weights_path = NULL;
    char *output_weights_path = NULL;

    // Parse args
    struct option longopts[] = {
        {"rate", 1, NULL, 'r'},
        {"alpha", 1, NULL, 'a'},
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
            case 'i':  // input weights
                input_weights_path = optarg;
                break;
            case 'o':  // output weights
                output_weights_path = optarg;
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

    Samples test_samples, train_samples;
    get_samples(&test_samples, test_inputs_path, test_outputs_path, 10);
    get_samples(&train_samples, train_inputs_path, train_outputs_path, 10);

    int hidden_layer_sizes[2] = {256, 64};
    func_ptr activation_funs[3] = {&ReLU, &ReLU, &softmax};
    func_ptr activation_funs_der[3] = {&ReLU_der, &ReLU_der, &softmax_der};

    MLP mlp = create_mlp(test_samples.inputs[0]->cols - 1, test_samples.outputs[0]->cols, 2,
                         hidden_layer_sizes, activation_funs, activation_funs_der);

    if (input_weights_path != NULL) {
        load_weights(&mlp, input_weights_path);
    } else {
        initialize_weights(&mlp, 42);
    }

    train(&mlp, &train_samples, learning_rate, num_batches, batch_size, alpha);

    // double test_res = test(&mlp, &test_samples, NULL);
    // printf("%f\n", test_res);
    predict(&mlp, &train_samples, "train_predictions.csv");
    predict(&mlp, &test_samples, "test_predictions.csv");

    if (output_weights_path != NULL) {
        save_weights(&mlp, output_weights_path);
    }

    // free model
    free_mlp(&mlp);

    // free samples
    free_samples(&test_samples);
    free_samples(&train_samples);
}
