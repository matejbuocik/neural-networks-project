#ifndef CSV_PARSER
#define CSV_PARSER

#include <string.h>
#include "matrices.h"


/* Print matrices from `matrix_array` */
void print_matrices(Matrix **matrix_array, int num_matrices);

/* Data for the network */
typedef struct {
    Matrix **inputs;    /* Input vectors */
    Matrix **outputs;   /* Output vectors */
    int num_samples;    /* Number of samples */
    int num_classes;    /* Number of output classes */
} Samples;

/* Get samples from inputs and outputs files */
void get_samples(Samples *samples, char *inputs_path, char *outputs_path, int num_classes);

/* Free the Samples struct */
void free_samples(Samples *samples);

#endif
