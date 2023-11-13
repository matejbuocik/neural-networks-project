#ifndef CSV_PARSER
#define CSV_PARSER

#include <string.h>
#include "matrices.h"


int parse_csv_vectors(const char* filename, Matrix ***ptr_to_mat_array, int is_input);

int parse_classification_labels(const char *filename, int categories, Matrix ***ptr_to_mat_array);

void print_matrices(Matrix **matrix_array, int num_matrices);

#endif
