#ifndef CSV_PARSER
#define CSV_PARSER

#include <string.h>
#include "matrices.h"


int parse_csv_file(const char* filename, Matrix ***ptr_to_mat_array);

void print_matrices(Matrix **matrix_array, int num_matrices);

#endif
