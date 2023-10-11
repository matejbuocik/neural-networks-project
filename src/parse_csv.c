#include "parse_csv.h"
#include <ctype.h>


int parse_csv_file(const char* filename, Matrix ***ptr_to_mat_array) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(1);
    }

    int cols = 0;
    int num_matrices = 0;
    int c;

    // Count the number of columns (number of commas in one line, assume each line has the same number of columns)
    while ((c = fgetc(file)) != '\n' && c != EOF) {
        if (c == ',') {
            cols++;
        }
    }
    cols++; // Add 1 for the last column

    // Rewind the file
    rewind(file);

    // Count the number of lines in the file into num_matrices
    bool in_line = false;

    while ((c = fgetc(file)) != EOF) {
        if (isdigit(c) && !in_line) {
            num_matrices++;
            in_line = true;
        }
        if (c == '\n') {
            in_line = false;
        }
    }

    // Rewind the file
    rewind(file);

    // Allocate an array of num_matrices Matrix structures
    (*ptr_to_mat_array) = (Matrix**)malloc(num_matrices * sizeof(Matrix*));

    // Create matrices with the appropriate dimensions
    for (int i = 0; i < num_matrices; i++) {
        (*ptr_to_mat_array)[i] = create_mat(1, cols);
    }

    // Go through the file and parse all the values
    int row = 0;
    int col = 0;
    char buffer[256]; // Assuming a maximum token length of 256 characters
    memset(buffer, 0, sizeof(buffer)); // Clear the buffer

    while ((c = fgetc(file)) != EOF) {
        if (c == ',' || c == '\n') {
            // Convert the buffer to a double and store it in the matrix
            double value = atof(buffer);
            set_element((*ptr_to_mat_array)[row], 0, col, value);
            memset(buffer, 0, sizeof(buffer)); // Clear the buffer
            col++;

            if (col == cols) {
                col = 0;
                row++;
            }
        } else {
            // Append the character to the buffer
            size_t len = strlen(buffer);
            if (len < sizeof(buffer) - 1) {
                buffer[len] = (char)c;
            }
        }
    }

    fclose(file);
    return num_matrices;
}

void print_matrices(Matrix **matrix_array, int num_matrices) {
    for (int i = 0; i < num_matrices; i++) {
        printf("Matrix %d:\n", i + 1);
        for (int row = 0; row < matrix_array[i]->rows; row++) {
            for (int col = 0; col < matrix_array[i]->cols; col++) {
                printf("%lf ", get_element(matrix_array[i], row, col));
            }
            printf("\n");
        }
        printf("\n");
    }
}