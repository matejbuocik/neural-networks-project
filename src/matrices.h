#include <stdio.h>
#include <stdlib.h>


// Define a matrix structure
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

// Function to create a new matrix
Matrix create_matrix(int rows, int cols);

// Function to free memory used by a matrix
void free_matrix(Matrix* mat);

// Function to set the value of a specific element in the matrix
void set_element(Matrix* mat, int row, int col, double value);

// Function to get the value of a specific element in the matrix
double get_element(const Matrix* mat, int row, int col);

// Function to add two matrices
Matrix add_matrices(const Matrix* mat1, const Matrix* mat2);

// Function to multiply two matrices
Matrix multiply_matrices(const Matrix* mat1, const Matrix* mat2);

// Function to create a matrix from a 2D array
Matrix matrix_from_array(int rows, int cols, double* array);
