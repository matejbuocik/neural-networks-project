#include <stdio.h>
#include <stdlib.h>


// Define a matrix structure
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

Matrix create_mat(int rows, int cols);

void free_mat(Matrix* mat);

void set_element(Matrix* mat, int row, int col, double value);

double get_element(const Matrix* mat, int row, int col);

void add_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out);

Matrix add_mat(const Matrix* mat1, const Matrix* mat2);

void sub_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out);

Matrix sub_mat(const Matrix* mat1, const Matrix* mat2);

void mult_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out);

Matrix mult_mat(const Matrix* mat1, const Matrix* mat2);

void apply_to_mat_with_out(const Matrix *mat, const Matrix *out, double (*fun)(double));

Matrix mat_from_array(int rows, int cols, double* array);
