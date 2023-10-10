#ifndef SIMPLE_MATRIXES
#define SIMPLE_MATRIXES

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


/* Matrix structure */
typedef struct {
    int rows;           /* Number of rows */
    int cols;           /* Number of columns */
    double** data;      /* 2d array, data[row][col]*/
} Matrix;

/* Create a new Matrix*/
Matrix *create_mat(int rows, int cols);

/* Free memory used by Matrix*/
void free_mat(Matrix* mat);

/* Set element at a given position to value */
void set_element(Matrix* mat, int row, int col, double value);

/* Get element at a given position */
double get_element(const Matrix* mat, int row, int col);

/* Add two matrices, store result in out*/
void add_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out);

/* Add two matrices, create a new one for result*/
Matrix *add_mat(const Matrix* mat1, const Matrix* mat2);

/* Subtract two matrices, store result in out*/
void sub_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out);

/* Subtract two matrices, create a new one for result*/
Matrix *sub_mat(const Matrix* mat1, const Matrix* mat2);

/* Multiply two matrices, store result in out*/
void mult_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out);

/* Multiply two matrices, create a new one for result*/
Matrix *mult_mat(const Matrix* mat1, const Matrix* mat2);

void mult_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* result);

void mult_scal_with_out(const Matrix* mat1, double fact, const Matrix* result);

void apply_to_mat_with_out(const Matrix *mat, const Matrix *out, double (*fun)(double), bool transpose_out);

double sum_mat(Matrix *mat);

Matrix *transpose_mat(const Matrix* input);

Matrix *mat_from_array(int rows, int cols, double* array);

Matrix *copy_mat(Matrix* mat);

#endif
