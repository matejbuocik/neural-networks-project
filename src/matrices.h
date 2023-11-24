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

/* Multiply two matrices, store result in `out`*/
void multiply_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* out, bool back_prop);

/* Apply `fun` to every element of the matrix, store result in `out`*/
void apply_func_mat(const Matrix *mat, const Matrix *out, double (*fun)(double), bool transpose_out);

/* Multiply matrix by scalar, store result in `out` */
void multiply_scalar_mat(const Matrix* mat, double fact, const Matrix* out);

/* Subtract two matrices, store result in `out`*/
void subtract_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* out);

void add_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* out);

/* Multiply two matrices, only by elements, store result in `result` */
void elem_multiply_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* result);

/* Subtract two matrices, create a new one for result*/
Matrix *sub_mat(const Matrix* mat1, const Matrix* mat2);

/* Sum of all elements of the matrix */
double sum_mat(Matrix *mat);

/* Transpose matrix, return a new one */
Matrix *transpose_mat(const Matrix* input);

#endif
