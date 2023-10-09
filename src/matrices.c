#include "matrices.h"
#include <assert.h>


Matrix create_mat(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;

    mat.data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat.data[i] = (double*)malloc(cols * sizeof(double));
    }

    return mat;
}

void free_mat(Matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
}

void set_element(Matrix* mat, int row, int col, double value) {
    mat->data[row][col] = value;
}

double get_element(const Matrix* mat, int row, int col) {
    return mat->data[row][col];
}

void add_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out) {
    int rows = mat1->rows;
    int cols = mat1->cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
        }
    }
}

Matrix add_mat(const Matrix* mat1, const Matrix* mat2) {
    int rows = mat1->rows;
    int cols = mat1->cols;

    Matrix result = create_mat(rows, cols);

    add_mat_with_out(mat1, mat2, &result);

    return result;
}

void sub_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out) {
    int rows = mat1->rows;
    int cols = mat1->cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out->data[i][j] = mat1->data[i][j] - mat2->data[i][j];
        }
    }
}

Matrix sub_mat(const Matrix* mat1, const Matrix* mat2) {
    int rows = mat1->rows;
    int cols = mat1->cols;

    Matrix result = create_mat(rows, cols);

    sub_mat_with_out(mat1, mat2, &result);

    return result;
}

void mult_mat_with_out(const Matrix* mat1, const Matrix* mat2, const Matrix* out) {
    int rows1 = mat1->rows;
    int cols1 = mat1->cols;
    int cols2 = mat2->cols;

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols1; k++) {
                sum += mat1->data[i][k] * mat2->data[k][j];
            }
            out->data[i][j] = sum;
        }
    }
}

Matrix mult_mat(const Matrix* mat1, const Matrix* mat2) {
    int rows1 = mat1->rows;
    int cols2 = mat2->cols;

    Matrix result = create_mat(rows1, cols2);

    mult_mat_with_out(mat1, mat2, &result);

    return result;
}

void apply_to_mat_with_out(const Matrix *mat, const Matrix *out, double (*fun)(double)) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            out->data[i][j] = fun(mat->data[i][j]);
        }
    }
}

Matrix matrix_from_array(int rows, int cols, double* array) {
    Matrix mat = create_mat(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat.data[i][j] = array[i * cols + j];
        }
    }

    return mat;
}
