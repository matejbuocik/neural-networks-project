#include "matrices.h"
#include <assert.h>


Matrix *create_mat(int rows, int cols) {
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;

    mat->data = (double**)malloc(rows * sizeof(double*));
    for (int row = 0; row < rows; row++) {
        mat->data[row] = (double*)calloc(cols, sizeof(double));
    }

    return mat;
}

void free_mat(Matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }

    free(mat->data);
    free(mat);
}

void set_element(Matrix* mat, int row, int col, double value) {
    mat->data[row][col] = value;
}

double get_element(const Matrix* mat, int row, int col) {
    return mat->data[row][col];
}

void multiply_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* out, bool back_prop) {
    int rows1 = mat1->rows;
    int cols1 = mat1->cols;
    int cols2 = mat2->cols;

    int offset = 0;
    if (back_prop) {
        // Skip biases row, used during backpropagation
        offset = 1;
    }

    for (int i = 0; i < rows1 - offset; i++) {
        for (int j = 0; j < cols2; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols1; k++) {
                sum += mat1->data[i + offset][k] * mat2->data[k][j];
            }
            out->data[i][j] = sum;
        }
    }
}

void apply_func_mat(const Matrix *mat, const Matrix *out, double (*fun)(double), bool transpose_out) {
    if (transpose_out) {
        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) {
                out->data[j][i] = fun(mat->data[i][j]);
            }
        }
    } else {
        int offset = 0;
        if (mat->cols + 1 == out->cols) {
            // applying activation function to get neuron output in forward pass
            offset = 1;
            out->data[0][0] = 1;
        }

        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) {
                out->data[i][j + offset] = fun(mat->data[i][j]);
            }
        }
    }
}

void multiply_scalar_mat(const Matrix* mat1, double fact, const Matrix* result) {
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            result->data[i][j] = mat1->data[i][j] * fact;
        }
    }
}
// TODO: maybe use macro for the repeating cycles
void subtract_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* out) {
    int rows = mat1->rows;
    int cols = mat1->cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out->data[i][j] = mat1->data[i][j] - mat2->data[i][j];
        }
    }
}

void add_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* out) {
    int rows = mat1->rows;
    int cols = mat1->cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
        }
    }
}

void elem_multiply_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* result) {
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            result->data[i][j] = mat1->data[i][j] * mat2->data[i][j];
        }
    }
}

void elem_divide_mat(const Matrix* mat1, const Matrix* mat2, const Matrix* result) {
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            result->data[i][j] = mat1->data[i][j] / mat2->data[i][j];
        }
    }
}

void add_scalar_mat(const Matrix* mat1, double member, const Matrix* out) {
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            out->data[i][j] = mat1->data[i][j] + member;
        }
    }
}

Matrix *sub_mat(const Matrix* mat1, const Matrix* mat2) {
    int rows = mat1->rows;
    int cols = mat1->cols;

    Matrix *result = create_mat(rows, cols);

    subtract_mat(mat1, mat2, result);

    return result;
}

double sum_mat(Matrix *mat) {
    double sum = 0.0;

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            sum += mat->data[i][j];
        }
    }

    return sum;
}

Matrix *transpose_mat(const Matrix* input) {
    Matrix *result = create_mat(input->cols, input->rows);

    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            result->data[j][i] = input->data[i][j];
        }
    }

    return result;
}

Matrix *mat_from_array(int rows, int cols, double* array) {
    Matrix *mat = create_mat(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat->data[i][j] = array[i * cols + j];
        }
    }

    return mat;
}

Matrix *copy_mat(Matrix* mat) {
    Matrix *result = create_mat(mat->rows, mat->cols);

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result->data[i][j] = mat->data[i][j];
        }
    }

    return result;
}
