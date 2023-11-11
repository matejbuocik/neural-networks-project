#include "dynamic_array.h"


DynamicArray* create_dynamic_array(size_t initial_capacity) {
    DynamicArray* arr = (DynamicArray*)malloc(sizeof(DynamicArray));
    if (!arr) {
        perror("Failed to create dynamic array");
        exit(1);
    }

    arr->data = (void**)malloc(initial_capacity * sizeof(void*));
    if (!arr->data) {
        perror("Failed to allocate memory for dynamic array data");
        exit(1);
    }

    arr->size = 0;
    arr->capacity = initial_capacity;

    return arr;
}

void destroy_dynamic_array(DynamicArray* arr) {
    free(arr->data);
    free(arr);
}

void append(DynamicArray* arr, void* value) {
    if (arr->size == arr->capacity) {
        // Double the capacity if the array is full
        size_t new_capacity = arr->capacity * 2;
        arr->data = (void**)realloc(arr->data, new_capacity * sizeof(void*));
        if (!arr->data) {
            perror("Failed to resize dynamic array");
            exit(1);
        }
        arr->capacity = new_capacity;
    }

    arr->data[arr->size++] = value;
}

void* pop(DynamicArray* arr) {
    if (arr->size == 0) {
        return NULL; // Nothing to pop
    }

    void* popped_value = arr->data[arr->size - 1];
    arr->size--;

    return popped_value;
}

void* get(DynamicArray* arr, size_t index) {
    if (index >= arr->size) {
        return NULL; // Index out of bounds
    }

    return arr->data[index];
}

void set(DynamicArray* arr, size_t index, void* value) {
    if (index >= arr->size) {
        return; // Index out of bounds
    }

    arr->data[index] = value;
}
