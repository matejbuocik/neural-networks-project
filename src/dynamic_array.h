#include <stdio.h>
#include <stdlib.h>

typedef struct {
    void** data;
    size_t size;
    size_t capacity;
} DynamicArray;

DynamicArray* create_dynamic_array(size_t initial_capacity);

void destroy_dynamic_array(DynamicArray* arr);

void append(DynamicArray* arr, void* value);

void* pop(DynamicArray* arr);

void* get(DynamicArray* arr, size_t index);

void set(DynamicArray* arr, size_t index, void* value);
