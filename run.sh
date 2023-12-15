#!/bin/bash
echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network
echo "gcc -O3 -fopenmp -Wall -Wextra -lm -o mlp src/model_fashion.c src/MLP.c src/activation_functions.c src/parse_csv.c src/matrices.c"
gcc -O3 -fopenmp -Wall -Wextra -lm -o mlp src/model_fashion.c src/MLP.c src/activation_functions.c src/parse_csv.c src/matrices.c

echo "#################"
echo "     RUNNING     "
echo "#################"

echo "./mlp"
./mlp
