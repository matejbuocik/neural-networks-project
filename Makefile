all: mlp

mlp: src/model_fashion.c src/MLP.c src/MLP.h src/activation_functions.c src/parse_csv.c src/matrices.c
	gcc -O3 -fopenmp -Wall -Wextra -lm -o mlp $^

con: src/model_fashion_conl.c src/MLP.c src/activation_functions.c src/parse_csv.c src/matrices.c src/convolution_layer.c
	gcc -O3 -fopenmp -Wall -Wextra -lm -o con $^

.PHONY: all
