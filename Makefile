all: mlp

mlp: src/model_fashion.c src/MLP.c src/activation_functions.c src/parse_csv.c src/matrices.c
	gcc -Wall -Wextra -lm -o mlp $^

.PHONY: all
