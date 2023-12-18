P_SRC = ./perceptron/src
UTILS = ./utils
P_DIR = ./perceptron
P_BUILD = ${P_DIR}/build
P_X = ${P_BUILD}/a.out

p:
	g++ -c ${P_SRC}/perceptron.cpp
	g++ -c ${UTILS}/utils.cpp
	g++ -c ${P_SRC}/main.cpp
	g++ -o ${P_X} main.o utils.o perceptron.o
	${P_X}

mlp:
	g++ -o ./multilayer-perceptron/build/a.out ./multilayer-perceptron/src/main.cpp ./utils/utils.cpp
	./multilayer-perceptron/build/a.out