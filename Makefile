P_SRC = ./perceptron/src
UTILS = ./utils
P_DIR = ./perceptron
P_BUILD = ${P_DIR}/build
P_X = ${P_BUILD}/a.out

MLP_DIR = ./multilayer-perceptron
MLP_SRC = ${MLP_DIR}/src
MLP_BUILD = ${MLP_DIR}/build
MLP_X = ${MLP_BUILD}/a.out

p:
	g++ -c ${P_SRC}/perceptron.cpp
	g++ -c ${UTILS}/utils.cpp
	g++ -c ${P_SRC}/main.cpp
	g++ -o ${P_X} main.o utils.o perceptron.o
	${P_X}

mlp:
	g++ -I ./utils -g -c ${UTILS}/LinearAlgebra.cpp
	g++ -I ./utils -g -c ${MLP_SRC}/MultilayerPerceptron.cpp
	g++ -I ./utils -g -c ${UTILS}/utils.cpp
	g++ -I ./utils -g -c ${MLP_SRC}/main.cpp
	g++ -I ./utils -o ${MLP_X} -g main.o utils.o MultilayerPerceptron.o LinearAlgebra.o
	${MLP_X}

dmlp:
	gdb ${MLP_X}