p:
	g++ ./perceptron/src/main.cpp ./perceptron/src/perceptron.cpp ./utils/utils.cpp -g -o ./perceptron/build/a.out
	./perceptron/build/a.out

mlp:
	g++ ./multilayer-perceptron/src/main.cpp ./utils/utils.cpp -g -o ./multilayer-perceptron/build/a.out
	./multilayer-perceptron/build/a.out