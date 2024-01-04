# The Percetron
- Just a perceptron, a single Perceptron capable of binary classification.
- Provides a foundational understanding of feedforward neural networks.
- Implementation includes weights, activation functions, loss functions, bias terms, and backpropagation—all from scratch.

# Multilayer Perceptron (Feed-Forward Neural Network)
- Explores the workings of larger neural networks.
- Focuses on the implementation of backpropagation within a broader network context.
- Is comprised of multiple layers of interlinked perceptrons.

# The Transformer

This project implements the Transformer model architecture in C++, following the design principles outlined in the paper "Attention is All You Need" by Vaswani et al. The Transformer model is a versatile neural network architecture widely used in natural language processing tasks.

## Overview

The C++ implementation of the Transformer model comprises several key components:

1. **Encoder:** Responsible for processing the input sequence and transforming it into continuous representations. It consists of multiple identical layers, each with two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.

2. **Decoder:** Takes the encoder output and generates predictions for the output sequence. Similar to the encoder, it contains multiple identical layers, each with three sub-layers: a masked multi-head self-attention mechanism, a multi-head attention mechanism where queries come from the decoder, and keys and values come from the encoder, and a position-wise fully connected feed-forward network.

3. **Multi-Head Attention:** Enables the model to focus on different parts of the input sequence for each position. It consists of multiple parallel attention layers, each producing different linear transformations of the input.

4. **Positional Encoding:** Since the transformer model doesn't inherently understand the order of the sequence, positional encodings are added to the input embeddings to provide the model with information about the relative or absolute position of the tokens in the sequence.

5. **Position-wise Feed-Forward Networks:** Fully connected layers applied to each position separately and identically.

## Getting Started

### Prerequisites

- C++ compiler (I used g++, it is also used in the Makefile)

### Compilation

1. Clone the repository:

   ```bash
   git clone git@github.com:ni-eminen/the-transformer-project.git
   ```

2. Build the project:

   ```bash
   cd the-transformer-project
   make
   ```

## Contributing

Pull requests and contributions are welcome. For major changes, please open an issue first to discuss the changes you would like to make.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Vaswani, A., et al. (2017). "Attention is All You Need." In Advances in Neural Information Processing Systems.

