# picograd - an autograd engine in c

an implementation of an automatic differentiation (autograd) engine in less than 1,000 lines of c. it provides a foundation for building and training simple neural networks from scratch, supporting various layer types, activation functions, and optimizations. heavily inspired from andrej karpathy's micrograd.

everything in picograd is built around the `Value` struct. each instance represents a single node in the computational graph which forms the basis for backpropagation during training. each node has an associated backward function which then allows it to know how to calculate gradients with respect to its inputs.

these values represent things like weights, biases, activations, and more, but they can also represent intermediate computations throughout the network for things like forward passes, loss calculations, etc. this allows for values with different purposes to be managed separately and allows for impermanent values to be freed from memory once they're no longer needed.  

```c
// TLDR: structure of Value node in graph
// data -> scalar value (1.0, 2.0, etc)
// grad -> gradient calculated during backprop
// backward -> pointer that stores derivative function associated with operation
// prev -> ValueArray of "child" Values
//         forward: these are nodes before the current node in the forward pass
//         backward: these are nodes after the current node in the backward pass

struct Value{
  double data; 
  double grad;  
  void (*backward)(Value*); 
  ValueArray* prev; 
};

Value* create_value(double scalar, bool is_temp){
  Value* node = (Value*)malloc(sizeof(Value));
  node->data = scalar;
  node->grad = 0.0;
  node->backward = NULL;
  node->prev = (ValueArray*)malloc(sizeof(ValueArray));
  initialize_array(node->prev, 1);

  if (is_temp){
    insert_array(&temp_values, node); // neuron forward pass, losses, eps values, etc
  } else{
    insert_array(&network_params, node); // neuron weights / biases, things that need gradients
  }

  return node;

```

## features

- **automatic differentiation**: 
  - implements backpropagation for 10 operations.
- **basic neural network components**:
  - neurons
  - layers
  - multi-layer perceptron (mlp)
- **activation functions**:
  - relu
  - sigmoid
  - tanh
- **normalization**:
  - batch normalization
- **loss functions**:
  - mean squared error (mse)
  - binary cross-entropy
- **optimization**:
  - stochastic gradient descent (sgd)
  - learning rate decay
  - l2 regularization
- **dynamic memory management**: 
  - a custom (poorly built) implementation for creating and freeing various structures.

## project structure

- `types.h`: defines the core data structures and enums used throughout the project.
- `ops.h`: implements basic operations and their corresponding gradients.
- `nn.h`: contains neural network component implementations (neurons, layers, mlp, normalization).
- `matrix.h`: provides matrix operations for up to three dimensions.
- `memory.h`: manages memory allocation and deallocation for the project.
- `train.h`: implements the training loop and associated functions.
- `micrograd.h`: main header file that includes all other headers.

## example usage

here's a basic approach of how i would use picograd to train a simple mlp to solve XOR:

```c
#include "../picograd.h"
// tracks permanent Value objects that need gradients (think weights and biases)
ValueArray network_params;
// tracks temporary Value objects that can be routinely flushed (think intermediate results during a forward pass)
ValueArray temp_values;

// EXAMPLE:
// XOR logic gate
// input: 4 XOR states
// output: their expected outputs
#define BATCH_SIZE 4 // how many samples are you providing?
#define NUM_INPUTS 2 // how many inputs are in each sample?
#define NUM_HIDDEN 16 // how many Neurons are in each hidden layer?
#define NUM_OUTPUTS 1 // how many outputs should the MLP calculate?
#define NUM_EPOCHS 1000 // how long should we train the MLP?
#define LEARNING_RATE 1e-1 // what's the learning rate?
#define LEARNING_RATE_DECAY 0 // do you want a learning rate decay? (i.e. 1e-3)
#define LOSS_TYPE CROSS_ENTROPY // loss

int main(){
  initialize_array(&network_params, 1); // for memory management, this needs to be initialized
  initialize_array(&temp_values, 1); // for memory management, this needs to be initialized

  // create input and output data
  double r_xor_inputs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double xor_outputs[4] = {0, 1, 1, 0};
  double** xor_inputs = reshape_input(r_xor_inputs, BATCH_SIZE, NUM_INPUTS, 1, 'd');

  // create network configuration
  size_t num_layers = 3;
  size_t layer_sizes[] = {NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS};
  ActivationType layer_activations[] = {RELU, SIGMOID}; // relu: input->hidden, sigmoid: hidden->output
  NormalizationType layer_normalizations[] = {-1, -1}; // no batch normalization needed
  NetworkConfig* config = create_network_config(num_layers, layer_sizes, layer_activations, layer_normalizations);
  MLP* n = initialize_mlp(config);

  print_network_info("Created Network Configuration", 
                     config->num_layers, 
                     config->layer_sizes, 
                     config->layer_activations, 
                     config->layer_normalizations, 
                     NULL, 
                     n);

  free_network_config(config);

  // training parameters
  TrainingParams params = {
    .num_samples = BATCH_SIZE,
    .num_inputs = NUM_INPUTS,
    .num_outputs = NUM_OUTPUTS,
    .num_epochs = NUM_EPOCHS,
    .learning_rate = LEARNING_RATE,
    .lr_decay = LEARNING_RATE_DECAY,
    .loss_type = LOSS_TYPE
  };

  // train the model
  train_mlp(n, xor_inputs, xor_outputs, &params);

  // test the model
  printf("\nTesting the model:\n");
  for (int i = 0; i < 4; i++){
    ValueArray* input = array_to_value_array(xor_inputs[i], 2);
    ValueArray* output = mlp_forward(n, input, false);
    printf("Input: (%f, %f), Output: %f\n", xor_inputs[i][0], xor_inputs[i][1], output->values[0]->data);
    free_array(input);
    free_array(output);
  }

  // Epoch 1000, Loss: 0.001823

  // Testing the model:
  // Input: (0.000000, 0.000000), Output: 0.003383
  // Input: (0.000000, 1.000000), Output: 0.998622
  // Input: (1.000000, 0.000000), Output: 0.998600
  // Input: (1.000000, 1.000000), Output: 0.001030
  
  free_mlp(n);
  free_network_params();

  return 0;

}
```

you can try out the other examples inside the examples folder by running something like:

```
gcc -O2 -o xor xor.c -lm
gcc -O2 -o binary binary.c -lm
gcc -O2 -o sequence sequence.c -lm
```

## some things to consider

this is my first project in c and was meant to serve as a tutorial to the language. prior to starting picograd, i saw c as a very intimidating language that i could barely write fizz-buzz in; however, after building all this out, i actually feel somewhat competent when using it (with a bit of help from claude at least). as frustrating as it was resolving segfaults and memory leaks, i now actually enjoy writing c and i believe this project made me a better programmer and "ml engineer." 

on top of that, i am much more confident in my knowledge of the foundation of neural networks after having to implement nearly everything from scratch in a language i am not fluent in whatsoever. it was hard, but i enjoy doing difficult things.

with that being said, this is clearly not a battle-tested engine, nor was it ever meant to be! i would not recommend using it as there are plenty of better options out there. there's a million different things i could continue to implement, improve, or refactor, and i may do so as time goes on. the main goal of this was to learn more about c and strengthen my knowledge on the fundamentals of neural networks, and i believe i accomplished both of those things.