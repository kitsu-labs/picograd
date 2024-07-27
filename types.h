#ifndef TYPES
#define TYPES

//--------------ENUMS--------------
typedef enum{
  NO_ACTIVATION = -1,
  RELU,
  SIGMOID,
  TANH
} ActivationType;

typedef enum{
  MSE,
  CROSS_ENTROPY,
} LossType;

typedef enum{
  NO_NORMALIZATION = -1,
  BATCH,
  LAYER
} NormalizationType;

//--------------STRUCTS--------------
typedef struct Value Value;
typedef struct ValueArray ValueArray;
typedef struct Neuron Neuron;
typedef struct Layer Layer; 
typedef struct MLP MLP; 
typedef struct Matrix2D Matrix2D; 
typedef struct TrainingParams TrainingParams;
typedef struct NetworkConfig NetworkConfig;
typedef struct Normalization Normalization;
typedef struct MVP MVP;

// TLDR: structure of Value node in graph with:
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

// TLDR: dynamic array of Value nodes, kinda sorta like a 1D Tensor
// values -> array of pointers to Values
// used -> currently filled slots in the array
// size -> total size of the array 
struct ValueArray{
  Value** values;
  size_t used;
  size_t size;
};

// TLDR: a single Neuron in the network/Layer/MLP
// w -> array of Value weights
// b -> a singular bias Value
// nin -> total number of inputs to the Neuron
struct Neuron{
  Value** w;
  Value* b;
  size_t nin;
};

// TLDR: a single Layer in the network/MLP
// neurons -> array of Neurons in a Layer
// nin -> number of inputs received by a Layer
// nout -> number of output Neurons produced by a Layer
// activation -> activation function associated with this Layer
// normalziation -> normalization type applied to this Layer or Batch
// norm -> stores all normalization params
struct Layer{
  Neuron** neurons;
  size_t nin;
  size_t nout;
  ActivationType activation;
  NormalizationType normalization;
  Normalization* norm;
};

// TLDR: an MLP
// layers -> array of Layers
// num_layers -> number of Layers in MLP
struct MLP{
  Layer** layers;
  size_t num_layers;
};

// TLDR: a 2D Matrix
// data -> array of pointers pointing to an array of pointers to Values
// rows -> number of rows in the Matrix
// cols -> number of cols in the Matrix
struct Matrix2D{
  Value*** data;
  size_t rows;
  size_t cols;
};

// TLDR: contains all the parameters needed at training time
// num_samples -> how many samples are you providing?
// num_inputs -> how many inputs are in each sample?
// num_outputs -> how many outputs should the MLP calculate?
// num_epochs -> how long should we train the MLP?
// learning_rate -> what's the learning rate?
// lr_decay -> how fast should we taper off the learning rate?
// loss_type -> what kind of loss should be used?
struct TrainingParams{
  size_t num_samples;
  size_t num_inputs;
  size_t num_outputs;
  size_t num_epochs;
  double learning_rate;
  double lr_decay;
  LossType loss_type;
};

// TLDR: network configuration to make initializing the network easier
struct NetworkConfig{
  size_t num_layers;
  size_t* layer_sizes;
  ActivationType* layer_activations;
  NormalizationType* layer_normalizations;
};

// TLDR: contains all the parameters associated with normalization 
// num_features -> number of features / inputs to be normalized
// gamma -> learnable scaling factors for each feature
// beta -> learnable offset values for each feature
// running_mean -> running average of the features
// running_var -> running variance of the features
// eps -> small value added to variance to prevent div by zero
// momentum -> momentum used in the running mean and variance calculation
struct Normalization{
  NormalizationType type;
  size_t num_features;
  Value** gamma;
  Value** beta;
  double* running_mean;
  double* running_var;
  double eps;
  double momentum;
};

// TLDR: mean and variance calculation for a batch
// mean -> mean of the batch
// var -> variance of the batch
// unbiased_var -> unbiased variance of the batch (batch_size-1)
struct MVP {
  double mean;
  double var;
  double unbiased_var;
};

// circular dependencies because i suck
extern ValueArray network_params;
extern ValueArray temp_values;
Value* create_value(double, bool);
Value* apply_activation(Value*, ActivationType);
Normalization* create_normalization(NormalizationType, size_t, double, double);
ValueArray* apply_normalization(Layer*, ValueArray*, bool);
void insert_array(ValueArray*, Value*);
void free_array(ValueArray*);
Matrix2D* create_matrix2d(size_t, size_t);
Matrix2D* valuearray_to_matrix2d(ValueArray*, size_t, size_t);
ValueArray* matrix2d_to_valuearray(Matrix2D*);
void free_matrix2d(Matrix2D*);

#endif // !TYPES
