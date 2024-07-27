#ifndef NN
#define NN

#include <string.h>
#include "types.h"
#include "ops.h"
#include "matrix.h"

//--------------INITIALIZATION--------------
// TLDR: allocates memory for and creates new Value
// node is added to network_params or temp_values to make cleanup easier later
Value* create_value(double scalar, bool is_temp){
  Value* node = (Value*)malloc(sizeof(Value));
  node->data = scalar;
  node->grad = 0.0;
  node->backward = NULL;
  node->prev = (ValueArray*)malloc(sizeof(ValueArray));
  initialize_array(node->prev, 1);

  if (is_temp){ // kind of wish i made this "is_permanent" instead, but cba to refactor it all
    insert_array(&temp_values, node); // neuron forward pass, losses, eps values, etc
  } else{
    insert_array(&network_params, node); // neuron weights / biases, things that need gradients
  }

  return node;
}

// TLDR: allocates memory for and creates new Neuron
Neuron* create_neuron(size_t nin){
  Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
  neuron->w = (Value**)malloc(nin * sizeof(Value)); // allocate array of nin Value pointers (1 weight per 1 input)
  neuron->b = create_value(0.0, false); // bias term, false because this is a learnable parameter
  neuron->nin = nin; // number of inputs to this neuron

  for (size_t i = 0; i < nin; i++){ // generate a weight for each input
    neuron->w[i] = create_value((double)rand() / RAND_MAX * 2.0 - 1.0, false); // -1 to 1
    // neuron->w[i] = create_value((double)rand() / RAND_MAX, false); // 0 to 1
  }

  return neuron;
}

// TLDR: allocates memory for and creates new Layer of Neurons
Layer* create_layer(size_t nin, size_t nout, ActivationType activation, NormalizationType normalization){
  if (nin == 0 || nout == 0){
    fprintf(stderr, "Error: Layer dimensions must be greater than 0\n");
    return NULL;
  }

  Layer* layer = (Layer*)malloc(sizeof(Layer));
  layer->neurons = (Neuron**)malloc(nout * sizeof(Neuron)); // allocate array of nout Neuron pointers
  layer->nin = nin; // number of inputs each Neuron receives
  layer->nout = nout; // number of outputs representing Neurons in this Layer
  layer->activation = activation; // links activation function chosen to this Layer
  layer->normalization = normalization; // links normalization to this Layer
  layer->norm = NULL; // normalization could be NO_NORMALIZATION, so we don't set this yet

  // loops through nout, creates Neuron with nin inputs, inserts into layer->neurons
  for (size_t i = 0; i < nout; i++){
    layer->neurons[i] = create_neuron(nin);
  }

  // will this layer be normalized via BatchNorm or LayerNorm before the activation?
  if (normalization != NO_NORMALIZATION){
    layer->norm = create_normalization(normalization, nin, 1e-5, 0.01); // default values, probably should allow customization
  }

  return layer;
}

// TLDR: allocates memory for and creates an MLP
MLP* create_mlp(size_t num_layers, size_t* n_neurons, ActivationType* layer_activations, NormalizationType* layer_normalizations){
  if (num_layers < 2){
    fprintf(stderr, "Error: MLP must have at least 2 layers (input and output)\n");
    return NULL;
  }
  if (n_neurons == NULL || layer_activations == NULL){
    fprintf(stderr, "Error: n_neurons and layer_activations must not be NULL\n");
    return NULL;
  }

  MLP* mlp = (MLP*)malloc(sizeof(MLP));
  mlp->num_layers = num_layers-1; // subtract one because num_layers includes the input, so num_layers == len(n_neurons_in_each_layer_not_including_input_lol)
  mlp->layers = (Layer**)malloc((num_layers-1) * sizeof(Layer*)); // allocate array of num_layers-1

  for (size_t i = 0; i < num_layers-1; i++){ // create each Layer and apply normalization/activation
    mlp->layers[i] = create_layer(n_neurons[i], n_neurons[i+1], layer_activations[i], layer_normalizations[i]); // connect each Layer together (a->output = b->input)
  }

  return mlp;
}

// TLDR: creates the normalization Layer for BatchNorm or LayerNorm
Normalization* create_normalization(NormalizationType type, size_t num_features, double eps, double momentum){
  Normalization* norm = (Normalization*)malloc(sizeof(Normalization));
  norm->type = type;
  norm->num_features = num_features;
  norm->eps = eps;
  norm->momentum = momentum;
  norm->gamma = (Value**)malloc(num_features * sizeof(Value*));
  norm->beta = (Value**)malloc(num_features * sizeof(Value*));
  
  for (size_t i = 0; i < num_features; i++){
    norm->gamma[i] = create_value(1.0, false); // initialized to have zero impact, false because it's a learnable param
    norm->beta[i] = create_value(0.0, false); // initialized to have zero impact, false because it's a learnable param
  }

  // BatchNorm
  if (type == BATCH){
    norm->running_mean = (double*)calloc(num_features, sizeof(double)); // initialize num_features means and set all to zero
    norm->running_var = (double*)malloc(num_features * sizeof(double)); // initialize num_features vars
    for (size_t i = 0; i < num_features; i++){
      norm->running_var[i] = 1.0; // and set all of them to 1.0
    }
  }

  return norm;
}

// TLDR: serves the purpose of setting up the initialization of a network
NetworkConfig* create_network_config(size_t num_layers, size_t* layer_sizes, ActivationType* layer_activations, NormalizationType* layer_normalizations){
  if (num_layers < 2 || layer_sizes == NULL || layer_activations == NULL || layer_normalizations == NULL){
    fprintf(stderr, "Invalid input parameters for create_network_config\n");
    return NULL;
  }

  NetworkConfig* config = malloc(sizeof(NetworkConfig));
  if (config == NULL) {
    fprintf(stderr, "Memory allocation failed for NetworkConfig\n");
    return NULL;
  }

  config->num_layers = num_layers;
  config->layer_sizes = malloc(num_layers * sizeof(size_t));
  config->layer_activations = malloc((num_layers - 1) * sizeof(ActivationType));
  config->layer_normalizations = malloc((num_layers - 1) * sizeof(NormalizationType));

  if (config->layer_sizes == NULL || config->layer_activations == NULL || config->layer_normalizations == NULL) {
    fprintf(stderr, "Memory allocation failed for NetworkConfig members\n");
    free(config->layer_sizes);
    free(config->layer_activations);
    free(config->layer_normalizations);
    free(config);
    return NULL;
  }

  memcpy(config->layer_sizes, layer_sizes, num_layers * sizeof(size_t));
  memcpy(config->layer_activations, layer_activations, (num_layers - 1) * sizeof(ActivationType));
  memcpy(config->layer_normalizations, layer_normalizations, (num_layers - 1) * sizeof(NormalizationType));
  return config;
}

MLP* initialize_mlp(NetworkConfig* config){
  if (config == NULL){
    fprintf(stderr, "Invalid NetworkConfig in initialize_mlp\n");
    return NULL;
  }

  MLP* mlp = create_mlp(config->num_layers, config->layer_sizes, config->layer_activations, config->layer_normalizations);
  return mlp;
}

//--------------FORWARD--------------
// TLDR: performs a forward pass for a Neuron
Value* neuron_forward(Neuron* neuron, ValueArray* inputs){
  if (neuron == NULL || inputs == NULL){
    fprintf(stderr, "Error: Neuron or inputs are NULL in neuron_forward\n");
    return NULL;
  }
  if (neuron->nin != inputs->used){
    fprintf(stderr, "Error: Mismatch between neuron inputs and provided inputs\n");
    return NULL;
  }

  Value* output = create_value(0.0, true); // keeps track of running sum (wi*xi)

  for (size_t i = 0; i < neuron->nin; i++){
    Value* product = v_mul(neuron->w[i], inputs->values[i]); // element-wise mul wi*xi
    output = v_add(output, product); // add that product to the output
  }

  output = v_add(output, neuron->b); // adds singular bias term after all (wi*xi)
  return output; // returns a single Neuron's output: the sum of all weighted inputs + bias
}

// TLDR: performs a forward pass for a Layer
ValueArray* layer_forward(Layer* layer, ValueArray* inputs, bool is_training){
  if (layer == NULL || inputs == NULL){
    fprintf(stderr, "Error: Layer or inputs are NULL in layer_forward\n");
    fprintf(stderr, "Layer: %p, Inputs: %p\n", (void*)layer, (void*)inputs);
    return NULL;
  }
  if (layer->nin != inputs->used){
    fprintf(stderr, "Error: Mismatch between layer inputs and provided inputs\n");
    fprintf(stderr, "Layer->nin: %zu, Inputs->used: %zu\n", layer->nin, inputs->used);
    return NULL;
  }

  ValueArray* normalized_inputs = apply_normalization(layer, inputs, is_training); // normalizes inputs (if toggled, otherwise this ends up = to inputs)
  ValueArray* output = (ValueArray*)malloc(sizeof(ValueArray));
  initialize_array(output, layer->nout);

  for (size_t i = 0; i < layer->nout; i++){
    insert_array(output, apply_activation(neuron_forward(layer->neurons[i], normalized_inputs), layer->activation)); // forward pass and activate Neuron
  }

  if (normalized_inputs != inputs){
    free_array(normalized_inputs);
  }

  return output;
}

// TLDR: performs a forward pass for the MLP
ValueArray* mlp_forward(MLP* mlp, ValueArray* inputs, bool is_training){
  if (mlp == NULL || inputs == NULL){
    fprintf(stderr, "Error: MLP or inputs are NULL in mlp_forward\n");
    return NULL;
  }
  if (mlp->layers[0]->nin != inputs->used){
    fprintf(stderr, "Error: Mismatch between MLP inputs and provided inputs\n");
    return NULL;
  }

  ValueArray* curr = inputs; // initialized to input array, tracks current input for each Layer
  ValueArray* output = NULL; // initialized to NULL, tracks each Layer's output

  for (size_t i = 0; i < mlp->num_layers; i++){
    output = layer_forward(mlp->layers[i], curr, is_training); // forward pass each Layer
    if (i > 0) free_array(curr); // to avoid memory leaks, we need to free the previous Layer's output / this Layer's input
    curr = output; // output of a->forward becomes b->input
  }

  return output; // returns the final Layer's output: an array of forwarded Neurons in the last Layer, aka the output of entire MLP
}

// TLDR: performs a forward pass for the BatchNorm Layer
// this is most likely an imperfect implementation (as is the rest of this project)
ValueArray* batch_norm_forward(Normalization* norm, Matrix2D* inputs, bool is_training){
  if (inputs == NULL){
    fprintf(stderr, "Error: Input Matrix2D is NULL in batch_norm_forward\n");
    return NULL;
  }
  if (inputs->cols != norm->num_features){
    fprintf(stderr, "Error: Input features (%zu) don't match normalization features (%zu)\n", inputs->cols, norm->num_features);
    return NULL;
  }

  Matrix2D* output = create_matrix2d(inputs->rows, inputs->cols); // shape is important here, so a Matrix makes sense

  for (size_t j = 0; j < norm->num_features; j++){
    double input = inputs->data[0][j]->data;
    
    if (is_training){ // update the running mean and variance during training for use at inference
      norm->running_mean[j] = (1 - norm->momentum) * norm->running_mean[j] + norm->momentum * input;
      double diff = input - norm->running_mean[j];
      norm->running_var[j] = (1 - norm->momentum) * norm->running_var[j] + norm->momentum * diff * diff;
    }

    // normalization
    double mean_to_use = is_training ? norm->running_mean[j] : norm->running_mean[j]; // training and inference means are different
    double var_to_use = is_training ? norm->running_var[j] : norm->running_var[j]; // training and inference variances are different
    Value* centered = v_sub(inputs->data[0][j], create_value(mean_to_use, true));
    Value* variance = v_add(create_value(var_to_use, true), create_value(norm->eps, true));
    Value* inv_std = v_pow(variance, create_value(-0.5, true));
    Value* normalized = v_mul(centered, inv_std);
    
    // scale and shift
    output->data[0][j] = v_add(v_mul(normalized, norm->gamma[j]), norm->beta[j]); // gamma and beta are learnable parameters
  }

  ValueArray* result = matrix2d_to_valuearray(output); // convert matrix back to a ValueArray
  free_matrix2d(output); // we don't need the Matrix representation anymore
  return result;
}

#endif // !NN
