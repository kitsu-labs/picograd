#ifndef MEMORY
#define MEMORY

#include "types.h"
#include "ops.h"
#include "nn.h"

//--------------MEMORY--------------
// TLDR: frees all temporary Values in graph
// uses a global array (temp_values) to track all temp Value objects
// these are things like Values in the forward pass of a Neuron or random Values initialized throughout
// we won't need those any more after they're forwarded, so they can be discarded
void free_temp_values(){
  for (size_t i = 0; i < temp_values.used; i++){ // check all the temp Values
    Value* val = temp_values.values[i];
    if (val->prev){ // if it has children
      free(val->prev->values); // free all the child Values
      free(val->prev); // then free the child ValueArray
    }
    free(val); // then free the temp Value
  }
  temp_values.used = 0; // we don't free temp_values.values as we will be reusing the array many times 
                        // i could free(temp_values.values) here, but then i'd have to reallocate it which seems wasteful
                        // again, this approach sucks dick, but i don't know how else to do it lol
}

// TLDR: frees all Values in graph
// uses a global array (network_params) to track all permanent Value objects
// this is mainly weights and biases that persist throughout epochs
// this fucking sucks btw, but it simplifies memory management
void free_network_params(){
  for (size_t i = 0; i < network_params.used; i++){ // check all the permanent Values
    Value* val = network_params.values[i];
    if (val->prev){ // if it has children
      free(val->prev->values); // free all the child Values
      free(val->prev); // then free the child ValueArray
    }
    free(val); // then free the temp Value
 }
  free_temp_values();
  free(network_params.values); // this is only called at the end of the program, so we can free the array
  free(temp_values.values); // we should also free the temp_values since we're at the end of the program
}

// TLDR: frees a ValueArray and it's memory
void free_array(ValueArray* arr){
  if (arr == NULL) return;

  if (arr->values){ // if it has Values in it
    free(arr->values); // free the array of Value pointers
  }
  arr->used = 0; // reset used slots
  arr->size = 0; // reset array size
  free(arr); // free the ValueArray itself
}

// TLDR: frees a Matrix2D and it's memory
void free_matrix2d(Matrix2D* mat){
  if (mat == NULL) return;

  for (size_t i = 0; i < mat->rows; i++){
    free(mat->data[i]); // free memory for each row
  }

  free(mat->data); // free memory for array of row pointers
  free(mat); // free the matrix2d itself
}

// TLDR: frees a Neuron and it's memory
void free_neuron(Neuron* neuron){
  if (neuron == NULL) return;

  for (size_t i = 0; i < neuron->nin; i++){
    if (neuron->w[i] != NULL){
      neuron->w[i] = NULL; // just set to NULL, free_network_params will free it
    }
  }

  free(neuron->w); // free the array of weight pointers
  neuron->b = NULL; // just set to NULL, free_network_params will free it
  free(neuron); // free Neuron itself
}

// TLDR: frees all parameters associated to normalization and it's memory
void free_normalization(Normalization* norm){
  if (norm == NULL) return;

  free(norm->gamma); // free gamma
  free(norm->beta); // free beta
  if (norm->type == BATCH){
    free(norm->running_mean); // free the running mean array
    free(norm->running_var); // free the running variance array
  }
  free(norm); // free the normalization Layer itself
}

// TLDR: frees a Layer and it's memory
void free_layer(Layer* layer){
  if (layer == NULL) return;

  for (size_t i = 0; i < layer->nout; i++){
    if (layer->neurons[i] != NULL){ // if there's Neurons stored in a Layer
      free_neuron(layer->neurons[i]); // free that Neuron in the Layer
    }
  }

  free(layer->neurons); // free the array of Neuron pointers

  if (layer->norm != NULL){ // also make sure to handle the normalization Layer
    free_normalization(layer->norm);
  }

  free(layer); // free the Layer itself
}

// TLDR: frees a MLP and it's memory
void free_mlp(MLP* mlp){
  if (mlp == NULL) return;

  for (size_t i = 0; i < mlp->num_layers; i++){
    if (mlp->layers[i] != NULL){
      free_layer(mlp->layers[i]); // free each Layer in the MLP
    }
  }
  
  free(mlp->layers); // free the array of Layer pointers
  free(mlp); // free the MLP itself
}

// TLDR: frees the initial setup of the network
void free_network_config(NetworkConfig* config){
  if (config == NULL) return;
  
  free(config->layer_sizes);
  free(config->layer_activations);
  free(config->layer_normalizations);
  free(config);
}

void free_reshape(double** reshaped, size_t num_samples) {
  if (reshaped == NULL) return;

  for (size_t i = 0; i < num_samples; i++){ // free memory allocated for each row
    free(reshaped[i]);
  }

  free(reshaped); // free the array of pointers
}

#endif // !MEMORY
