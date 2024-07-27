#include "../picograd.h"
ValueArray network_params;
ValueArray temp_values;

// EXAMPLE:
// XOR logic gate
// input: 4 XOR states
// output: their expected outputs
#define BATCH_SIZE 4 // how many samples are you providing?
#define NUM_INPUTS 2 // how many inputs are in each sample?
#define NUM_HIDDEN 16 // how many Neurons are in each hidden layer?
#define NUM_OUTPUTS 1 // how many outputs should the MLP calculate?
#define NUM_EPOCHS 1010 // how long should we train the MLP?
#define LEARNING_RATE 1e-1 // what's the learning rate?
#define LEARNING_RATE_DECAY 0 // do you want a learning rate decay? (i.e. 1e-3)
#define LOSS_TYPE CROSS_ENTROPY // loss

int main(){
  initialize_array(&network_params, 1);
  initialize_array(&temp_values, 1);

  // create input and output data
  double r_xor_inputs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double xor_outputs[4] = {0, 1, 1, 0};
  double** xor_inputs = reshape_input(r_xor_inputs, BATCH_SIZE, NUM_INPUTS, 1, 'd');

  // create network configuration
  size_t num_layers = 3;
  size_t layer_sizes[] = {NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS};
  ActivationType layer_activations[] = {RELU, SIGMOID};
  NormalizationType layer_normalizations[] = {-1, -1};
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

  // clean up
  free_mlp(n);
  free_network_params();

  return 0;
}
