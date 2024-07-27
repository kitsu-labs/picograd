#include "../picograd.h"
ValueArray network_params; 
ValueArray temp_values;

// EXAMPLE:
// we want the network to predict if random numbers are even or odd
// input: binary representation of an integer between 0->255
// output: value between 0 and 1 representing even (0) or odd (1)
#define BATCH_SIZE 10 // how many samples are you providing?
#define NUM_INPUTS 8 // how many inputs are in each sample?
#define NUM_HIDDEN 4 // how many Neurons are in each hidden layer?
#define NUM_OUTPUTS 1 // how many outputs should the MLP calculate?
#define NUM_EPOCHS 1000 // how long should we train the MLP?
#define LEARNING_RATE 1e-2 // what's the learning rate?
#define LEARNING_RATE_DECAY 0 // do you want a learning rate decay? (i.e. 1e-3)
#define LOSS_TYPE MSE // loss

int main(){
  initialize_array(&network_params, 1);
  initialize_array(&temp_values, 1);

  // create input and output data
  int* input_data = (int*)malloc(BATCH_SIZE * sizeof(int));
  double* output_data = (double*)malloc(BATCH_SIZE * sizeof(double));
  double input_bits[BATCH_SIZE][NUM_INPUTS];

  for (int i = 0; i < BATCH_SIZE; i++){
    input_data[i] = rand() % 256; // between 0 and 255
    output_data[i] = (double)(input_data[i] % 2); // determine even or odd

    // convert integer -> binary via bitmask magic
    for (int j = 0; j < NUM_INPUTS; j++){
      input_bits[i][j] = (input_data[i] & (1 << j)) ? 1.0 : 0.0;
    }
  }

  double** xs = reshape_input(input_bits, BATCH_SIZE, NUM_INPUTS, 1, 'd');

  // create network configuration
  size_t num_layers = 4;
  size_t layer_sizes[] = {NUM_INPUTS, NUM_HIDDEN, NUM_HIDDEN, NUM_OUTPUTS};
  ActivationType layer_activations[] = {RELU, RELU, SIGMOID};
  NormalizationType layer_normalizations[] = {BATCH, BATCH, -1};
  NetworkConfig* config = create_network_config(num_layers, layer_sizes, layer_activations, layer_normalizations);
  MLP* n = initialize_mlp(config);

  print_network_info("Created Network Configuration", 
                   config->num_layers, 
                   config->layer_sizes, 
                   config->layer_activations, 
                   config->layer_normalizations, 
                   NULL, 
                   n);

  // group all the training params together
  TrainingParams params = {
    .num_samples = BATCH_SIZE,
    .num_inputs = NUM_INPUTS,
    .num_outputs = NUM_OUTPUTS,
    .num_epochs = NUM_EPOCHS,
    .learning_rate = LEARNING_RATE,
    .lr_decay = LEARNING_RATE_DECAY,
    .loss_type = LOSS_TYPE,
  };

  // train the model
  train_mlp(n, xs, output_data, &params);

  // test the model
  printf("\nTesting the model:\n");
  for (int i = 0; i < BATCH_SIZE; i++){
    ValueArray* x = array_to_value_array(xs[i], NUM_INPUTS);
    ValueArray* y_pred = mlp_forward(n, x, false);
   
    printf("Input: %d (", input_data[i]);
    for (int j = NUM_INPUTS - 1; j >= 0; j--) {
        printf("%d", (int)xs[i][j]);
    }
    printf("), Prediction: %.4f, Actual: %f\n", 
           y_pred->values[0]->data, 
           output_data[i]);

    free_array(x);
    free_array(y_pred);
  }

  // clean up
  for (int i = 0; i < BATCH_SIZE; i++){
      free(xs[i]);
  }

  free(input_data);
  free(output_data);
  free(xs);
  free_mlp(n);
  free_network_config(config);
  free_network_params();
  return 0;
}
