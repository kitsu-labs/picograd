#include "../picograd.h"
ValueArray network_params;
ValueArray temp_values;

// EXAMPLE:
// simple sequence prediction problem
// input: sequences of 5 numbers
// output: predict the next number in the sequence
#define BATCH_SIZE 20 // how many samples are you providing?
#define SEQUENCE_LENGTH 5 // what's the context window of each sample?
#define NUM_INPUTS 1 // how many features for each sequence step?
#define NUM_HIDDEN 8 // how many Neurons are in each hidden layer?
#define NUM_OUTPUTS 1 // how many outputs should the MLP calculate?
#define NUM_EPOCHS 2000 // how long should we train the MLP?
#define LEARNING_RATE 1e-4 // what's the learning rate?
#define LEARNING_RATE_DECAY 1e-4 // do you want a learning rate decay? (i.e. 1e-3)
#define LOSS_TYPE MSE // loss

// generate a simple sequence: each number is the sum of the two preceding numbers
void generate_sequence(double* sequence, int length){
  sequence[0] = rand() % 10;
  sequence[1] = rand() % 10;
  for (int i = 2; i < length; i++){
    sequence[i] = sequence[i-1] + sequence[i-2];
  }
}

int main(){
  initialize_array(&network_params, 1);
  initialize_array(&temp_values, 1);

  // create input and output data
  double input_data[BATCH_SIZE][SEQUENCE_LENGTH][NUM_INPUTS]; // (50, 5, 1)
  double target_data[BATCH_SIZE][NUM_OUTPUTS]; // (50, 1)

  for (int i = 0; i < BATCH_SIZE; i++){ // for 50 batches
    double sequence[SEQUENCE_LENGTH + 1];
    generate_sequence(sequence, SEQUENCE_LENGTH + 1); // first sequence length are the actual sequence, +1 is the target

    for (int j = 0; j < SEQUENCE_LENGTH; j++){
      input_data[i][j][0] = sequence[j]; // inserting input elements to batch i at slot j
    }
    target_data[i][0] = sequence[SEQUENCE_LENGTH]; // each batch only has 1 output, the target number
  }

  // reshape 3d input data -> 2d input data
  double** reshaped_input = reshape_input(input_data, BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS, 'd');

  // create network configuration
  size_t num_layers = 4;
  size_t layer_sizes[] = {SEQUENCE_LENGTH * NUM_INPUTS, NUM_HIDDEN, NUM_HIDDEN, NUM_OUTPUTS};
  ActivationType layer_activations[] = {TANH, TANH, -1};
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

  // set up training parameters
  TrainingParams params = {
    .num_samples = BATCH_SIZE,
    .num_inputs = SEQUENCE_LENGTH * NUM_INPUTS,
    .num_outputs = NUM_OUTPUTS,
    .num_epochs = NUM_EPOCHS,
    .learning_rate = LEARNING_RATE,
    .lr_decay = LEARNING_RATE_DECAY,
    .loss_type = LOSS_TYPE
  };

  // train the model
  train_mlp(n, reshaped_input, target_data[0], &params);

  // test the model
  printf("Testing the model:\n");
  for (int i = 0; i < 5; i++){
    double test_sequence[SEQUENCE_LENGTH + 1];
    generate_sequence(test_sequence, SEQUENCE_LENGTH + 1);
    
    double test_input[SEQUENCE_LENGTH][NUM_INPUTS]; // 2d because we're just testing single examples, don't need batches
    for (int j = 0; j < SEQUENCE_LENGTH; j++){
      test_input[j][0] = test_sequence[j];
    }
    
    double** reshaped_test = reshape_input(test_input, 1, SEQUENCE_LENGTH, NUM_INPUTS, 'd');
    ValueArray* input_array = array_to_value_array(reshaped_test[0], SEQUENCE_LENGTH * NUM_INPUTS);
    ValueArray* output = mlp_forward(n, input_array, false); // false because we're at inference
    
    printf("Input sequence: ");
    for (int j = 0; j < SEQUENCE_LENGTH; j++){
      printf("%.1f ", test_sequence[j]);
    }

    printf("\n");
    printf("Predicted next number: %.1f\n", output->values[0]->data);
    printf("Actual next number: %.1f\n\n", test_sequence[SEQUENCE_LENGTH]);
    
    free_array(input_array);
    free_array(output);
    free_reshape(reshaped_test, 1);
  }

  // clean up
  free_mlp(n);
  free_network_config(config);
  free_reshape(reshaped_input, BATCH_SIZE);
  free_network_params();

  return 0;
}
