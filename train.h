#ifndef TRAIN
#define TRAIN

#include "types.h"
#include "ops.h"
#include "nn.h"
#include "memory.h"

//--------------TRAINING--------------
// TLDR: houses all the activation functions in one place
Value* apply_activation(Value* input, ActivationType type){
  if (input == NULL){
    fprintf(stderr, "Error: Input is NULL in apply_activation\n");
    return NULL;
  }

  switch(type){
    case RELU:
      return v_relu(input);
    case SIGMOID:
      return v_sigmoid(input);
    case TANH:
      return v_tanh(input);
    case NO_ACTIVATION:
      return input;
    default:
      return input;
  }
}

// TLDR: houses all the normalization functions in one place
ValueArray* apply_normalization(Layer* layer, ValueArray* inputs, bool is_training) {
  if (layer->norm == NULL) return inputs;

  if (inputs->used != layer->norm->num_features){
    fprintf(stderr, "Error: Input size (%zu) doesn't match number of features (%zu)\n", inputs->used, layer->norm->num_features);
    return NULL;
  }

  // for BatchNorm, we need to reshape the input into a 1 x num_features Matrix
  Matrix2D* input_matrix = valuearray_to_matrix2d(inputs, 1, layer->norm->num_features);
  if (input_matrix == NULL){
    fprintf(stderr, "Error: Failed to convert inputs to Matrix2D\n");
    return NULL;
  }

  ValueArray* result;
  switch (layer->normalization){
    case BATCH:
      result = batch_norm_forward(layer->norm, input_matrix, is_training);
      break;
    case LAYER:
      fprintf(stderr, "Layer normalization not implemented yet\n");
      result = inputs;
      break;
    case NO_NORMALIZATION:
      result = inputs;
      break;
    default:
      result = inputs;
      break;
  }

  free_matrix2d(input_matrix);
  return result;
}

// TLDR: calculates mean squared error loss (MSE)
Value* mse_loss(ValueArray* y_pred, ValueArray* targets, size_t output_size){
  Value* loss = create_value(0.0, true); // initialize loss to zero

  for (size_t i = 0; i < output_size; i++){
    Value* diff = v_sub(y_pred->values[i], targets->values[i]); // calculates difference between target and prediction
    Value* sqdiff = v_pow(diff, create_value(2.0, true)); // squares that difference
    loss = v_add(loss, sqdiff); // add that squared difference to the total loss
  }

  return v_div(loss, create_value(output_size, true)); // calculate the mean (total_loss / nout)
}

// TLDR: calculates binary cross entropy loss (CE)
Value* cross_entropy_loss(ValueArray* y_pred, ValueArray* targets, size_t output_size){
  Value* loss = create_value(0.0, true); // initialize loss to zero

  for (size_t i = 0; i < output_size; i++) {
    Value* y = y_pred->values[i]; // gets the predicted value
    Value* t = targets->values[i]; // gets the target value
    Value* log_y = v_log(y); // takes the log of the predicted value
    Value* log_1_minus_y = v_log(v_sub(create_value(1.0, true), y)); // takes the log of (1 - predicted)
    Value* term1 = v_mul(t, log_y); // multiplies the target by the log of the predicted value
    Value* term2 = v_mul(v_sub(create_value(1.0, true), t), log_1_minus_y); // multiplies (1 - target) by the log of (1 - predicted)
    loss = v_sub(loss, v_add(term1, term2)); // subtracts (t * log(predicted) + (1-target) * log(1-predicted)) from the current loss
  }

  return v_div(loss, create_value(output_size, true)); // returns loss / output_size
}

// TLDR: switch/case loss function for simplicity
Value* calculate_loss(ValueArray* y_pred, ValueArray* targets, LossType type, size_t output_size){
  if (y_pred == NULL || targets == NULL){
    fprintf(stderr, "Error: Predictions or targets are NULL in calculate_loss\n");
    return NULL;
  }
  if (y_pred->used != targets->used || y_pred->used != output_size){
    fprintf(stderr, "Error: Mismatch in dimensions for loss calculation\n");
    return NULL;
  }

  switch(type){
    case MSE:
      return mse_loss(y_pred, targets, output_size);
    case CROSS_ENTROPY:
      return cross_entropy_loss(y_pred, targets, output_size);
    default:
      fprintf(stderr, "Unknown loss type\n");
      return NULL;
  }
}

// TLDR: updates weights using SGD
void update_weights(Value* val, double lr){
  // val->data -= lr * val->grad; // moves weight in direction that reduces loss
  val->data -= lr * (val->grad + 1e-5 * val->data); // with regularization
}

// TLDR: trains the MLP
void train_mlp(MLP* mlp, double** xs, double* ys, TrainingParams* params){
  if (mlp == NULL || xs == NULL || ys == NULL || params == NULL){
    fprintf(stderr, "Error: Invalid parameters in train_mlp\n");
    return;
  }
  if (params->num_samples == 0 || params->num_inputs == 0 || params->num_outputs == 0){
    fprintf(stderr, "Error: Invalid training parameters\n");
    return;
  }

  for (size_t epoch = 0; epoch < params->num_epochs; epoch++){ // for every training epoch
    double total_loss = 0.0; // initialize total loss

    for (size_t i = 0; i < network_params.used; i++){ // zerograd: reset all gradients
      network_params.values[i]->grad = 0.0;
    }

    for (size_t i = 0; i < params->num_samples; i++){ // for every sample provded
      ValueArray* x = array_to_value_array(xs[i], params->num_inputs); // turn those scalar inputs into a ValueArray
      ValueArray* y_true = array_to_value_array(&ys[i], params->num_outputs); // turn those scalar targets into a ValueArray
      ValueArray* y_pred = mlp_forward(mlp, x, true); // forward pass the MLP with those inputs which output predictions

      Value* loss_val = calculate_loss(y_pred, y_true, params->loss_type, params->num_outputs); // calculate the loss across this sample
      backward(loss_val); // backprop to calulate and propagate the gradients
      total_loss += loss_val->data; // add the loss for this sample to the total loss

      free_array(x); // free the ValueArray of sample inputs
      free_array(y_pred); // free the ValueArray of Value predictions
      free_array(y_true); // free the ValueArray of Value targets
      free_temp_values(); // free the ValueArray of forward pass Values
    }

    if (epoch % 10 == 0){
      printf("Epoch %zu, Loss: %f\n", epoch, total_loss / params->num_samples); // print out updates
    }

    for (size_t i = 0; i < network_params.used; i++){
      update_weights(network_params.values[i], params->learning_rate); // update weights
    }

    params->learning_rate *= 1 - params->lr_decay; // decay the learning rate
  }
}

// TLDR: messy helper function to print the network state, thank you claude
void print_network_info(const char* title, size_t num_layers, size_t* layer_sizes, ActivationType* activations, NormalizationType* normalizations, NetworkConfig* config, MLP* mlp){

  printf("%s:\n", title);
  printf("Number of layers: %zu\n", num_layers);
  
  printf("Layer sizes: ");
  for (size_t i = 0; i < num_layers; i++){
    printf("%zu ", layer_sizes[i]);
  }
  printf("\n");
  
  printf("Layer activations: ");
  for (size_t i = 0; i < num_layers - 1; i++){
    switch(activations[i]){
      case NO_ACTIVATION: printf("NO_ACTIVATION "); break;
      case RELU: printf("RELU "); break;
      case SIGMOID: printf("SIGMOID "); break;
      case TANH: printf("TANH "); break;
    }
  }
  printf("\n");
  
  printf("Layer normalizations: ");
  for (size_t i = 0; i < num_layers - 1; i++){
    switch(normalizations[i]){
      case NO_NORMALIZATION: printf("NO_NORMALIZATION "); break;
      case BATCH: printf("BATCH "); break;
      case LAYER: printf("LAYER "); break;
    }
  }
  printf("\n\n");

  if (config != NULL){
    printf("Created NetworkConfig:\n");
    printf("Number of layers: %zu\n", config->num_layers);
    printf("Layer sizes: ");
    for (size_t i = 0; i < config->num_layers; i++){
      printf("%zu ", config->layer_sizes[i]);
    }
    printf("\n");
    
    printf("Layer activations: ");
    for (size_t i = 0; i < config->num_layers - 1; i++){
      switch(config->layer_activations[i]){
        case NO_ACTIVATION: printf("NO_ACTIVATION "); break;
        case RELU: printf("RELU "); break;
        case SIGMOID: printf("SIGMOID "); break;
        case TANH: printf("TANH "); break;
      }
    }
    printf("\n");
    
    printf("Layer normalizations: ");
    for (size_t i = 0; i < config->num_layers - 1; i++){
      switch(config->layer_normalizations[i]){
        case NO_NORMALIZATION: printf("NO_NORMALIZATION "); break;
        case BATCH: printf("BATCH "); break;
        case LAYER: printf("LAYER "); break;
      }
    }
    printf("\n\n");
  }

  if (mlp != NULL){
    printf("Created MLP structure:\n");
    printf("Number of computational layers: %zu\n", mlp->num_layers);

    for (size_t i = 0; i < mlp->num_layers; i++){
      printf("Layer %zu: %zu inputs, %zu outputs\n", i, mlp->layers[i]->nin, mlp->layers[i]->nout);
      printf("  Activation: ");
      switch(mlp->layers[i]->activation){
        case NO_ACTIVATION: printf("NO_ACTIVATION\n"); break;
        case RELU: printf("RELU\n"); break;
        case SIGMOID: printf("SIGMOID\n"); break;
        case TANH: printf("TANH\n"); break;
      }

      printf("  Normalization: ");
      switch(mlp->layers[i]->normalization){
        case NO_NORMALIZATION: printf("NO_NORMALIZATION\n"); break;
        case BATCH: printf("BATCH\n"); break;
        case LAYER: printf("LAYER\n"); break;
      }
    }
    printf("\n");
  }
}
#endif // !TRAIN
