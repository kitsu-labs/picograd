#ifndef MATRIX_H
#define MATRIX_H

#include "types.h"
#include "ops.h"

// TLDR: function to create a 2D matrix of Value pointers
Matrix2D* create_matrix2d(size_t rows, size_t cols){
  Matrix2D* mat = malloc(sizeof(Matrix2D));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = malloc(rows * sizeof(Value**)); // allocate memory for rows (array of pointers to Value pointers)

  for (size_t i = 0; i < rows; i++){
    mat->data[i] = malloc(cols * sizeof(Value*)); // allocate memory for cols (array of Value pointers)
  }

  return mat;
}

// TLDR: function to convert a ValueArray to a Matrix2D
Matrix2D* valuearray_to_matrix2d(ValueArray* arr, size_t rows, size_t cols){
  if (arr == NULL){
    fprintf(stderr, "Error: Input ValueArray is NULL\n");
    return NULL;
  }

  if (arr->used != rows * cols){ // the Matrix should be the same size as the inputs so we can reshape it
    fprintf(stderr, "Error: ValueArray size (%zu) doesn't match specified dimensions (%zu x %zu = %zu)\n", arr->used, rows, cols, rows * cols);
    return NULL;
  }

  Matrix2D* mat = create_matrix2d(rows, cols); // create new Matrix2D
  for (size_t i = 0; i < rows; i++){ // iterate over all the rows in the Matrix
    for (size_t j = 0; j < cols; j++){ // then in each column assign the proper Value
      mat->data[i][j] = arr->values[i * cols + j]; // i * cols finds the start of the row in the 1D array, j shifts it to the correct column
    }
  }

  return mat;
}

// TLDR: function to convert a Matrix2D to a ValueArray
ValueArray* matrix2d_to_valuearray(Matrix2D* mat){
  ValueArray* arr = malloc(sizeof(ValueArray));
  initialize_array(arr, mat->rows * mat->cols); // initialize ValueArray with total number of Matrix elements

  for (size_t i = 0; i < mat->rows; i++){
    for (size_t j = 0; j < mat->cols; j++){
      insert_array(arr, mat->data[i][j]); // fill with Values from Matrix2D
    }
  }

  return arr;
}

// TLDR: reshapes any input (1D, 2D, 3D) into a 2D array where:
// batch_size = y dimension (top to bottom, like rows of independent samples)
// timestep = x dimension (left to right, like a context window)
// num_channels = z dimension (depth wise, like an embedding dimension)
double** reshape_input(void* input, size_t batch_size, size_t timestep, size_t num_channels, char input_type){
  size_t total_samples, num_features;

  if ((batch_size == 0 || batch_size == 1) && (num_channels == 0 || num_channels == 1)){
    // assuming 1D input -> float input_1d[] = {1.0, 2.0, 3.0, 4.0, 5.0} 
    // B: 1 (single batch/example)
    // T: 5 (five time steps, each representing a value)
    // C: 1 (single channel, no depth/dimensionality)
    total_samples = 1; // batch size of 1 sample
    num_features = timestep; // timestep number of features (5)
  } else if (num_channels == 0 || num_channels == 1){
    // assuming 2D input -> float input_2d[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    // B: 4 (four batches/samples, each representing a row)
    // T: 2 (two time steps, each representing a value in each column)
    // C: 1 (single channel, no depth/dimensionality)
    total_samples = batch_size; // batch size of 4 samples
    num_features = timestep; // number of features is equal to timestep (2)
  } else{
    // assuming 3D input -> float input_3d[3][4][2] = {{{}}};
    // B: 3 (three batches/samples, each representing a row)
    // T: 4 (four time steps, each representing a value in each column)
    // C: 2 (two channels or slices, each representing depth or an embedding dimension of sorts)
    total_samples = batch_size; // keep batch size of n samples
    num_features = num_channels * timestep; // combine timestep and features (i.e. context window * embedding dimension)
  }

  double** reshaped = malloc(total_samples * sizeof(double*)); // allocate memory for the reshaped array

  for (size_t i = 0; i < total_samples; i++){
    reshaped[i] = malloc(num_features * sizeof(double)); // allocate memory for the features
  }

  // reshape the input data
  switch(input_type){
    case 'f':{
      float* float_input = (float*)input;
      if ((batch_size == 0 || batch_size == 1) && (num_channels == 0 || num_channels == 1)){
        for (size_t t = 0; t < timestep; t++){ // 1D input
          reshaped[0][t] = (double)float_input[t];
        }
      } else if (num_channels == 0 || num_channels == 1){ // 2D input
          for (size_t b = 0; b < batch_size; b++){
            for (size_t t = 0; t < timestep; t++){
              reshaped[b][t] = (double)float_input[b * timestep + t]; // move to start of correct batch, then offset by t
            }
          }
      } else { // 3D input
        for (size_t b = 0; b < batch_size; b++){
          for (size_t t = 0; t < timestep; t++){
            for (size_t c = 0; c < num_channels; c++){
              reshaped[b][t * num_channels + c] = (double)float_input[(b * timestep * num_channels) + (t * num_channels) + c]; // move to start of correct batch, then move to timestep t, then access channel/dimension c
            }
          }
        }
      }
      break;
    }

    case 'd':{
      double* double_input = (double*)input;
      if ((batch_size == 0 || batch_size == 1) && (num_channels == 0 || num_channels == 1)){
        for (size_t t = 0; t < timestep; t++){ // 1D input
          reshaped[0][t] = double_input[t];
        }
      } else if (num_channels == 0 || num_channels == 1){ // 2D input
        for (size_t b = 0; b < batch_size; b++){
          for (size_t t = 0; t < timestep; t++){
            reshaped[b][t] = double_input[b * timestep + t];
          }
        }
      } else{ // 3D input
        for (size_t b = 0; b < batch_size; b++){
          for (size_t t = 0; t < timestep; t++){
            for (size_t c = 0; c < num_channels; c++){
              reshaped[b][t * num_channels + c] = double_input[(b * timestep * num_channels) + (t * num_channels) + c];
            }
          }
        }
      }
      break;
    }

    default:
      fprintf(stderr, "Unsupported input type in reshape_input\n");
      for (size_t i = 0; i < total_samples; i++){
        free(reshaped[i]);
      }
      free(reshaped);
      return NULL;
  }

  return reshaped;
}

#endif // MATRIX_H
