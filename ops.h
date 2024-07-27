#ifndef OPS
#define OPS

#include "types.h"
#include "matrix.h"

//--------------ARRAY INITIALIZATION/INSERTION--------------
// TLDR: allocates memory for new array of Values
void initialize_array(ValueArray* arr, size_t initial_size){
  arr->values = (Value**)malloc(initial_size*sizeof(Value*)); // allocates memory for arr slots, size of Value struct
  arr->used = 0; // zero slots are initially allocated
  arr->size = initial_size; // number of slots = initial_size
}

// TLDR: inserts Value into ValueArray, handles realloc if full
void insert_array(ValueArray* arr, Value* element){
  if (arr->used == arr->size){ // if array is at max allocation
    arr->size *= 2; // double the array size, O(n), but reallocs happen less often as size grows, so avg cost is O(1) for large n
    arr->values = (Value**)realloc(arr->values, arr->size*sizeof(Value*)); // realloc memory
  }

  arr->values[arr->used++] = element; // insert to end of array, O(1)
}

// TLDR: converts scalar array to ValueArray of Values
ValueArray* array_to_value_array(double* arr, size_t size){
  ValueArray* a = malloc(sizeof(ValueArray));
  initialize_array(a, size); // initializes array to be size of input scalars

  for (size_t i = 0; i < size; i++){
    insert_array(a, create_value(arr[i], true)); // turn scalars into Values and insert into ValueArray
  }

  return a;
}

//--------------GRADIENTS--------------
// TLDR: prevents gradients from exploding resulting in NaN
// node -> the "output" (going forward) / the "parent" (going backward)
static inline void clip_grad(Value* node, float min_val, float max_val){
  node->grad = fmaxf(fminf(node->grad, max_val), min_val);
}

// TLDR: handles gradient in case of addition during backprop
// f(x) = x + y
// f'(x) = 1 (since both inputs contribute equally to the output)
void add_backward(Value* node){
  node->prev->values[0]->grad += node->grad;
  node->prev->values[1]->grad += node->grad;
  clip_grad(node->prev->values[0], -50.0, 50.0);
  clip_grad(node->prev->values[1], -50.0, 50.0);
}

// TLDR: handles gradient in case of subtraction during backprop
// f(x) = x - y
// f'(x) = 1
// f'(y) = -1
void sub_backward(Value* node){
  node->prev->values[0]->grad += node->grad;
  node->prev->values[1]->grad -= node->grad;
  clip_grad(node->prev->values[0], -50.0, 50.0);
  clip_grad(node->prev->values[1], -50.0, 50.0);
}

// TLDR: handles gradient in case of multiplication during backprop
// f(x) = x^n
// f'(x) = nx^(n-1)
void mul_backward(Value* node){
  node->prev->values[0]->grad += node->prev->values[1]->data * node->grad;
  node->prev->values[1]->grad += node->prev->values[0]->data * node->grad;
  clip_grad(node->prev->values[0], -50.0, 50.0);
  clip_grad(node->prev->values[1], -50.0, 50.0);
}

// TLDR: handles gradient in case of division during backprop
// f(x, y) = x / y
// f'(x) = 1 / y
// f'(y) = -x / y^2
void div_backward(Value* node){
  node->prev->values[0]->grad += (1.0 / node->prev->values[1]->data) * node->grad;
  node->prev->values[1]->grad += (-node->prev->values[0]->data / (node->prev->values[1]->data * node->prev->values[1]->data)) * node->grad;
  clip_grad(node->prev->values[0], -50.0, 50.0);
  clip_grad(node->prev->values[1], -50.0, 50.0);
}

// TLDR: handles gradient in case of power during backprop
// f(x, y) = x^y
// f'(x) = y * x^(y-1)
// f'(y) = x^y * ln(x) (for x > 0)
void pow_backward(Value* node){
  node->prev->values[0]->grad += node->prev->values[1]->data * pow(node->prev->values[0]->data, node->prev->values[1]->data - 1) * node->grad;
  if (node->prev->values[0]->data > 0){ // log base must be > 0
    node->prev->values[1]->grad += log(node->prev->values[0]->data) * pow(node->prev->values[0]->data, node->prev->values[1]->data) * node->grad;
  }
  clip_grad(node->prev->values[0], -50.0, 50.0);
  clip_grad(node->prev->values[1], -50.0, 50.0);
}

// TLDR: handles gradient in case of exponentiation during backprop
// f(x) = e^x
// f'(x) = e^x
void exp_backward(Value* node){
  node->prev->values[0]->grad += node->data * node->grad;
  clip_grad(node->prev->values[0], -50.0, 50.0);
}

// TLDR: handles gradient in case of natural logarithm during backprop
// f(x) = ln(x)
// f'(x) = 1/x
void log_backward(Value* node) {
  node->prev->values[0]->grad += node->grad * (1.0 / node->prev->values[0]->data);
  clip_grad(node->prev->values[0], -50.0, 50.0);
}

// TLDR: handles gradient in case of tanh activation during backprop
// f(x) = tanh(x)
// f'(x) = 1 - tanh^2(x)
void tanh_backward(Value* node){
  node->prev->values[0]->grad += (1-node->data*node->data) * node->grad;
  clip_grad(node->prev->values[0], -50.0, 50.0);
}

// TLDR: handles gradient in case of relu activation during backprop
// f(x) = max(0, x)
// f'(x) = (x > 0) ? 1 : 0
void relu_backward(Value* node) {
  node->prev->values[0]->grad += (node->data > 0) ? node->grad : 0;
  clip_grad(node->prev->values[0], -50.0, 50.0);
}

// TLDR: handles gradient in case of sigmoid activation during backprop
// f(x) = 1 / (1 + exp(-x))
// f'(x) = f(x) * (1 - f(x))
void sigmoid_backward(Value* node) {
  Value* a = node->prev->values[0];
  double sigmoid_value = node->data;
  double gradient = sigmoid_value * (1 - sigmoid_value);
  a->grad += gradient * node->grad;
  clip_grad(node->prev->values[0], -50.0, 50.0);
}

//--------------OPERATIONS--------------
// TLDR: adds two Values going forward
Value* v_add(Value* a, Value* b){
  Value* output = create_value(a->data + b->data, true);
  insert_array(output->prev, a); // output node obtains a as a child node
  insert_array(output->prev, b); // output node obtains b as a child node
  output->backward = add_backward; // attaches associated backprop function
  return output;
}

// TLDR: subs two Values going forward
Value* v_sub(Value* a, Value* b){
  Value* output = create_value(a->data - b->data, true);
  insert_array(output->prev, a); // output node obtains a as a child node
  insert_array(output->prev, b); // output node obtains b as a child node
  output->backward = sub_backward; // attaches associated backprop function
  return output;
}

// TLDR: muls two Values going forward
Value* v_mul(Value* a, Value* b){
  Value* output = create_value(a->data * b->data, true);
  insert_array(output->prev, a); // output node obtains a as a child node
  insert_array(output->prev, b); // output node obtains b as a child node
  output->backward = mul_backward; // attaches associated backprop function
  return output;
}

// TLDR: divs two Values going forward
Value* v_div(Value* a, Value* b){
  if(b->data == 0.0){
    printf("Div by zero. Adding eps.");
    b->data += 0.0001;
  }
  Value* output = create_value(a->data / b->data, true);
  insert_array(output->prev, a); // output node obtains a as a child node
  insert_array(output->prev, b); // output node obtains b as a child node
  output->backward = div_backward; // attaches associated backprop function
  return output;
}

// TLDR: pow two Values going forward
Value* v_pow(Value* a, Value* b){
  Value* output = create_value(pow(a->data, b->data), true);
  insert_array(output->prev, a); // output node obtains a as a child node
  insert_array(output->prev, b); // output node obtains b as a child nodes
  output->backward = pow_backward; // attaches associated backprop function
  return output;
}

// TLDR: exp Value going forward
Value* v_exp(Value* a){
  Value* output = create_value(exp(a->data), true);
  insert_array(output->prev, a); // output node obtains a as a child node
  output->backward = exp_backward; // attaches associated backprop function
  return output;
}

// TLDR: log Value going forward
Value* v_log(Value* a) {
  Value* output = create_value(log(a->data), true);
  insert_array(output->prev, a); // output node obtains a as a child node
  output->backward = log_backward; // attaches associated backprop function
  return output;
}

// TLDR: tanh Value going forward
Value* v_tanh(Value* a){
  Value* output = create_value((exp(2*a->data)-1) / (exp(2*a->data)+1), true);
  insert_array(output->prev, a); // output node obtains a as a child node
  output->backward = tanh_backward; // attaches associated backprop function
  return output;
}

// TLDR: relu Value going forward
Value* v_relu(Value* a) {
  Value* output = create_value(fmax(0, a->data), true);
  insert_array(output->prev, a); // output node obtains a as a child node
  output->backward = relu_backward; // attaches associated backprop function
  return output;
}

// TLDR: sigmoid Value going forward
Value* v_sigmoid(Value* a) {
  double sigmoid_value = 1 / (1 + exp(-a->data));
  Value* output = create_value(sigmoid_value, true);
  insert_array(output->prev, a); // output node obtains a as a child node
  output->backward = sigmoid_backward; // attaches associated backprop function
  return output;
}

//--------------BACKPROP--------------
// TLDR: topological sort for the graph of Values
void build_topo(Value* node, ValueArray* topo, ValueArray* visited){
  for (size_t i = 0; i < visited->used; i++){ // base case: loop through visited nodes, if curr node is already visited, return
    if (visited->values[i] == node) return;
  }

  insert_array(visited, node); // set curr node to visited

  for (size_t i = 0; i < node->prev->used; i++){ // recursively loop through all children (previous nodes in forward pass)
    build_topo(node->prev->values[i], topo, visited);
  }
  
  insert_array(topo, node); // after visiting all children, place curr node in topo sort
}

// TLDR: computes backward pass
void backward(Value* node){
  ValueArray topo;
  initialize_array(&topo, 1); // array to hold the topological sort of the graph
  ValueArray visited;
  initialize_array(&visited, 1); // array to keep track of visited nodes during topological sort
  build_topo(node, &topo, &visited); // build the topological sort starting from the output node 

  node->grad = 1.0; // set gradient of output node to 1.0 (dL/dL = 1)
  for (int i = topo.used - 1; i >= 0; i--){ // iterate through sort from back to front
    if(topo.values[i]->backward) topo.values[i]->backward(topo.values[i]); // if the node has a backward function, compute and distribute gradients
  }
  
  // frees temp memory used for topological sort once backward pass is done
  free(topo.values);
  free(visited.values);
}

#endif // !OPS
