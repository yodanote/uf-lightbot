#include <iostream>
#include "fann.h"

using namespace std;

int 
main(int argc, char *argv[]) 
{
  const unsigned int num_inputs = 2;
  const unsigned int num_outputs = 1;
  const unsigned int num_layers = 3;
  const unsigned int num_neurons_hidden = 4;
  const float desired_error = 0.0001;
  const unsigned int max_iterations = 500000;
  const unsigned int iterations_between_reports = 1000;
  
  struct fann *ann = fann_create_standard(num_layers,
                                 num_inputs,
                                 num_neurons_hidden,
                                 num_outputs);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  fann_train_on_file(ann, "xor.data", max_iterations, iterations_between_reports, desired_error);
  fann_save(ann, "xor_float.net");
  fann_destroy(ann);

  return 0;
}
