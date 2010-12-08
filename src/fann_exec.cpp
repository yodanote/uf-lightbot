#include <iostream>
#include "fann.h"

using namespace std;

int main()
{
  fann_type *calc_out;
  fann_type input[2];

  struct fann *ann = fann_create_from_file("xor_float.net");

  input[0] = -1;
  input[1] = 1;
  calc_out = fann_run(ann, input);
  cout<<"xor test ("<<input[0]<<", "<<input[1]<<") = "<<calc_out[0]<<endl;
  fann_destroy(ann);

  return 0;
}


