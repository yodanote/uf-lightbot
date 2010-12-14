#include <iostream>
#include "fann.h"

using namespace std;

int main()
{
  fann_type *calc_out;
  fann_type input[10];

  struct fann *ann = fann_create_from_file("emotions.net");

  input[0] = 0.87;
  input[1] = 1.20;
  input[2] = 1.03;
  input[3] = 1.45;
  input[4] = 0.96;
  input[5] = 1.00;
  input[6] = 0.98;
  input[7] = 1.486;
  input[8] = 1.042;
  input[9] = 1.016;
  
  calc_out = fann_run(ann, input);
  cout<<calc_out[0]<<" "<<calc_out[1]<<" "<<calc_out[2]<<" "<<calc_out[3]<<endl;
  fann_destroy(ann);

  return 0;
}


