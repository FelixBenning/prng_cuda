#include "random.h"
#include<assert.h>
#include<stdio.h>

int input_handler(int argc, char *argv[]){
  const int DEFAULT_NUM = pow(10,6);

	if(argc == 1) {
    return DEFAULT_NUM;
  } else {
    assert(argc = 2);
    char *end;
    long num = strtol(argv[1], &end, 10);
    if(*end == '^' || *end == 'e' || *end == 'E'){
      long expon = strtol(end+1, &end, 10);
      return (int) round(pow((double) num, (double) expon));
    }
    return (int) num;
  } 
}

double mean(double* x, int len){
  long double result = 0;
  for(int ii=0; ii<len; ii++) {
    result+= (long double) x[ii]/((double) len);
  }
  return (double) result;
}

int main(int argc, char *argv[]){
  int number = input_handler(argc, argv);
  double lambda = 2;
  
  double *result;
  uint64_t *d_rng_state = NULL;

  gpu_r_exp(number, lambda, &result, &d_rng_state);


  printf("mean: %f\n", mean(result, number));

  free(result);
  cudaFree(d_rng_state);
}

