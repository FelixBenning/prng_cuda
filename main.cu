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


int main(int argc, char *argv[]){
  int number = input_handler(argc, argv);
  double lambda = 2;
  
  uint64_t *d_rng_state = NULL;

  double *result =  rng::gpu_r_exp(number, lambda, &d_rng_state);

  test::statistical_exp_tests(result, number, lambda);

  bench::gpu_r_exp((int) 10e6, 50, lambda, &d_rng_state);

  free(result);
  cudaFree(d_rng_state);
}

