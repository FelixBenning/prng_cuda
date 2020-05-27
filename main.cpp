#include "random.h"
#include <assert.h>
#include<stdio.h>
#include<cmath>

RngState* rngState;

int input_handler(int argc, char *argv[]){
  const int DEFAULT_NUM = std::pow(10,6);

	if(argc == 1) {
    return DEFAULT_NUM;
  } else {
    assert(argc = 2);
    char *end;
    long num = strtol(argv[1], &end, 10);
    if(*end == '^' || *end == 'e' || *end == 'E'){
      long expon = strtol(end+1, &end, 10);
      return (int) std::round(std::pow((double) num, (double) expon));
    }
    return (int) num;
  } 
}


int main(int argc, char *argv[]){
  int number = input_handler(argc, argv);
  double lambda = 2;
  
  Config* conf = new Config();
  int count = conf->deviceCount();
  int totalNumThreads =0;
  for(int ii=0; ii<count; ii++){
    totalNumThreads+=conf->totalNumThreads(ii);    
  }
  rngState = new RngState(2*totalNumThreads);

  double *result =  rng::gpu_r_exp<double>(number, lambda);

  test::statistical_exp_tests(result, number, lambda);

  printf("benchmark double\n");
  bench::bench([=]{free(rng::gpu_r_exp((int) 10e6, lambda));}, 50);
  printf("benchmark float\n");
  bench::bench([=]{free(rng::gpu_r_exp((int) 10e6, (float) lambda));}, 50);

  free(result);
  delete rngState;
}

