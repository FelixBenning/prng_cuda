#include<stdlib.h>
#include<stdint.h>
#include<assert.h>
#include<stdio.h>

#define LARGEST_U_INT64 0x1p64 //=2^64


//Middle Square Weyl Sequence PRNG for seeds (note that it is intended to return uint32_t)
const uint64_t WEYL_CONST = 0xb5ad4eceda1ce2a9;

inline static uint64_t msws(uint64_t* x, uint64_t* w){
 (*x) *= (*x);
 (*x) += ((*w) += WEYL_CONST);
 return (*x) = ((*x)>>32) | ((*x)<<32);
}
void generate_seeds(int number, uint64_t *result){
  uint64_t x =0, w=0;
  for(int ii=0; ii<number; ii++){
    result[ii] = msws(&x, &w);
  }
}

__device__ uint64_t xorshift128plus(uint64_t* sp) {
  uint64_t x = sp[0],
  y = sp[1];
  sp[0] = y;
  x^= x << 23;
  x^= y ^ (x>>17) ^ (y>>26);
  sp[1]=x;
  return x+y;
}

__device__ double r_unif(uint64_t* seed) {
  return ( (double) xorshift128plus(seed) )/LARGEST_U_INT64;
}

__device__ double r_exp(double lambda, double unif_rv){
  return -lambda*log(unif_rv);
}

__device__ void zip_r_exp(int number, double lambda, uint64_t* seed, double* result){
  for(int ii=0; ii<number; ii++){
   result[ii]= r_exp(lambda, r_unif(seed));
  }
}

__global__ void r_Exp(
  int cycles, 
  int residual_threads,
  double lambda,
  uint64_t* seeds,
  double* result
)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 double* result_slice;
 if(idx < residual_threads) {
   result_slice = result + idx*(cycles+1);
 } else {
   result_slice = result + idx*cycles + residual_threads;
 }
 uint64_t* seed_slice = seeds + 2*idx;
 uint64_t local_seed[2] = {seed_slice[0], seed_slice[1]};

 zip_r_exp(cycles, lambda, local_seed, result_slice);
 
 seed_slice[0]=local_seed[0];
 seed_slice[1]=local_seed[1];
}

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

typedef struct {
  int min_threads_full_use;
  int min_threads_block_full_use;
  int min_blocks_full_use;
}Hardware;

Hardware get_hardware(){
  cudaDeviceProp prop;
  cudaGetDeviceProperties( &prop, 0);
  printf("name: %s\n", prop.name);

  int threads_in_warp = prop.warpSize;
  int sm_count = prop.multiProcessorCount;
  int warp_per_sm;
  
  //https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications 
  //table: architecture specifications of the compute capability major.minor version
  switch(prop.major){
    case 1:
      warp_per_sm=1;
      break;
    case 2:
      warp_per_sm=2;
      break;
    case 3:
      warp_per_sm=4;
      break;
    case 5:
      warp_per_sm=4;
      break;
    case 6:
      if(prop.minor == 0) {
        warp_per_sm = 2;
      } else {
        warp_per_sm = 4;
      }
      break;
    case 7:
      warp_per_sm = 4;
      break;
    default:
      printf("Warning: default warp_per_sm = 4 used");
      warp_per_sm = 4;
      break;
  }

  int min_threads_block_full_use = threads_in_warp * warp_per_sm;
  int min_threads_full_use = min_threads_block_full_use * sm_count;

  Hardware hardware = {
    min_threads_full_use,
    min_threads_block_full_use,
    /*min_blocks_full_use=*/sm_count
  };

  return hardware;
}

inline int round_up_division(int dividend, int divisor) {
  return 1+(1-dividend)/divisor;
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

  const Hardware HARDWARE = get_hardware();

  int threads_per_block = HARDWARE.min_threads_block_full_use;
  int blocks = HARDWARE.min_blocks_full_use;
  int par_threads = HARDWARE.min_threads_full_use;

  int cycles = number/par_threads;
  int residual_threads = number%par_threads;

  double *result, *d_result; 
  uint64_t *seeds, *d_seeds;
  int result_bytes = number * sizeof(double);
  int seed_bytes = 2* par_threads * sizeof(uint64_t);

  result = (double*) malloc(result_bytes);
  cudaMalloc((void**) &d_result, result_bytes);   
  
  seeds = (uint64_t*) malloc(seed_bytes);
  generate_seeds(2*par_threads, seeds);

  cudaMalloc((void**) &d_seeds, seed_bytes);
  cudaMemcpy(d_seeds, seeds, seed_bytes, cudaMemcpyHostToDevice);
  
  
  r_Exp <<<blocks, threads_per_block>>> (cycles, residual_threads, lambda, d_seeds, d_result);

  cudaMemcpy(result, d_result, result_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_result);

  printf("mean: %f\n", mean(result, number));

  free(result);
  cudaFree(d_seeds);
  free(seeds);
}

