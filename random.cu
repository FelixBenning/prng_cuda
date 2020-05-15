#include "random.h"
#include<stdlib.h>

#define LARGEST_U_INT64 0x1p64 //=2^64

const Hardware HARDWARE = get_hardware();

//Middle Square Weyl Sequence PRNG for seeds (note that it is intended to return uint32_t)
const uint64_t WEYL_CONST = 0xb5ad4eceda1ce2a9;

inline int round_up_division(int dividend, int divisor) {
  return 1+(1-dividend)/divisor;
}

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

void gpu_r_exp(int number, double lambda, double** result, uint64_t **d_rng_state) {
  int threads_per_block = HARDWARE.min_threads_block_full_use;
  int blocks = HARDWARE.min_blocks_full_use;
  int par_threads = HARDWARE.min_threads_full_use;
  
  int seed_bytes = 2* par_threads * sizeof(uint64_t);
  if(*d_rng_state == NULL) {
    uint64_t *seeds;
    seeds = (uint64_t*) malloc(seed_bytes);
    generate_seeds(2*par_threads, seeds);
    cudaMalloc((void**) d_rng_state, seed_bytes);
    cudaMemcpy(*d_rng_state, seeds, seed_bytes, cudaMemcpyHostToDevice);
    free(seeds);
  }
  
  int result_bytes = number * sizeof(double);
  *result = (double*) malloc(result_bytes);

  int cycles = number/par_threads;
  int residual_threads = number%par_threads;

  double *d_result; 
  cudaMalloc((void**) &d_result, result_bytes);   

  r_Exp <<<blocks, threads_per_block>>> (cycles, residual_threads, lambda, *d_rng_state, d_result);

  cudaMemcpy(*result, d_result, result_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_result);
}
