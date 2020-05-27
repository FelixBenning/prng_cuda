#include "random.h"
#include<stdlib.h>
#include <stdio.h>
#include <assert.h>

#define LARGEST_U_INT64 0x1p64 //=2^64

inline int round_up_division(int dividend, int divisor) {
  return 1+(1-dividend)/divisor;
}

namespace rng{
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

  __device__ float r_exp(float lambda, double unif_rv){
    return -lambda*logf(unif_rv);
  }

  template<class T>
    __device__ void zip_r_exp(int number, T lambda, uint64_t* seed, T* result){
      for(int ii=0; ii<number; ii++){
        result[ii]= r_exp(lambda, r_unif(seed));
      }
    }

  template<class T>
    __global__ void r_Exp(
        int cycles, 
        int residual_threads,
        T lambda,
        uint64_t* seeds,
        T* result
        )
    {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      T* result_slice;
      if(idx < residual_threads) {
        result_slice = result + idx*(cycles+1);
      } else {
        result_slice = result + idx*cycles + residual_threads;
      }
      uint64_t* seed_slice = seeds + 2*idx;
      uint64_t local_seed[2] = {seed_slice[0], seed_slice[1]};

      zip_r_exp<T>(cycles, lambda, local_seed, result_slice);

      // handle residual threads
      if(idx < residual_threads) {
        result_slice[cycles] = r_exp(lambda, r_unif(local_seed));
      }
      seed_slice[0]=local_seed[0];
      seed_slice[1]=local_seed[1];
    }

  template double* gpu_r_exp<double>(const int, const double);
  template float* gpu_r_exp<float>(const int, const float);

  template<class T>
    T* gpu_r_exp(const int number, const T lambda) {

      Config* conf = new Config();
      int threads_per_block = conf->threadsPerBlock(0); 
      int blocks = conf->blocks(0);
      int par_threads =conf->totalNumThreads(0); 

      assert(rngState->size()>=2*par_threads);


      int cycles = number/par_threads;
      int residual_threads = number%par_threads;

      T *gpu_result; 
      int result_bytes = number * sizeof(T);
      cudaMalloc((void**) &gpu_result, result_bytes);   

      uint64_t* rng_state_ptr = rngState->borrow();
      r_Exp <<<blocks, threads_per_block>>> (cycles, residual_threads, lambda, rng_state_ptr, gpu_result);
      rngState->unborrow(rng_state_ptr);

      T* result = (T*) malloc(result_bytes);
      cudaMemcpy(result, gpu_result, result_bytes, cudaMemcpyDeviceToHost);
      cudaFree(gpu_result);
      return result;
    }
}
