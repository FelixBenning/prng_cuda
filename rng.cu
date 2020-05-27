#include "random.h"
#include<stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <limits>

#define LARGEST_U_INT64 0x1p64 //=2^64
#define GPU_ID 0

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

  __device__ double pseudInv_exp(double lambda, double unif_rv){
    return -lambda*log(unif_rv);
  }

  __device__ float pseudInv_exp(float lambda, double unif_rv){
    return -lambda*logf(unif_rv);
  }

  __device__ float custom_rv(float exp_rv){
    return exp_rv * cosf(exp_rv);
  }

  template<class T>
    __device__ T composed_custom(T lambda, uint64_t* seed){
      return custom_rv(pseudInv_exp(lambda, r_unif(seed)));
    }

  template<class T>
    __device__ T composed_exp(T lambda, uint64_t* seed){
      return pseudInv_exp(lambda, r_unif(seed));
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
      int iter = cycles;
      if(idx < residual_threads) {
        result_slice = result + idx*(++iter);
      } else {
        result_slice = result + idx*iter + residual_threads;
      }
      uint64_t* seed_slice = seeds + 2*idx;
      uint64_t local_seed[2] = {seed_slice[0], seed_slice[1]};

      while(iter--){
        result_slice[iter]= composed_exp(lambda, local_seed);
      }

      seed_slice[0]=local_seed[0];
      seed_slice[1]=local_seed[1];
    }

  __device__ float generate_max(const int maxOfN, float lambda,  uint64_t* seed){
    float runningMax = logf(0);
    for(int ii=0; ii<maxOfN; ii++){
      runningMax = fmaxf(composed_custom(lambda, seed), runningMax);
    }
    return runningMax;
  }

  template<class T>
    __global__ void max_rv(
      int number,
      int threadTotal,
      int maxOfN,
      T lambda,
      uint64_t* seeds,
      T* result
      )
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t* seed_slice = seeds+2*idx;
    uint64_t local_seed[2] = {seed_slice[0], seed_slice[1]};
    T* result_slice = result + idx*number;
    int cycle = number/threadTotal;
    while(cycle--){
      result_slice[cycle] = generate_max(maxOfN, lambda, local_seed);
    }

    int warpSize = 64;

    result_slice = result + cycle*threadTotal;
    int residual = number%threadTotal;
    for(int ii=2; ii<= warpSize; ii=ii<<1){
      if(residual>=threadTotal/ii){
        residual -= threadTotal/ii;

        int maxOfn = maxOfN/ii + ((idx%ii < maxOfN%ii) ? 1 : 0); 
        float local_result = generate_max(maxOfn, lambda, local_seed);
        //Since ii<=warpsize the order is determined as the threads within a warp can only advance together
        if(idx%ii){
          result_slice[idx/ii] = local_result;
        } else {
          for(int jj = 1; jj<ii; jj++){
            if(jj == idx%ii){
              result_slice[idx/ii] = fmaxf(result_slice[idx/ii], local_result);
            }
          }
        }
        result_slice += threadTotal/ii;
      }
    }

    //now residual<threadTotal/warpSize
    if(idx/warpSize /*warpIndex*/ < residual){
        int maxOfn = maxOfN/warpSize + ((idx%warpSize < maxOfN%warpSize) ? 1 : 0); 
        float local_result = generate_max(maxOfn, lambda, local_seed);
        //the order is determined as the threads within a warp can only advance together
        if(idx%warpSize){
          result_slice[idx/warpSize] = local_result;
        } else {
          for(int jj = 1; jj<warpSize; jj++){
            if(jj == idx%warpSize){
              result_slice[idx/warpSize] = fmaxf(result_slice[idx/warpSize], local_result);
            }
          }
        }
    }

    seed_slice[0] = local_seed[0];
    seed_slice[1] = local_seed[1];
  }

  template double* gpu_r_exp<double>(const int, const double);
  template float* gpu_r_exp<float>(const int, const float);

  template<class T>
    T* gpu_r_exp(const int number, const T lambda) {

      Config* conf = new Config();
      int threads_per_block = conf->threadsPerBlock(GPU_ID); 
      int blocks = conf->blocks(GPU_ID);
      int par_threads =conf->totalNumThreads(GPU_ID); 

      assert(rngState->size()>=2*par_threads);


      int cycles = number/par_threads;
      int residual = number%par_threads;

      T *gpu_result; 
      int result_bytes = number * sizeof(T);
      cudaMalloc((void**) &gpu_result, result_bytes);   

      uint64_t* rng_state_ptr = rngState->borrow();
      r_Exp <<<blocks, threads_per_block>>> (cycles, residual, lambda, rng_state_ptr, gpu_result);
      rngState->unborrow(rng_state_ptr);

      T* result = (T*) malloc(result_bytes);
      cudaMemcpy(result, gpu_result, result_bytes, cudaMemcpyDeviceToHost);
      cudaFree(gpu_result);
      return result;
    }
  
  //template double* gpu_max_rv<double>(const int, const int, const double);
  template float* gpu_max_rv<float>(const int, const int, const float);

  template<class T>
    T* gpu_max_rv(const int number, const int maxOfN, const T lambda){
      Config* conf = new Config();
      int threads_per_block = conf->threadsPerBlock(GPU_ID);
      int blocks = conf->blocks(GPU_ID);
      int threads = conf->totalNumThreads(GPU_ID);
      
      assert(rngState->size()>=2*threads);

      
      T *gpu_result;
      int result_bytes = number * sizeof(T);
      cudaMalloc((void**) &gpu_result, result_bytes);

      uint64_t* rng_state_ptr = rngState->borrow();
      max_rv <<<blocks, threads_per_block>>> (number, threads, maxOfN, lambda, rng_state_ptr, gpu_result);
      rngState->unborrow(rng_state_ptr);

      T* result = (T*) malloc(result_bytes);
      cudaMemcpy(result, gpu_result, result_bytes, cudaMemcpyDeviceToHost);
      cudaFree(gpu_result);
      return result;
    }
}
