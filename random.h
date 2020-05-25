#ifndef PRNG_CUDA
#define PRNG_CUDA
#include <stdint.h>
#include <functional>
#include <cuda_runtime_api.h>


typedef struct {
  int min_threads_full_use;
  int min_threads_block_full_use;
  int min_blocks_full_use;
}Hardware;

class Config {
  public: 
    int deviceCount();
    int cudaCores(int device);
    int threadsPerWarp(int device);
    int smCount(int device);
    int warpPerSm(int device);
    void printName(int device);

    int blocks();
    int threadsPerBlock();

    Config();
    
  private:
    int device_count;
    cudaDeviceProp* prop;
    int *warp_per_sm;
};

namespace hardware{
  const Hardware get();
}

namespace rng {
  double* gpu_r_exp(int number, const double lambda, uint64_t **d_rng_state);
}

class RngState {
  public:
    uint64_t* borrow();
    void unborrow(uint64_t *gpu_mem_loc);
    int size();
    RngState(int size); 
    ~RngState();

  private:
    bool borrowed = false;
    int state_size;
    uint64_t* gpu_mem_loc;
    uint64_t* generate_gpu_seeds(int number);
};


namespace test {
  void statistical_exp_tests(double* vec, int len, double lambda);
}

namespace bench {
  void bench(std::function<double*()>& funct_to_bench, int repeats);
  void gpu_r_exp(int number, int repeats, double lambda, uint64_t** gpu_rng_state);
}

#endif
