#ifndef PRNG_CUDA
#define PRNG_CUDA
#include <stdint.h>
#include <functional>
#include <cuda_runtime_api.h>


class Config {
  public: 
    int deviceCount();
    int cudaCores(int device);
    int threadsPerWarp(int device);
    int smCount(int device);
    int warpPerSm(int device);
    void printName(int device);

    int blocks(int device);
    int threadsPerBlock(int device);
    int totalNumThreads(int device);

    Config();
    
  private:
    int device_count;
    cudaDeviceProp* prop;
    int *warp_per_sm;
};

class RngState {
  public:
    uint64_t* borrow();
    void unborrow(uint64_t *gpu_mem_loc);
    int size() const;
    RngState(int size); 
    ~RngState();

  private:
    bool borrowed = false;
    int state_size;
    uint64_t* gpu_mem_loc;
    uint64_t* generate_gpu_seeds(int number);
};

extern RngState* rngState;

namespace rng {
  template<class T>
  T* gpu_r_exp(int number, const T lambda);
}

namespace test {
  void statistical_exp_tests(double* vec, int len, double lambda);
}

namespace bench {
  void bench(const std::function<void()>& funct_to_bench, int repeats);
  void gpu_r_exp(int number, int repeats, double lambda);
}

#endif
