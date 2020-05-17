#ifndef PRNG_CUDA
#define PRNG_CUDA
#include <stdint.h>


typedef struct {
  int min_threads_full_use;
  int min_threads_block_full_use;
  int min_blocks_full_use;
}Hardware;

namespace hardware{
  const Hardware get();
}

namespace rng {
  void gpu_r_exp(int number, double lambda, double** result, uint64_t **d_rng_state);
}


namespace test {
  void statistical_exp_tests(double* vec, int len, double lambda);
}

namespace bench {
  void gpu_r_exp(int number, int repeats, double lambda, uint64_t** gpu_rng_state);
}

#endif
