#include "random.h"
#include <assert.h>

int RngState::size() const{
  return this->state_size;
}

uint64_t* RngState::borrow(){
  assert(!this->borrowed);
  this->borrowed = true;
  return this->gpu_mem_loc;
}

void RngState::unborrow(uint64_t *gpu_mem_loc){
  this->gpu_mem_loc = gpu_mem_loc;
  this->borrowed = false;
}
//Middle Square Weyl Sequence PRNG for seeds (note that it is intended to return uint32_t)
const uint64_t WEYL_CONST = 0xb5ad4eceda1ce2a9;

inline uint64_t msws(uint64_t* x, uint64_t* w){
  (*x) *= (*x);
  (*x) += ((*w) += WEYL_CONST);
  return (*x) = ((*x)>>32) | ((*x)<<32);
}
uint64_t* RngState::generate_gpu_seeds(int number){
  int seed_bytes = number * sizeof(uint64_t);
  uint64_t* seeds = (uint64_t*) malloc(seed_bytes);
  //printf("seed address mod 64: %lu\n", (long unsigned) seeds % 64);

  uint64_t x =0, w=0;
  for(int ii=0; ii<number; ii++){
    seeds[ii] = msws(&x, &w);
  }

  uint64_t* result;
  cudaMalloc((void**) &result, seed_bytes);
  printf("device seeds address mod 64: %lu\n", (long unsigned) result %64);
  cudaMemcpy(result, seeds, seed_bytes, cudaMemcpyHostToDevice);
  free(seeds);
  return result;
}

RngState::RngState(int size) {
  this->state_size = size;
  this->gpu_mem_loc = this->generate_gpu_seeds(size);
}
RngState::~RngState() {
  assert(!this->borrowed);
  cudaFree(this->gpu_mem_loc);     
}
