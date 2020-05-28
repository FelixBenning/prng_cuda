#include "random.h"
#include <stdio.h>
#include <assert.h>

const int THREADS_PER_CORE = 3;

int Config::blocks(int device){
  return this->smCount(device) * THREADS_PER_CORE;
}
int Config::threadsPerBlock(int device){
  int threads = this->totalNumThreads(device);
  int blocks = this->blocks(device);
  assert(!(threads%blocks));
  return threads/blocks;
}
int Config::totalNumThreads(int device){
  return this->cudaCores(device) * THREADS_PER_CORE;
}

int calculate_warps_per_sm(cudaDeviceProp prop){
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
  return warp_per_sm;
}

Config::Config(){
  int count = 0;
  cudaGetDeviceCount(&count);
  this->device_count=count;
  cudaDeviceProp props[count];
  int wps[count];
  for(int ii =0; ii<count; ii++){
    cudaGetDeviceProperties(&props[ii], ii);
    wps[ii] = calculate_warps_per_sm(props[ii]); 
  }
  this->prop = props;
  this->warp_per_sm = wps;
}

int Config::deviceCount() {
  return this->device_count;
}

int Config::threadsPerWarp(int device){
  return this->prop[device].warpSize;
}

int Config::smCount(int device){
  return this->prop[device].multiProcessorCount;
}

int Config::warpPerSm(int device){
  return this->warp_per_sm[device];
}

int Config::cudaCores(int device){
  return 
  this->warpPerSm(device)
    * this->threadsPerWarp(device)
    * this->smCount(device);
}

void  Config::printName(int device){
  printf("GPU #%d: %s\n", device, this->prop[device].name); 
}

