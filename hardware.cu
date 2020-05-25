#include "random.h"
#include <stdio.h>

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


const Hardware hardware::get(){
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
