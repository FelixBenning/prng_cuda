#include "random.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define NSEC_PER_SEC 1000000000.0;

namespace bench {
  inline timespec delta(timespec start, timespec finish){
    timespec result; 
    result.tv_sec = finish.tv_sec - start.tv_sec;
    result.tv_nsec = finish.tv_nsec - start.tv_nsec;
    return result;
  }
  bool smaller(timespec left, timespec right) {
    if(left.tv_sec<right.tv_sec){
      return true;
    } else if(left.tv_sec > right.tv_sec) {
      return false;
    } else {
      if (left.tv_nsec < right.tv_nsec) {
        return true;
      } else {
        return false;
      }
    }
  }

  void display_statistics(timespec diffs[], int length) {
    long sum_nsec = diffs[0].tv_nsec;
    long sum_sec = diffs[0].tv_sec;

    timespec min = diffs[0];
    timespec max = diffs[0];
    for(int ii=1; ii<length; ii++){
      sum_nsec += diffs[ii].tv_nsec;
      sum_sec += diffs[ii].tv_sec;
      if(smaller(diffs[ii], min)){
        min = diffs[ii];
      } if(smaller(max, diffs[ii])) {
        max = diffs[ii];
      }
    }

    double mean = ((double) sum_sec)/((double) length) 
      + ((double) sum_nsec)/((double) length) / NSEC_PER_SEC;
    
    double dmin = (double) min.tv_sec + ((double) min.tv_nsec)/NSEC_PER_SEC;
    double dmax = (double) max.tv_sec + ((double) max.tv_nsec)/NSEC_PER_SEC;

    printf(
      " min (sec)| mean     | max      (of %d)\n"
      " %.6f | %.6f | %.6f\n" ,
      length, dmin, mean, dmax
    );
  }

  void status_bar(int iteration, int total){
    float progress = ((float) iteration)/ (float) total;
    int per50 = (int) (progress*50);
      printf("\r[");
      for(int i=0; i<per50; i++) putchar('#');
      for(int i=per50; i<50; i++) putchar(' ');
      printf("]");
  }

  void gpu_r_exp(
    int number,
    int repeats,
    double lambda,
    uint64_t** gpu_rng_state
  )
  {
    printf(
      "Benchmarking 'gpu_r_exp' generating %g RV with lambda = %g)'\n",
      (float) number,
      lambda
    );

    struct timespec start[repeats], finish[repeats], diff[repeats];
    for(int ii=0; ii< repeats; ii++){
      clock_gettime( CLOCK_MONOTONIC, &start[ii]);
      rng::gpu_r_exp(number, lambda, gpu_rng_state);
      clock_gettime(CLOCK_MONOTONIC, &finish[ii]);

      status_bar(ii, repeats);
      diff[ii] = delta(start[ii], finish[ii]);
      fflush(stdout);
    }
    status_bar(repeats, repeats); printf("\n");
    display_statistics(diff, repeats);
  }

  void bench(
    std::function<double*()> &funct_to_bench,
    int repeats
  )
  {
    struct timespec start[repeats], finish[repeats], diff[repeats];
    for(int ii=0; ii< repeats; ii++){
      clock_gettime( CLOCK_MONOTONIC, &start[ii]);
      funct_to_bench();
      clock_gettime(CLOCK_MONOTONIC, &finish[ii]);

      status_bar(ii, repeats);
      diff[ii] = delta(start[ii], finish[ii]);
      fflush(stdout);
    }
    status_bar(repeats, repeats); printf("\n");
    display_statistics(diff, repeats);
  }

}
