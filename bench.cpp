#include "random.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define NSEC_PER_SEC 1000000000.0

namespace bench {
  inline timespec delta(timespec start, timespec finish){
    timespec result; 
    bool carry = finish.tv_nsec < start.tv_nsec;
    result.tv_sec = finish.tv_sec - start.tv_sec - (int) carry;
    result.tv_nsec = 
      carry ? 
        NSEC_PER_SEC - (start.tv_nsec - finish.tv_nsec) 
        : finish.tv_nsec - start.tv_nsec;
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
      }
      
      if(smaller(max, diffs[ii])) {
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

  void bench(
    const std::function<void()> &callable,
    int repeats
  )
  {
    struct timespec start[repeats], finish[repeats], diff[repeats];
    for(int ii=0; ii< repeats; ii++){
      status_bar(ii, repeats);
      fflush(stdout);

      clock_gettime( CLOCK_MONOTONIC, &start[ii]);
      callable();
      clock_gettime(CLOCK_MONOTONIC, &finish[ii]);

      diff[ii] = delta(start[ii], finish[ii]);
    }
    status_bar(repeats, repeats); printf("\n");
    display_statistics(diff, repeats);
  }

}
