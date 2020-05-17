#include "random.h"
#include <assert.h>
#include<stdio.h>

template <typename T>
inline T square(T x){
  return x*x;
}

double mean(double* x, int len){
  long double result = 0;
  for(int ii=0; ii<len; ii++) {
    result+= (long double) x[ii]/((long double) len);
  }
  return (double) result;
}

double var(double* x, int len, double mean){
  long double result = 0;
  for(int ii=0; ii<len; ii++) {
    result += square((long double) x[ii] -mean)/(long double) len;
  }
  return result;
}

namespace test {
  void statistical_exp_tests(double* vec, int len, double lambda){
    double eps = 0.01;

    double m = mean(vec, len);
    printf("mean: %f\n", m);
    assert(m-eps < lambda && lambda < m+eps);

    double v = var(vec, len, m);
    printf("var: %f\n", v);
    assert(v-eps < square(lambda) && square(lambda) < v+eps);

  }
}
