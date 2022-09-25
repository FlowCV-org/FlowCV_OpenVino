//
// Created by Richard Wardlow
//

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef INC_1EUROFILTER__ONE_EURO_FILTER_HPP_
#define INC_1EUROFILTER__ONE_EURO_FILTER_HPP_

class one_euro_filter {
  public:
    one_euro_filter();
    one_euro_filter(float t0, float x0, float dx0, float min_cutoff, float beta, float d_cutoff);
    void Init(float t0, float x0, float dx0, float min_cutoff, float beta, float d_cutoff);
    void SetMinCutoff(float min_cutoff);
    void SetBeta(float beta);
    void SetDCutoff(float d_cutoff);
    float SmoothingFactor(float t_e, float cutoff);
    float ExponentialSmoothing(float a, float x, float prev);
    float Filter(float t, float value);
  private:
    float min_cutoff_;
    float beta_;
    float d_cutoff_;
    float x_prev_;
    float dx_prev_;
    float t_prev_;
};

#endif //INC_1EUROFILTER__ONE_EURO_FILTER_HPP_
