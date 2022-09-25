//
// Created by Richard Wardlow
//

#include "one_euro_filter.hpp"

one_euro_filter::one_euro_filter()
{
    min_cutoff_ = 1.0f;
    beta_ = 0.0f;
    d_cutoff_ = 1.0f;
    x_prev_ = 0.0f;
    dx_prev_ = 0.0f;
    t_prev_ = 0.0f;
}

one_euro_filter::one_euro_filter(float t0, float x0, float dx0, float min_cutoff, float beta, float d_cutoff)
{
    min_cutoff_ = min_cutoff;
    beta_ = beta;
    d_cutoff_ = d_cutoff;
    x_prev_ = x0;
    dx_prev_ = dx0;
    t_prev_ = t0;
}

void one_euro_filter::Init(float t0, float x0, float dx0, float min_cutoff, float beta, float d_cutoff)
{
    min_cutoff_ = min_cutoff;
    beta_ = beta;
    d_cutoff_ = d_cutoff;
    x_prev_ = x0;
    dx_prev_ = dx0;
    t_prev_ = t0;
}

float one_euro_filter::SmoothingFactor(float t_e, float cutoff)
{
    float r = 2.0f * M_PI * cutoff * t_e;

    return r / (r +1);
}

float one_euro_filter::ExponentialSmoothing(float a, float x, float prev)
{
    return a * x + (1.0f - a) * prev;
}

float one_euro_filter::Filter(float t, float value)
{
    float t_e = t - t_prev_;
    float a_d = SmoothingFactor(t_e, d_cutoff_);
    float dx = (value - x_prev_) / t_e;
    float dx_hat = ExponentialSmoothing(a_d, dx, dx_prev_);

    float cut_off = min_cutoff_ + beta_ * abs(dx_hat);
    float a = SmoothingFactor(t_e, cut_off);
    float x_hat = ExponentialSmoothing(a, value, x_prev_);

    x_prev_ = x_hat;
    dx_prev_ = dx_hat;
    t_prev_ = t;

    return x_hat;
}

void one_euro_filter::SetMinCutoff(float min_cutoff)
{
    min_cutoff_ = min_cutoff;
}

void one_euro_filter::SetBeta(float beta)
{
    beta_ = beta;
}

void one_euro_filter::SetDCutoff(float d_cutoff)
{
    d_cutoff_ = d_cutoff;
}
