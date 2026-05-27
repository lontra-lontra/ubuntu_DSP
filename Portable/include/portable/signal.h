#pragma once

#include <cmath>
#include <functional>
#include <vector>

struct Signal
{
    std::function<float(float)> func;
    std::function<float()> get_domain_lower_bound;
    std::function<float()> get_domain_upper_bound;
    float precision = 0.0f;
};

struct Distribuition
{
    std::function<float(float, float)> func;
    std::function<float()> get_domain_lower_bound;
    std::function<float()> get_domain_upper_bound;
};

float get_safely_from_signal(Signal signal, float t);

Signal infer_signal_from_distribution(Distribuition distribution, float dt);

Distribuition infer_distribution_from_signal(Signal signal, float dt = NAN);

Signal infer_signal_from_vector(
    const std::vector<float> &values,
    float t0,
    float dt);

Distribuition convolution_as_distribution(
    Distribuition left,
    Distribuition right,
    float dt);

std::vector<float> sample_signal_to_vector(
    Signal signal,
    float start_time,
    int sample_count,
    float dt);
