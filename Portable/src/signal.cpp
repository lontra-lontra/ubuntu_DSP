#include "portable/signal.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <fftw3.h>

float get_safely_from_signal(Signal signal, float t)
{
    const float lower = signal.get_domain_lower_bound();
    const float upper = signal.get_domain_upper_bound();

    const bool below_lower = std::isfinite(lower) && t < lower;
    const bool above_upper = std::isfinite(upper) && t > upper;

    if (below_lower || above_upper)
    {
        std::cerr << "Error: signal(" << t
                  << ") is not defined"
                  << " the domain is [" << lower << "," << upper << "]"
                  << std::endl;
    }

    return signal.func(t);
}

Signal infer_signal_from_distribution(Distribuition distribution, float dt)
{
    Signal signal;
    signal.precision = dt;

    signal.func = [distribution, dt](float t) -> float
    {
        if (dt <= 0.0f)
        {
            return 0.0f;
        }

        const float t_inf = t - dt * 0.5f;
        const float t_sup = t + dt * 0.5f;
        return distribution.func(t_inf, t_sup) / dt;
    };

    signal.get_domain_lower_bound = [distribution]() -> float
    {
        return distribution.get_domain_lower_bound();
    };

    signal.get_domain_upper_bound = [distribution]() -> float
    {
        return distribution.get_domain_upper_bound();
    };

    return signal;
}

Distribuition infer_distribution_from_signal(Signal signal, float dt)
{
    if (std::isnan(dt))
    {
        dt = signal.precision;
    }

    Distribuition distribution;
    distribution.func = [signal, dt](float t_inf, float t_sup) -> float
    {
        if (dt <= 0.0f)
        {
            return 0.0f;
        }

        if (t_sup < t_inf)
        {
            std::swap(t_inf, t_sup);
        }

        const float signal_lower = signal.get_domain_lower_bound();
        const float signal_upper = signal.get_domain_upper_bound();

        float start = t_inf;
        float end = t_sup;

        if (std::isfinite(signal_lower))
        {
            start = std::max(start, signal_lower);
        }
        if (std::isfinite(signal_upper))
        {
            end = std::min(end, signal_upper);
        }

        if (end <= start)
        {
            return 0.0f;
        }

        float integral = 0.0f;
        for (float x = start; x < end; x += dt)
        {
            const float x_next = std::min(x + dt, end);
            const float x_mid = 0.5f * (x + x_next);
            integral += signal.func(x_mid) * (x_next - x);
        }

        return integral;
    };

    distribution.get_domain_lower_bound = [signal]() -> float
    {
        return signal.get_domain_lower_bound();
    };

    distribution.get_domain_upper_bound = [signal]() -> float
    {
        return signal.get_domain_upper_bound();
    };

    return distribution;
}

Signal infer_signal_from_vector(
    const std::vector<float> &values,
    float t0,
    float dt)
{
    const std::shared_ptr<std::vector<float>> owned_values =
        std::make_shared<std::vector<float>>(values);

    Signal signal;
    signal.func = [owned_values, t0, dt](float t) -> float
    {
        if (dt <= 0.0f || owned_values->empty())
        {
            return 0.0f;
        }

        const int index =
            static_cast<int>(std::floor((t - t0) / dt + 0.5f));
        if (index < 0 || index >= static_cast<int>(owned_values->size()))
        {
            return 0.0f;
        }

        return (*owned_values)[static_cast<size_t>(index)];
    };

    signal.get_domain_lower_bound = [t0]() -> float
    {
        return t0;
    };

    signal.get_domain_upper_bound = [owned_values, t0, dt]() -> float
    {
        if (owned_values->empty())
        {
            return t0;
        }

        return t0 + dt * static_cast<float>(owned_values->size() - 1);
    };

    signal.precision = dt;
    return signal;
}

Distribuition convolution_as_distribution(
    Distribuition left,
    Distribuition right,
    float dt)
{
    Distribuition empty_distribution;
    empty_distribution.func = [](float, float) -> float
    {
        return 0.0f;
    };
    empty_distribution.get_domain_lower_bound = []() -> float
    {
        return 0.0f;
    };
    empty_distribution.get_domain_upper_bound = []() -> float
    {
        return 0.0f;
    };

    if (dt <= 0.0f)
    {
        return empty_distribution;
    }

    const Signal left_signal = infer_signal_from_distribution(left, dt);
    const Signal right_signal = infer_signal_from_distribution(right, dt);

    const float left_lower = left_signal.get_domain_lower_bound();
    const float left_upper = left_signal.get_domain_upper_bound();
    const float right_lower = right_signal.get_domain_lower_bound();
    const float right_upper = right_signal.get_domain_upper_bound();

    if (!std::isfinite(left_lower) || !std::isfinite(left_upper) ||
        !std::isfinite(right_lower) || !std::isfinite(right_upper) ||
        left_upper < left_lower || right_upper < right_lower)
    {
        return empty_distribution;
    }

    const int left_count =
        std::max(1, static_cast<int>(std::floor((left_upper - left_lower) / dt + 0.5f)) + 1);
    const int right_count =
        std::max(1, static_cast<int>(std::floor((right_upper - right_lower) / dt + 0.5f)) + 1);

    const std::vector<float> left_samples =
        sample_signal_to_vector(left_signal, left_lower, left_count, dt);
    const std::vector<float> right_samples =
        sample_signal_to_vector(right_signal, right_lower, right_count, dt);

    if (left_samples.empty() || right_samples.empty())
    {
        return empty_distribution;
    }

    const int convolution_count =
        static_cast<int>(left_samples.size() + right_samples.size() - 1);

    int fft_size = 1;
    while (fft_size < convolution_count)
    {
        fft_size <<= 1;
    }
    const int fft_bins = fft_size / 2 + 1;

    float *left_time =
        static_cast<float *>(fftwf_malloc(sizeof(float) * static_cast<size_t>(fft_size)));
    float *right_time =
        static_cast<float *>(fftwf_malloc(sizeof(float) * static_cast<size_t>(fft_size)));
    float *output_time =
        static_cast<float *>(fftwf_malloc(sizeof(float) * static_cast<size_t>(fft_size)));
    fftwf_complex *left_freq =
        static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * static_cast<size_t>(fft_bins)));
    fftwf_complex *right_freq =
        static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * static_cast<size_t>(fft_bins)));

    if (!left_time || !right_time || !output_time || !left_freq || !right_freq)
    {
        if (left_time) fftwf_free(left_time);
        if (right_time) fftwf_free(right_time);
        if (output_time) fftwf_free(output_time);
        if (left_freq) fftwf_free(left_freq);
        if (right_freq) fftwf_free(right_freq);
        return empty_distribution;
    }

    std::fill(left_time, left_time + fft_size, 0.0f);
    std::fill(right_time, right_time + fft_size, 0.0f);
    std::fill(output_time, output_time + fft_size, 0.0f);
    for (int index = 0; index < static_cast<int>(left_samples.size()); ++index)
    {
        left_time[index] = left_samples[static_cast<size_t>(index)];
    }
    for (int index = 0; index < static_cast<int>(right_samples.size()); ++index)
    {
        right_time[index] = right_samples[static_cast<size_t>(index)];
    }

    fftwf_plan left_plan =
        fftwf_plan_dft_r2c_1d(fft_size, left_time, left_freq, FFTW_ESTIMATE);
    fftwf_plan right_plan =
        fftwf_plan_dft_r2c_1d(fft_size, right_time, right_freq, FFTW_ESTIMATE);
    fftwf_plan inverse_plan =
        fftwf_plan_dft_c2r_1d(fft_size, left_freq, output_time, FFTW_ESTIMATE);

    if (!left_plan || !right_plan || !inverse_plan)
    {
        if (left_plan) fftwf_destroy_plan(left_plan);
        if (right_plan) fftwf_destroy_plan(right_plan);
        if (inverse_plan) fftwf_destroy_plan(inverse_plan);
        fftwf_free(left_time);
        fftwf_free(right_time);
        fftwf_free(output_time);
        fftwf_free(left_freq);
        fftwf_free(right_freq);
        return empty_distribution;
    }

    fftwf_execute(left_plan);
    fftwf_execute(right_plan);

    for (int bin = 0; bin < fft_bins; ++bin)
    {
        const float left_re = left_freq[bin][0];
        const float left_im = left_freq[bin][1];
        const float right_re = right_freq[bin][0];
        const float right_im = right_freq[bin][1];

        left_freq[bin][0] = left_re * right_re - left_im * right_im;
        left_freq[bin][1] = left_re * right_im + left_im * right_re;
    }

    fftwf_execute(inverse_plan);

    std::vector<float> convolution_values(
        static_cast<size_t>(convolution_count),
        0.0f);
    const float scale = 1.0f / static_cast<float>(fft_size);
    for (int index = 0; index < convolution_count; ++index)
    {
        convolution_values[static_cast<size_t>(index)] =
            output_time[index] * scale;
    }

    fftwf_destroy_plan(left_plan);
    fftwf_destroy_plan(right_plan);
    fftwf_destroy_plan(inverse_plan);
    fftwf_free(left_time);
    fftwf_free(right_time);
    fftwf_free(output_time);
    fftwf_free(left_freq);
    fftwf_free(right_freq);

    const Signal convolution_signal =
        infer_signal_from_vector(convolution_values, left_lower + right_lower, dt);
    return infer_distribution_from_signal(convolution_signal, dt);
}

std::vector<float> sample_signal_to_vector(
    Signal signal,
    float start_time,
    int sample_count,
    float dt)
{
    if (sample_count <= 0 || dt <= 0.0f)
    {
        return {};
    }

    std::vector<float> values(static_cast<size_t>(sample_count), 0.0f);
    for (int index = 0; index < sample_count; ++index)
    {
        const float t = start_time + static_cast<float>(index) * dt;
        values[static_cast<size_t>(index)] = signal.func(t);
    }

    return values;
}
