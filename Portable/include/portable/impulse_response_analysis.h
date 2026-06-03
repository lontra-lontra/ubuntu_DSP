#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include <fftw3.h>

inline std::vector<std::complex<float>> estimate_frequency_response_half(
    const std::vector<float> &array_input,
    const std::vector<float> &array_output)
{
    const int sample_count =
        static_cast<int>(std::min(array_input.size(), array_output.size()));
    const int fft_bins = sample_count / 2 + 1;

    std::vector<std::complex<float>> transfer_half;
    if (sample_count <= 0)
    {
        return transfer_half;
    }

    float *input_in =
        static_cast<float *>(fftwf_malloc(sizeof(float) * sample_count));
    float *output_in =
        static_cast<float *>(fftwf_malloc(sizeof(float) * sample_count));
    fftwf_complex *input_out =
        static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * fft_bins));
    fftwf_complex *output_out =
        static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * fft_bins));

    if (!input_in || !output_in || !input_out || !output_out)
    {
        std::cerr << "FFTW allocation failed\n";
        if (input_in) fftwf_free(input_in);
        if (output_in) fftwf_free(output_in);
        if (input_out) fftwf_free(input_out);
        if (output_out) fftwf_free(output_out);
        return transfer_half;
    }

    for (int i = 0; i < sample_count; ++i)
    {
        input_in[i] = array_input[static_cast<size_t>(i)];
        output_in[i] = array_output[static_cast<size_t>(i)];
    }

    fftwf_plan input_plan =
        fftwf_plan_dft_r2c_1d(sample_count, input_in, input_out, FFTW_ESTIMATE);
    fftwf_plan output_plan =
        fftwf_plan_dft_r2c_1d(sample_count, output_in, output_out, FFTW_ESTIMATE);

    if (!input_plan || !output_plan)
    {
        std::cerr << "FFTW plan creation failed\n";
        if (input_plan) fftwf_destroy_plan(input_plan);
        if (output_plan) fftwf_destroy_plan(output_plan);
        fftwf_free(input_in);
        fftwf_free(output_in);
        fftwf_free(input_out);
        fftwf_free(output_out);
        return transfer_half;
    }

    fftwf_execute(input_plan);
    fftwf_execute(output_plan);

    float max_input_magnitude = 0.0f;
    for (int k = 0; k < fft_bins; ++k)
    {
        const float re = input_out[k][0];
        const float im = input_out[k][1];
        max_input_magnitude =
            std::max(max_input_magnitude, std::sqrt(re * re + im * im));
    }

    const float magnitude_threshold =
        std::max(1e-9f, max_input_magnitude * 1e-3f);

    transfer_half.assign(
        static_cast<size_t>(fft_bins),
        std::complex<float>(0.0f, 0.0f));
    for (int k = 0; k < fft_bins; ++k)
    {
        const std::complex<float> x(input_out[k][0], input_out[k][1]);
        const std::complex<float> y(output_out[k][0], output_out[k][1]);

        if (std::abs(x) >= magnitude_threshold)
        {
            transfer_half[static_cast<size_t>(k)] = y / x;
        }
    }

    fftwf_destroy_plan(input_plan);
    fftwf_destroy_plan(output_plan);
    fftwf_free(input_in);
    fftwf_free(output_in);
    fftwf_free(input_out);
    fftwf_free(output_out);

    return transfer_half;
}

inline std::vector<float> frequency_response_half_to_time_domain(
    const std::vector<std::complex<float>> &transfer_half_values,
    int signal_length = -1)
{
    const int fft_bins = static_cast<int>(transfer_half_values.size());
    if (fft_bins <= 0)
    {
        return {};
    }

    const int sample_count =
        signal_length > 0 ? signal_length : 2 * (fft_bins - 1);
    if (sample_count <= 0 || fft_bins != (sample_count / 2 + 1))
    {
        return {};
    }

    fftwf_complex *transfer_half =
        static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * fft_bins));
    float *impulse_out =
        static_cast<float *>(fftwf_malloc(sizeof(float) * sample_count));

    if (!transfer_half || !impulse_out)
    {
        if (transfer_half) fftwf_free(transfer_half);
        if (impulse_out) fftwf_free(impulse_out);
        return {};
    }

    for (int k = 0; k < fft_bins; ++k)
    {
        transfer_half[k][0] = transfer_half_values[static_cast<size_t>(k)].real();
        transfer_half[k][1] = transfer_half_values[static_cast<size_t>(k)].imag();
    }

    fftwf_plan impulse_plan =
        fftwf_plan_dft_c2r_1d(sample_count, transfer_half, impulse_out, FFTW_ESTIMATE);

    if (!impulse_plan)
    {
        fftwf_free(transfer_half);
        fftwf_free(impulse_out);
        return {};
    }

    std::vector<float> impulse_response(static_cast<size_t>(sample_count), 0.0f);
    fftwf_execute(impulse_plan);

    for (int n = 0; n < sample_count; ++n)
    {
        impulse_response[static_cast<size_t>(n)] =
            impulse_out[n] / static_cast<float>(sample_count);
    }

    fftwf_destroy_plan(impulse_plan);
    fftwf_free(transfer_half);
    fftwf_free(impulse_out);

    return impulse_response;
}

inline std::vector<std::complex<float>> advance_frequency_response_half(
    const std::vector<std::complex<float>> &transfer_half_values,
    int signal_length,
    double sample_advance)
{
    const int fft_bins = static_cast<int>(transfer_half_values.size());
    if (fft_bins <= 0 ||
        signal_length <= 0 ||
        fft_bins != (signal_length / 2 + 1) ||
        !std::isfinite(sample_advance))
    {
        return transfer_half_values;
    }

    constexpr double kTwoPi = 6.28318530717958647692;
    std::vector<std::complex<float>> advanced(
        static_cast<size_t>(fft_bins),
        std::complex<float>(0.0f, 0.0f));

    for (int k = 0; k < fft_bins; ++k)
    {
        const double phase =
            kTwoPi *
            sample_advance *
            static_cast<double>(k) /
            static_cast<double>(signal_length);
        const std::complex<float> rotator(
            static_cast<float>(std::cos(phase)),
            static_cast<float>(std::sin(phase)));
        advanced[static_cast<size_t>(k)] =
            transfer_half_values[static_cast<size_t>(k)] * rotator;
    }

    return advanced;
}
