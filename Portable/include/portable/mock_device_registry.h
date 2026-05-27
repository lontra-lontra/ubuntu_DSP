#pragma once

#include "portable/mockportaudio.h"

#include <vector>

using ExactConvolutionKernel = std::vector<float>;
using ExactConvolutionKernelBank = std::vector<ExactConvolutionKernel>;
using ExactConvolutionKernelMatrix = std::vector<ExactConvolutionKernelBank>;

void clear_mock_devices();

PaDeviceIndex register_mock_signal_generator_device(
    const char *name,
    int input_channels,
    int output_channels,
    double default_sample_rate,
    const std::vector<double> &supported_sample_rates,
    const std::vector<MockInputGenerator> &input_generators);

PaDeviceIndex register_mock_device_inferred_from_topology(
    const char *name,
    const char *frequency_matrix_csv_path,
    int input_channels,
    int output_channels,
    double default_sample_rate,
    const std::vector<double> &supported_sample_rates);

std::vector<float> stupid_and_garanteed_convolution(
    const std::vector<float> &input_samples,
    const std::vector<float> &kernel_samples);

PaDeviceIndex register_device_that_expects_exact_convolution(
    const char *name,
    double default_sample_rate,
    const std::vector<double> &supported_sample_rates,
    const std::vector<MockInputGenerator> &input_generators,
    const ExactConvolutionKernelMatrix &kernel_matrix);
