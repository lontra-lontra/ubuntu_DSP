#include "portable/mock_device_registry.h"

#include "portable/csv_utils.h"
#include "portable/impulse_response_analysis.h"
#include "portable/signal.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#ifndef PORTABLE_EXACT_CONV_VIEW_SCRIPT
#define PORTABLE_EXACT_CONV_VIEW_SCRIPT "Portable/scripts/show_exact_convolution_validation.py"
#endif

namespace
{

constexpr double kExactConvolutionTolerancePercent = 0.01;
constexpr double kExactConvolutionToleranceRatio =
    kExactConvolutionTolerancePercent / 100.0;
constexpr size_t kValidationProgressLineWidth = 160;

struct ExactConvolutionState
{
    std::vector<std::vector<float>> inputHistory;
    std::vector<std::vector<float>> outputHistory;
    ExactConvolutionKernelMatrix kernelMatrix;
};

std::string sanitize_for_filename(
    const std::string &text)
{
    std::string sanitized;
    sanitized.reserve(text.size());

    for (unsigned char ch : text)
    {
        if (std::isalnum(ch))
        {
            sanitized.push_back(static_cast<char>(ch));
        }
        else
        {
            sanitized.push_back('_');
        }
    }

    if (sanitized.empty())
    {
        return "mock_device";
    }

    return sanitized;
}

std::string build_python_command(
    const std::string &script_path,
    const std::string &csv_path)
{
    std::string command = "py -3 ";
    command += '"' + script_path + "\" \"" + csv_path + '"';
    return command;
}

double max_absolute_value(
    const std::vector<float> &values)
{
    double peak = 0.0;
    for (float value : values)
    {
        peak = std::max(peak, std::fabs(static_cast<double>(value)));
    }
    return peak;
}

void print_validation_progress(
    const char *stage,
    size_t current_output_channel,
    size_t total_output_channels,
    size_t current_input_channel,
    size_t total_input_channels)
{
    std::cout << '\r'
              << "[mock exact convolution] " << stage
              << " output " << (current_output_channel + 1) << "/" << total_output_channels
              << ", input " << (current_input_channel + 1) << "/" << total_input_channels
              << std::flush;
}

void print_validation_sample_progress(
    size_t current_output_channel,
    size_t total_output_channels,
    size_t current_input_channel,
    size_t total_input_channels,
    size_t current_sample,
    size_t total_samples)
{
    std::cout << '\r'
              << "[mock exact convolution] naive validating"
              << " output " << (current_output_channel + 1) << "/" << total_output_channels
              << ", input " << (current_input_channel + 1) << "/" << total_input_channels
              << ", sample " << (current_sample + 1) << "/" << total_samples
              << std::flush;
}

void clear_validation_progress_line()
{
    std::cout << '\r'
              << std::string(kValidationProgressLineWidth, ' ')
              << '\r'
              << std::flush;
}

bool save_exact_convolution_validation_csv(
    const ExactConvolutionState &state,
    const std::vector<std::vector<float>> &expected_outputs,
    double sample_rate,
    const std::string &csv_path)
{
    const size_t channel_count =
        std::min(state.outputHistory.size(), expected_outputs.size());
    if (channel_count == 0)
    {
        return false;
    }

    static const std::vector<float> empty_channel;

    size_t sample_count = 0;
    for (size_t channel = 0; channel < channel_count; ++channel)
    {
        if (channel < state.inputHistory.size())
        {
            sample_count = std::max(sample_count, state.inputHistory[channel].size());
        }
        sample_count = std::max(sample_count, state.outputHistory[channel].size());
        sample_count = std::max(sample_count, expected_outputs[channel].size());
    }

    std::vector<float> time_axis(sample_count, 0.0f);
    for (size_t sample = 0; sample < sample_count; ++sample)
    {
        time_axis[sample] =
            static_cast<float>(static_cast<double>(sample) / std::max(sample_rate, 1.0));
    }

    std::vector<std::string> headers = {"time (s)"};
    std::vector<const std::vector<float> *> columns = {&time_axis};

    for (size_t channel = 0; channel < channel_count; ++channel)
    {
        headers.push_back("input " + std::to_string(channel));
        columns.push_back(
            channel < state.inputHistory.size()
                ? &state.inputHistory[channel]
                : &empty_channel);
        headers.push_back("output " + std::to_string(channel));
        columns.push_back(&state.outputHistory[channel]);
        headers.push_back("expected output " + std::to_string(channel));
        columns.push_back(&expected_outputs[channel]);
    }

    return save_arrays_to_csv(csv_path, headers, columns);
}

void trim_trailing_all_zero_output_frames_if_needed(
    ExactConvolutionState &state)
{
    if (state.outputHistory.empty())
    {
        return;
    }

    size_t frame_count = state.outputHistory[0].size();
    for (const std::vector<float> &channel_history : state.outputHistory)
    {
        frame_count = std::min(frame_count, channel_history.size());
    }

    while (frame_count > 0)
    {
        bool all_zero = true;
        for (const std::vector<float> &channel_history : state.outputHistory)
        {
            if (std::fabs(channel_history[frame_count - 1]) > 0.0f)
            {
                all_zero = false;
                break;
            }
        }
        if (!all_zero)
        {
            break;
        }
        --frame_count;
    }

    for (std::vector<float> &channel_history : state.outputHistory)
    {
        if (channel_history.size() > frame_count)
        {
            channel_history.resize(frame_count);
        }
    }

    for (std::vector<float> &channel_history : state.inputHistory)
    {
        if (channel_history.size() > frame_count)
        {
            channel_history.resize(frame_count);
        }
    }
}

} // namespace

void clear_mock_devices()
{
    std::vector<MockDevice> *devices = mock_pa_get_devices_list();
    if (!devices)
    {
        return;
    }
    devices->clear();
}

PaDeviceIndex register_mock_signal_generator_device(
    const char *name,
    int input_channels,
    int output_channels,
    double default_sample_rate,
    const std::vector<double> &supported_sample_rates,
    const std::vector<MockInputGenerator> &input_generators)
{
    if (!name || !name[0])
    {
        throw std::invalid_argument(
            "register_mock_signal_generator_device: device name must be non-empty");
    }

    std::vector<MockDevice> *devices = mock_pa_get_devices_list();
    if (!devices)
    {
        throw std::runtime_error(
            "register_mock_signal_generator_device: device storage is null");
    }

    MockDevice device{};
    device.ownedName = name;
    device.info.name = device.ownedName.c_str();
    device.info.maxInputChannels = input_channels;
    device.info.maxOutputChannels = output_channels;
    device.info.defaultSampleRate = default_sample_rate;
    device.supportedSampleRates = supported_sample_rates;
    device.inputGenerators = input_generators;
    device.outputHistory.assign(
        static_cast<size_t>(output_channels),
        std::vector<float>());

    devices->push_back(device);
    MockDevice &stored = devices->back();
    stored.info.name = stored.ownedName.c_str();
    return static_cast<PaDeviceIndex>(devices->size() - 1);
}

std::vector<float> stupid_and_garanteed_convolution(
    const std::vector<float> &input_samples,
    const std::vector<float> &kernel_samples)
{
    if (input_samples.empty() || kernel_samples.empty())
    {
        return {};
    }

    const size_t output_sample_count =
        input_samples.size() + kernel_samples.size() - 1;
    std::vector<float> output_samples(output_sample_count, 0.0f);

    for (size_t output_index = 0; output_index < output_sample_count; ++output_index)
    {
        double accumulated = 0.0;
        const size_t kernel_begin =
            (output_index + 1 >= kernel_samples.size())
                ? output_index + 1 - kernel_samples.size()
                : 0;
        const size_t kernel_end =
            std::min(output_index + 1, input_samples.size());

        for (size_t input_index = kernel_begin; input_index < kernel_end; ++input_index)
        {
            const size_t kernel_index = output_index - input_index;
            accumulated +=
                static_cast<double>(input_samples[input_index]) *
                static_cast<double>(kernel_samples[kernel_index]);
        }

        output_samples[output_index] = static_cast<float>(accumulated);
    }

    return output_samples;
}

void accumulate_naive_convolution_prefix_with_progress(
    const std::vector<float> &input_samples,
    const std::vector<float> &kernel_samples,
    std::vector<float> &output_samples,
    size_t output_channel_index,
    size_t total_output_channels,
    size_t input_channel_index,
    size_t total_input_channels)
{
    if (output_samples.empty() || input_samples.empty() || kernel_samples.empty())
    {
        return;
    }

    const size_t output_sample_count = output_samples.size();
    const size_t progress_stride =
        std::max<size_t>(1024, input_samples.size() / 16);

    for (size_t input_index = 0; input_index < input_samples.size(); ++input_index)
    {
        if (input_index == 0 ||
            input_index + 1 == input_samples.size() ||
            input_index % progress_stride == 0)
        {
            print_validation_sample_progress(
                output_channel_index,
                total_output_channels,
                input_channel_index,
                total_input_channels,
                input_index,
                input_samples.size());
        }

        const double input_value =
            static_cast<double>(input_samples[input_index]);
        if (input_value == 0.0)
        {
            continue;
        }

        const size_t tap_count =
            std::min(kernel_samples.size(), output_sample_count - input_index);
        for (size_t tap = 0; tap < tap_count; ++tap)
        {
            output_samples[input_index + tap] +=
                static_cast<float>(
                    input_value *
                    static_cast<double>(kernel_samples[tap]));
        }
    }
}

std::vector<float> less_stupid_and_still_garanteed_mixed_convolution(
    const std::vector<std::vector<float>> &input_histories,
    const ExactConvolutionKernelBank &kernels_for_output,
    size_t output_sample_count,
    size_t output_channel_index,
    size_t total_output_channels)
{
    const size_t input_channel_count =
        std::min(input_histories.size(), kernels_for_output.size());

    std::vector<float> output_samples(output_sample_count, 0.0f);
    for (size_t input_channel = 0; input_channel < input_channel_count; ++input_channel)
    {
        print_validation_progress(
            "convolving",
            output_channel_index,
            total_output_channels,
            input_channel,
            input_channel_count);

        const std::vector<float> &input_samples = input_histories[input_channel];
        const std::vector<float> &kernel_samples = kernels_for_output[input_channel];
        if (input_samples.empty() || kernel_samples.empty())
        {
            continue;
        }

        accumulate_naive_convolution_prefix_with_progress(
            input_samples,
            kernel_samples,
            output_samples,
            output_channel_index,
            total_output_channels,
            input_channel,
            input_channel_count);
    }

    return output_samples;
}

PaDeviceIndex register_device_that_expects_exact_convolution(
    const char *name,
    double default_sample_rate,
    const std::vector<double> &supported_sample_rates,
    const std::vector<MockInputGenerator> &input_generators,
    const ExactConvolutionKernelMatrix &kernel_matrix)
{
    if (!name || !name[0])
    {
        throw std::invalid_argument(
            "register_device_that_expects_exact_convolution: device name must be non-empty");
    }

    if (default_sample_rate <= 0.0)
    {
        throw std::invalid_argument(
            "register_device_that_expects_exact_convolution: sample rate must be positive");
    }

    if (input_generators.empty())
    {
        throw std::invalid_argument(
            "register_device_that_expects_exact_convolution: at least one input function is required");
    }

    if (kernel_matrix.empty())
    {
        throw std::invalid_argument(
            "register_device_that_expects_exact_convolution: at least one output kernel row is required");
    }

    const size_t input_channel_count = input_generators.size();
    for (const ExactConvolutionKernelBank &kernel_row : kernel_matrix)
    {
        if (kernel_row.size() != input_channel_count)
        {
            throw std::invalid_argument(
                "register_device_that_expects_exact_convolution: every output kernel row must match the input channel count");
        }

        for (const ExactConvolutionKernel &kernel : kernel_row)
        {
            if (kernel.empty())
            {
                throw std::invalid_argument(
                    "register_device_that_expects_exact_convolution: kernels must be non-empty");
            }
        }
    }

    const int output_channel_count = static_cast<int>(kernel_matrix.size());
    const std::shared_ptr<ExactConvolutionState> state =
        std::make_shared<ExactConvolutionState>();
    state->inputHistory.assign(
        input_channel_count,
        std::vector<float>());
    state->outputHistory.assign(
        static_cast<size_t>(output_channel_count),
        std::vector<float>());
    state->kernelMatrix = kernel_matrix;

    std::vector<MockInputGenerator> wrapped_input_generators;
    wrapped_input_generators.reserve(input_channel_count);
    for (size_t channel = 0; channel < input_generators.size(); ++channel)
    {
        const MockInputGenerator original_generator = input_generators[channel];
        wrapped_input_generators.push_back(
            [state, original_generator, channel](double time_seconds) -> float
        {
            if (channel == 0 && std::fabs(time_seconds) <= 1e-12)
            {
                state->inputHistory.assign(
                    state->inputHistory.size(),
                    std::vector<float>());
                state->outputHistory.assign(
                    state->kernelMatrix.size(),
                    std::vector<float>());
            }

            const float sample =
                original_generator ? original_generator(time_seconds) : 0.0f;
            if (channel < state->inputHistory.size())
            {
                state->inputHistory[channel].push_back(sample);
            }
            return sample;
        });
    }

    const PaDeviceIndex device_index =
        register_mock_signal_generator_device(
            name,
            static_cast<int>(input_channel_count),
            output_channel_count,
            default_sample_rate,
            supported_sample_rates,
            wrapped_input_generators);

    std::vector<MockDevice> *devices = mock_pa_get_devices_list();
    if (!devices ||
        device_index < 0 ||
        device_index >= static_cast<PaDeviceIndex>(devices->size()))
    {
        throw std::runtime_error(
            "register_device_that_expects_exact_convolution: failed to retrieve stored mock device");
    }

    MockDevice &device = (*devices)[static_cast<size_t>(device_index)];
    device.internalState = state.get();
    device.onStreamClosed =
        [state](MockDevice &validated_device, double sample_rate)
    {
        ExactConvolutionState *exact_state =
            static_cast<ExactConvolutionState *>(validated_device.internalState);
        if (!exact_state)
        {
            std::cerr << "[mock exact convolution] Missing internal state for device "
                      << (validated_device.info.name ? validated_device.info.name : "(null)")
                      << '\n';
            return;
        }

        exact_state->outputHistory = validated_device.outputHistory;
        trim_trailing_all_zero_output_frames_if_needed(*exact_state);

        const size_t channel_count =
            std::min(
                exact_state->outputHistory.size(),
                exact_state->kernelMatrix.size());
        if (channel_count == 0)
        {
            std::cerr << "[mock exact convolution] No channels available to validate for device "
                      << (validated_device.info.name ? validated_device.info.name : "(null)")
                      << '\n';
            return;
        }

        size_t total_output_sample_count = 0;
        for (size_t channel = 0; channel < channel_count; ++channel)
        {
            total_output_sample_count += exact_state->outputHistory[channel].size();
        }
        if (total_output_sample_count == 0)
        {
            std::cerr << "[mock exact convolution] No output samples were captured for device "
                      << (validated_device.info.name ? validated_device.info.name : "(null)")
                      << ". Validation skipped.\n";
            return;
        }

        std::vector<std::vector<float>> expected_outputs(
            channel_count,
            std::vector<float>());
        std::vector<double> channel_max_abs_diff(channel_count, 0.0);
        std::vector<double> channel_max_rel_error_percent(channel_count, 0.0);

        bool passed = true;
        for (size_t channel = 0; channel < channel_count; ++channel)
        {
            const std::vector<float> &actual_output_samples =
                exact_state->outputHistory[channel];
            expected_outputs[channel] =
                less_stupid_and_still_garanteed_mixed_convolution(
                    exact_state->inputHistory,
                    exact_state->kernelMatrix[channel],
                    actual_output_samples.size(),
                    channel,
                    channel_count);

            double max_abs_diff = 0.0;
            for (size_t sample = 0; sample < actual_output_samples.size(); ++sample)
            {
                max_abs_diff = std::max(
                    max_abs_diff,
                    std::fabs(
                        static_cast<double>(actual_output_samples[sample]) -
                        static_cast<double>(expected_outputs[channel][sample])));
            }

            const double expected_peak =
                max_absolute_value(expected_outputs[channel]);
            const double relative_error_ratio =
                (expected_peak > std::numeric_limits<double>::epsilon())
                    ? max_abs_diff / expected_peak
                    : max_abs_diff;
            const double relative_error_percent = 100.0 * relative_error_ratio;

            channel_max_abs_diff[channel] = max_abs_diff;
            channel_max_rel_error_percent[channel] = relative_error_percent;
            const bool channel_passed =
                relative_error_ratio <= kExactConvolutionToleranceRatio;
            if (!channel_passed)
            {
                passed = false;
            }

            clear_validation_progress_line();
            std::cout << "[mock exact convolution] output channel " << channel
                      << (channel_passed ? " OK" : " SCREAM")
                      << ": max_abs_diff=" << channel_max_abs_diff[channel]
                      << " max_rel_error_percent=" << channel_max_rel_error_percent[channel]
                      << '\n';
        }

        if (channel_count > 0)
        {
            std::cout << "[mock exact convolution] finished validating "
                      << channel_count << " output channels\n";
        }

        const std::string base_name =
            sanitize_for_filename(
                validated_device.info.name ? validated_device.info.name : "mock_device");
        const std::string csv_path =
            std::string(PORTABLE_OUTPUT_DIR) + "/" +
            base_name + "_exact_convolution_validation.csv";
        if (save_exact_convolution_validation_csv(
                *exact_state,
                expected_outputs,
                sample_rate,
                csv_path))
        {
            std::cout << "[mock exact convolution] Saved validation CSV: "
                      << csv_path << '\n';
            std::cout << "[mock exact convolution] View command: "
                      << build_python_command(PORTABLE_EXACT_CONV_VIEW_SCRIPT, csv_path)
                      << '\n';
        }
        else
        {
            std::cerr << "[mock exact convolution] Failed to save validation CSV: "
                      << csv_path << '\n';
        }

        if (passed)
        {
            double worst_percent = 0.0;
            for (double value : channel_max_rel_error_percent)
            {
                worst_percent = std::max(worst_percent, value);
            }

            std::cout << "[mock exact convolution] OK: "
                      << (validated_device.info.name ? validated_device.info.name : "(null)")
                      << " matched the expected mixed-channel convolution within "
                      << kExactConvolutionTolerancePercent
                      << "% peak-relative error. Worst channel = "
                      << worst_percent << "%\n";
            return;
        }

        std::cerr << "[mock exact convolution] SCREAM: "
                  << (validated_device.info.name ? validated_device.info.name : "(null)")
                  << " did not match the expected mixed-channel convolution within "
                  << kExactConvolutionTolerancePercent
                  << "% peak-relative error.\n";
        for (size_t channel = 0; channel < channel_count; ++channel)
        {
            std::cerr << "  channel " << channel
                      << ": max_abs_diff=" << channel_max_abs_diff[channel]
                      << " max_rel_error_percent=" << channel_max_rel_error_percent[channel]
                      << '\n';
        }
    };

    return device_index;
}



PaDeviceIndex register_mock_device_inferred_from_topology(
    const char *name,
    const char *frequency_matrix_csv_path,
    int input_channels,
    int output_channels,
    double default_sample_rate,
    const std::vector<double> &supported_sample_rates)
{
    if (!name || !name[0])
    {
        throw std::invalid_argument(
            "register_mock_device_inferred_from_topology: device name must be non-empty");
    }

    if (!frequency_matrix_csv_path || !frequency_matrix_csv_path[0])
    {
        throw std::invalid_argument(
            "register_mock_device_inferred_from_topology: frequency CSV path must be non-empty");
    }

    if (input_channels <= 0 || output_channels <= 0 || default_sample_rate <= 0.0)
    {
        throw std::invalid_argument(
            "register_mock_device_inferred_from_topology: invalid device dimensions or sample rate");
    }

    std::vector<MockDevice> *devices = mock_pa_get_devices_list();
    if (!devices)
    {
        throw std::runtime_error(
            "register_mock_device_inferred_from_topology: device storage is null");
    }

    std::ifstream input(frequency_matrix_csv_path);
    if (!input.is_open())
    {
        throw std::runtime_error(
            "register_mock_device_inferred_from_topology: failed to open topology CSV");
    }

    const auto split_semicolon_line = [](const std::string &line) -> std::vector<std::string>
    {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ';'))
        {
            fields.push_back(field);
        }
        if (!line.empty() && line.back() == ';')
        {
            fields.push_back(std::string());
        }
        return fields;
    };

    std::string header_line;
    if (!std::getline(input, header_line))
    {
        throw std::runtime_error(
            "register_mock_device_inferred_from_topology: topology CSV is empty");
    }

    const std::vector<std::string> headers = split_semicolon_line(header_line);
    if (headers.size() < 3 || (headers.size() - 1) % 2 != 0)
    {
        throw std::runtime_error(
            "register_mock_device_inferred_from_topology: unexpected frequency CSV header");
    }

    std::vector<std::pair<int, int>> column_pairs;
    column_pairs.reserve((headers.size() - 1) / 2);
    for (size_t column = 1; column + 1 < headers.size(); column += 2)
    {
        int re_input = -1;
        int re_output = -1;
        int im_input = -1;
        int im_output = -1;
        if (std::sscanf(headers[column].c_str(), "Re H input %d <- output %d", &re_input, &re_output) != 2 ||
            std::sscanf(headers[column + 1].c_str(), "Im H input %d <- output %d", &im_input, &im_output) != 2)
        {
            throw std::runtime_error(
                "register_mock_device_inferred_from_topology: could not parse response headers");
        }

        if (re_input != im_input || re_output != im_output ||
            re_input < 0 || re_input >= input_channels ||
            re_output < 0 || re_output >= output_channels)
        {
            throw std::runtime_error(
                "register_mock_device_inferred_from_topology: topology headers do not match requested channel counts");
        }

        column_pairs.emplace_back(re_input, re_output);
    }

    std::vector<std::vector<std::vector<std::complex<float>>>> topology_frequency(
        static_cast<size_t>(input_channels),
        std::vector<std::vector<std::complex<float>>>(
            static_cast<size_t>(output_channels),
            std::vector<std::complex<float>>()));

    std::string row_line;
    while (std::getline(input, row_line))
    {
        if (row_line.empty())
        {
            continue;
        }

        const std::vector<std::string> fields = split_semicolon_line(row_line);
        if (fields.empty() || fields[0].empty())
        {
            continue;
        }

        for (size_t pair_index = 0; pair_index < column_pairs.size(); ++pair_index)
        {
            const size_t real_column = 1 + pair_index * 2;
            const size_t imag_column = real_column + 1;
            const float real_value =
                (real_column < fields.size() && !fields[real_column].empty())
                    ? std::stof(fields[real_column])
                    : 0.0f;
            const float imag_value =
                (imag_column < fields.size() && !fields[imag_column].empty())
                    ? std::stof(fields[imag_column])
                    : 0.0f;

            const int input_channel = column_pairs[pair_index].first;
            const int output_channel = column_pairs[pair_index].second;
            topology_frequency[static_cast<size_t>(input_channel)]
                              [static_cast<size_t>(output_channel)]
                .push_back(std::complex<float>(real_value, imag_value));
        }
    }

    std::vector<std::vector<std::vector<float>>> topology_impulses(
        static_cast<size_t>(input_channels),
        std::vector<std::vector<float>>(
            static_cast<size_t>(output_channels),
            std::vector<float>()));

    for (int input_channel = 0; input_channel < input_channels; ++input_channel)
    {
        for (int output_channel = 0; output_channel < output_channels; ++output_channel)
        {
            const std::vector<std::complex<float>> &frequency_response =
                topology_frequency[static_cast<size_t>(input_channel)]
                                  [static_cast<size_t>(output_channel)];
            topology_impulses[static_cast<size_t>(input_channel)]
                             [static_cast<size_t>(output_channel)] =
                frequency_response_half_to_time_domain(frequency_response);
        }
    }

    const float sample_dt = 1.0f / static_cast<float>(default_sample_rate);

    MockDevice device{};
    device.ownedName = name;
    device.info.name = device.ownedName.c_str();
    device.info.maxInputChannels = input_channels;
    device.info.maxOutputChannels = output_channels;
    device.info.defaultSampleRate = default_sample_rate;
    device.supportedSampleRates = supported_sample_rates;
    device.outputHistory.assign(
        static_cast<size_t>(output_channels),
        std::vector<float>());
    device.inputGenerators.assign(
        static_cast<size_t>(input_channels),
        MockInputGenerator());

    devices->push_back(device);
    const PaDeviceIndex device_index =
        static_cast<PaDeviceIndex>(devices->size() - 1);
    MockDevice &stored = devices->back();
    stored.info.name = stored.ownedName.c_str();

    for (int input_channel = 0; input_channel < input_channels; ++input_channel)
    {
        std::vector<std::vector<float>> sampled_kernels(
            static_cast<size_t>(output_channels),
            std::vector<float>());

        for (int output_channel = 0; output_channel < output_channels; ++output_channel)
        {
            const std::vector<float> &impulse_response =
                topology_impulses[static_cast<size_t>(input_channel)]
                                 [static_cast<size_t>(output_channel)];

            const Signal kernel_signal =
                infer_signal_from_vector(impulse_response, 0.0f, sample_dt);
            const Distribuition kernel_distribution =
                infer_distribution_from_signal(kernel_signal, sample_dt);
            const Signal sampled_kernel_signal =
                infer_signal_from_distribution(kernel_distribution, sample_dt);
            sampled_kernels[static_cast<size_t>(output_channel)] =
                sample_signal_to_vector(
                    sampled_kernel_signal,
                    0.0f,
                    static_cast<int>(impulse_response.size()),
                    sample_dt);
        }

        const std::shared_ptr<std::vector<std::vector<float>>> shared_sampled_kernels =
            std::make_shared<std::vector<std::vector<float>>>(std::move(sampled_kernels));

        stored.inputGenerators[static_cast<size_t>(input_channel)] =
            [device_index,
             sample_dt,
             shared_sampled_kernels](double t) -> float
        {
            std::vector<MockDevice> *live_devices = mock_pa_get_devices_list();
            if (!live_devices ||
                device_index < 0 ||
                device_index >= static_cast<PaDeviceIndex>(live_devices->size()))
            {
                return 0.0f;
            }

            const MockDevice &live_device = (*live_devices)[static_cast<size_t>(device_index)];
            const long long sample_index =
                std::llround(t / static_cast<double>(sample_dt));
            if (sample_index < 0)
            {
                return 0.0f;
            }

            const size_t output_channel_count =
                std::min(shared_sampled_kernels->size(), live_device.outputHistory.size());
            float accumulated_sample = 0.0f;
            for (size_t output_channel = 0; output_channel < output_channel_count; ++output_channel)
            {
                const std::vector<float> &kernel =
                    (*shared_sampled_kernels)[output_channel];
                const std::vector<float> &history =
                    live_device.outputHistory[output_channel];
                if (kernel.empty() || history.empty())
                {
                    continue;
                }

                const long long max_tap =
                    std::min<long long>(
                        sample_index,
                        static_cast<long long>(kernel.size()) - 1);
                const long long min_tap =
                    std::max<long long>(
                        0,
                        sample_index - (static_cast<long long>(history.size()) - 1));
                for (long long tap = min_tap; tap <= max_tap; ++tap)
                {
                    const long long history_index = sample_index - tap;
                    accumulated_sample +=
                        kernel[static_cast<size_t>(tap)] *
                        history[static_cast<size_t>(history_index)];
                }
            }

            return accumulated_sample;
        };
    }

    return device_index;
}
