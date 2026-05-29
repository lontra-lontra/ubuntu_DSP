/*
Build from repo root:
  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=detect_timming -DPORTABLE_USE_MOCK=OFF
  cmake --build Portable/build --target portable_detect_timming --parallel

If `Portable/build` does not exist yet, came from another machine, or you changed
`PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` line before
the `cmake --build ...` line.

Run:
  ./Portable/build/portable_detect_timming

To try mock audio instead, rerun the first command with `-DPORTABLE_USE_MOCK=ON`.

This app reopens the selected Linux audio device for each sample-rate /
buffer-size pair, captures 3 bursts, and writes the inferred delays for
every configuration.
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef MOCK
#error "MOCK must be defined by the build system for this target."
#endif

#define CHANNELS 2

#define DEVICE_NAME "default"

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef BURST_COUNT
#define BURST_COUNT 3
#endif

#ifndef BURST_PERIOD_SECONDS
#define BURST_PERIOD_SECONDS 1.0
#endif

#ifndef CAPTURE_TIME_SECONDS
#define CAPTURE_TIME_SECONDS (BURST_COUNT * BURST_PERIOD_SECONDS)
#endif

#ifndef BURST_SILENCE_SECONDS
#define BURST_SILENCE_SECONDS 0.9
#endif

#ifndef BURST_TONE_SECONDS
#define BURST_TONE_SECONDS 0.1
#endif

#ifndef BURST_FREQUENCY_HZ
#define BURST_FREQUENCY_HZ 400.0
#endif

#ifndef BURST_AMPLITUDE
#define BURST_AMPLITUDE 0.8
#endif

#ifndef DETECTION_RELATIVE_THRESHOLD
#define DETECTION_RELATIVE_THRESHOLD 0.10
#endif

#ifndef DETECTION_MINIMUM_ABSOLUTE_THRESHOLD
#define DETECTION_MINIMUM_ABSOLUTE_THRESHOLD 1.0e-4
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#ifndef PORTABLE_PLOT_SCRIPT
#define PORTABLE_PLOT_SCRIPT "Portable/scripts/plot.py"
#endif

#ifndef PORTABLE_DETECT_TIMMING_PLOT_SCRIPT
#define PORTABLE_DETECT_TIMMING_PLOT_SCRIPT "Portable/scripts/plot_detect_timming_results.py"
#endif

#ifndef STREAM_COOLDOWN_MS
#define STREAM_COOLDOWN_MS 1000
#endif

#include "portable/audio_helpers.h"
#include "portable/csv_utils.h"

namespace
{

constexpr double kPi = 3.14159265358979323846;

const std::vector<double> frequencies_to_test = {
    44100.0,
    48000.0,
    8200.0,
    96000.0,
    176400.0,
  //  192000.0
  };

// Example sizes for ALSA-backed devices. Adjust this list if your interface
// advertises a different set of supported buffer sizes.
const std::vector<unsigned long> buffer_sizes_to_test = {
    16ul,
    32ul,
    64ul,
   // 128ul,
   // 192ul,
    256ul,
   // 384ul,
    512ul,
   // 768ul,
    1024ul};

struct DetectTimmingData
{
    float *savedOutput0 = nullptr;
    float *savedInput0 = nullptr;
    float *savedOutput1 = nullptr;
    float *savedInput1 = nullptr;
    int frameIndex = 0;
    int maxFrames = 0;
    double sampleRate = 48000.0;
    int inputUnderflowCount = 0;
    int inputOverflowCount = 0;
    int outputUnderflowCount = 0;
    int outputOverflowCount = 0;
};

struct TimingMeasurement
{
    double sampleRate = 0.0;
    unsigned long bufferSize = 0;
    std::vector<int> burstDelaysSamples;
    std::vector<int> output0ToInput0DelaysSamples;
    std::vector<int> output0ToInput1DelaysSamples;
    bool captureCompleted = false;
    int inputUnderflowCount = 0;
    int inputOverflowCount = 0;
    int outputUnderflowCount = 0;
    int outputOverflowCount = 0;
};

float burst_sample_at_time(double time_seconds)
{
    if (time_seconds < 0.0)
    {
        return 0.0f;
    }

    const double burst_position = std::fmod(time_seconds, BURST_PERIOD_SECONDS);
    if (burst_position < BURST_SILENCE_SECONDS ||
        burst_position >= BURST_SILENCE_SECONDS + BURST_TONE_SECONDS)
    {
        return 0.0f;
    }

    const double local_time = burst_position - BURST_SILENCE_SECONDS;
    return static_cast<float>(
        BURST_AMPLITUDE * std::sin(2.0 * kPi * BURST_FREQUENCY_HZ * local_time));
}

double absolute_sample(double sample)
{
    return std::fabs(sample);
}

std::string build_python_command(
    const std::string &script_path,
    const std::string &csv_path)
{
    std::string command = "py -3 ";
    command += '"' + script_path + "\" \"" + csv_path + '"';
    return command;
}

int detect_timming_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    DetectTimmingData *data = static_cast<DetectTimmingData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data ||
        !output ||
        !data->savedOutput0 ||
        !data->savedInput0 ||
        !data->savedOutput1 ||
        !data->savedInput1)
    {
        return paAbort;
    }

    if ((statusFlags & paInputUnderflow) != 0)
    {
        data->inputUnderflowCount++;
    }
    if ((statusFlags & paInputOverflow) != 0)
    {
        data->inputOverflowCount++;
    }
    if ((statusFlags & paOutputUnderflow) != 0)
    {
        data->outputUnderflowCount++;
    }
    if ((statusFlags & paOutputOverflow) != 0)
    {
        data->outputOverflowCount++;
    }

    const int frames_left = data->maxFrames - data->frameIndex;
    const int frames_to_process =
        std::max(0, std::min(static_cast<int>(framesPerBuffer), frames_left));

    for (int i = 0; i < frames_to_process; ++i)
    {
        const int frame_index = data->frameIndex;
        const double time_seconds =
            static_cast<double>(frame_index) / data->sampleRate;

        const float output0 = burst_sample_at_time(time_seconds);
        const float input0 = input ? input[i * CHANNELS + 0] : 0.0f;
        const float input1 = input ? input[i * CHANNELS + 1] : 0.0f;

        output[i * CHANNELS + 0] = output0;
        output[i * CHANNELS + 1] = input0;

        data->savedOutput0[frame_index] = output0;
        data->savedInput0[frame_index] = input0;
        data->savedOutput1[frame_index] = input0;
        data->savedInput1[frame_index] = input1;
        data->frameIndex++;
    }

    for (int i = frames_to_process; i < static_cast<int>(framesPerBuffer); ++i)
    {
        for (int channel = 0; channel < CHANNELS; ++channel)
        {
            output[i * CHANNELS + channel] = 0.0f;
        }
    }

    if (data->frameIndex >= data->maxFrames)
    {
        return paComplete;
    }

    return paContinue;
}

double peak_absolute_in_range(
    const std::vector<float> &signal,
    int start_sample,
    int end_sample)
{
    const int clamped_start = std::max(0, start_sample);
    const int clamped_end = std::min(static_cast<int>(signal.size()), end_sample);
    double peak = 0.0;

    for (int sample = clamped_start; sample < clamped_end; ++sample)
    {
        peak = std::max(
            peak,
            absolute_sample(signal[static_cast<size_t>(sample)]));
    }

    return peak;
}

int find_first_activity_sample_in_range(
    const std::vector<float> &signal,
    int start_sample,
    int end_sample,
    double threshold)
{
    const int clamped_start = std::max(0, start_sample);
    const int clamped_end = std::min(static_cast<int>(signal.size()), end_sample);

    for (int sample = clamped_start; sample < clamped_end; ++sample)
    {
        if (absolute_sample(signal[static_cast<size_t>(sample)]) >= threshold)
        {
            return sample;
        }
    }

    return -1;
}

std::vector<int> infer_burst_onsets_samples(
    const std::vector<float> &signal,
    double sample_rate,
    double search_tail_seconds)
{
    std::vector<int> burst_onsets(static_cast<size_t>(BURST_COUNT), -1);
    const int pre_burst_guard_samples =
        static_cast<int>(std::llround(0.01 * sample_rate));
    const int search_tail_samples =
        static_cast<int>(std::llround(search_tail_seconds * sample_rate));

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        const int expected_burst_start = static_cast<int>(std::llround(
            (static_cast<double>(burst_index) * BURST_PERIOD_SECONDS +
             BURST_SILENCE_SECONDS) *
            sample_rate));
        const int search_start =
            expected_burst_start - pre_burst_guard_samples;
        const int search_end = static_cast<int>(std::llround(
            expected_burst_start +
            (BURST_TONE_SECONDS * sample_rate) +
            search_tail_samples));

        const double peak =
            peak_absolute_in_range(signal, search_start, search_end);
        const double threshold =
            std::max(
                static_cast<double>(DETECTION_MINIMUM_ABSOLUTE_THRESHOLD),
                peak * DETECTION_RELATIVE_THRESHOLD);

        if (peak < DETECTION_MINIMUM_ABSOLUTE_THRESHOLD)
        {
            continue;
        }

        burst_onsets[static_cast<size_t>(burst_index)] =
            find_first_activity_sample_in_range(
                signal,
                search_start,
                search_end,
                threshold);
    }

    return burst_onsets;
}

std::vector<int> subtract_burst_onsets(
    const std::vector<int> &reference_onsets,
    const std::vector<int> &measured_onsets)
{
    const size_t burst_count =
        std::min(reference_onsets.size(), measured_onsets.size());
    std::vector<int> delays_samples(burst_count, -1);

    for (size_t burst_index = 0; burst_index < burst_count; ++burst_index)
    {
        if (reference_onsets[burst_index] >= 0 && measured_onsets[burst_index] >= 0)
        {
            delays_samples[burst_index] =
                measured_onsets[burst_index] - reference_onsets[burst_index];
        }
    }

    return delays_samples;
}

bool save_last_bursts_ploted_csv(
    const std::vector<float> &output0,
    const std::vector<float> &input0,
    const std::vector<float> &output1,
    const std::vector<float> &input1,
    int input_underflow_count,
    int input_overflow_count,
    int output_underflow_count,
    int output_overflow_count,
    double sample_rate,
    const std::string &csv_path)
{
    const size_t sample_count = output0.size();
    if (input0.size() != sample_count ||
        output1.size() != sample_count ||
        input1.size() != sample_count)
    {
        return false;
    }

    std::vector<float> time_axis(sample_count, 0.0f);
    std::vector<float> abs_output0(sample_count, 0.0f);
    std::vector<float> abs_input0(sample_count, 0.0f);
    std::vector<float> abs_input1(sample_count, 0.0f);
    std::vector<float> input_underflows(sample_count, static_cast<float>(input_underflow_count));
    std::vector<float> input_overflows(sample_count, static_cast<float>(input_overflow_count));
    std::vector<float> output_underflows(sample_count, static_cast<float>(output_underflow_count));
    std::vector<float> output_overflows(sample_count, static_cast<float>(output_overflow_count));
    for (size_t i = 0; i < sample_count; ++i)
    {
        time_axis[i] = static_cast<float>(static_cast<double>(i) / sample_rate);
        abs_output0[i] = static_cast<float>(absolute_sample(output0[i]));
        abs_input0[i] = static_cast<float>(absolute_sample(input0[i]));
        abs_input1[i] = static_cast<float>(absolute_sample(input1[i]));
    }

    return save_arrays_to_csv(
        csv_path,
        {
            "time (s)",
            "output 0",
            "input 0",
            "output 1",
            "input 1",
            "abs output 0",
            "abs input 0",
            "abs input 1",
            "input_underflows",
            "input_overflows",
            "output_underflows",
            "output_overflows"},
        {
            &time_axis,
            &output0,
            &input0,
            &output1,
            &input1,
            &abs_output0,
            &abs_input0,
            &abs_input1,
            &input_underflows,
            &input_overflows,
            &output_underflows,
            &output_overflows});
}

bool save_timming_results_csv(
    const std::vector<TimingMeasurement> &results,
    const std::string &csv_path)
{
    std::vector<float> sample_rates;
    std::vector<float> buffer_sizes;
    std::vector<float> capture_completed;
    std::vector<float> input_underflows;
    std::vector<float> input_overflows;
    std::vector<float> output_underflows;
    std::vector<float> output_overflows;
    std::vector<std::vector<float>> delay_samples_columns(
        static_cast<size_t>(BURST_COUNT),
        std::vector<float>());
    std::vector<std::vector<float>> delay_seconds_columns(
        static_cast<size_t>(BURST_COUNT),
        std::vector<float>());
    std::vector<std::vector<float>> relay_delay_samples_columns(
        static_cast<size_t>(BURST_COUNT),
        std::vector<float>());
    std::vector<std::vector<float>> relay_delay_seconds_columns(
        static_cast<size_t>(BURST_COUNT),
        std::vector<float>());
    std::vector<std::vector<float>> output0_to_input0_delay_samples_columns(
        static_cast<size_t>(BURST_COUNT),
        std::vector<float>());
    std::vector<std::vector<float>> output0_to_input0_delay_seconds_columns(
        static_cast<size_t>(BURST_COUNT),
        std::vector<float>());
    std::vector<std::vector<float>> output0_to_input1_delay_samples_columns(
        static_cast<size_t>(BURST_COUNT),
        std::vector<float>());
    std::vector<std::vector<float>> output0_to_input1_delay_seconds_columns(
        static_cast<size_t>(BURST_COUNT),
        std::vector<float>());

    sample_rates.reserve(results.size());
    buffer_sizes.reserve(results.size());
    capture_completed.reserve(results.size());
    input_underflows.reserve(results.size());
    input_overflows.reserve(results.size());
    output_underflows.reserve(results.size());
    output_overflows.reserve(results.size());
    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        delay_samples_columns[static_cast<size_t>(burst_index)].reserve(results.size());
        delay_seconds_columns[static_cast<size_t>(burst_index)].reserve(results.size());
        relay_delay_samples_columns[static_cast<size_t>(burst_index)].reserve(results.size());
        relay_delay_seconds_columns[static_cast<size_t>(burst_index)].reserve(results.size());
        output0_to_input0_delay_samples_columns[static_cast<size_t>(burst_index)].reserve(results.size());
        output0_to_input0_delay_seconds_columns[static_cast<size_t>(burst_index)].reserve(results.size());
        output0_to_input1_delay_samples_columns[static_cast<size_t>(burst_index)].reserve(results.size());
        output0_to_input1_delay_seconds_columns[static_cast<size_t>(burst_index)].reserve(results.size());
    }

    const auto append_delay_set =
        [](const std::vector<int> &delays_samples_source,
           double sample_rate,
           std::vector<std::vector<float>> &samples_columns,
           std::vector<std::vector<float>> &seconds_columns)
    {
        for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
        {
            float delay_samples = -1.0f;
            float delay_seconds = -1.0f;
            if (burst_index < static_cast<int>(delays_samples_source.size()) &&
                delays_samples_source[static_cast<size_t>(burst_index)] >= 0)
            {
                delay_samples = static_cast<float>(
                    delays_samples_source[static_cast<size_t>(burst_index)]);
                delay_seconds = delay_samples /
                                static_cast<float>(std::max(1.0, sample_rate));
            }

            samples_columns[static_cast<size_t>(burst_index)].push_back(delay_samples);
            seconds_columns[static_cast<size_t>(burst_index)].push_back(delay_seconds);
        }
    };

    for (const TimingMeasurement &result : results)
    {
        sample_rates.push_back(static_cast<float>(result.sampleRate));
        buffer_sizes.push_back(static_cast<float>(result.bufferSize));
        capture_completed.push_back(result.captureCompleted ? 1.0f : 0.0f);
        input_underflows.push_back(static_cast<float>(result.inputUnderflowCount));
        input_overflows.push_back(static_cast<float>(result.inputOverflowCount));
        output_underflows.push_back(static_cast<float>(result.outputUnderflowCount));
        output_overflows.push_back(static_cast<float>(result.outputOverflowCount));

        append_delay_set(
            result.burstDelaysSamples,
            result.sampleRate,
            delay_samples_columns,
            delay_seconds_columns);
        append_delay_set(
            result.burstDelaysSamples,
            result.sampleRate,
            relay_delay_samples_columns,
            relay_delay_seconds_columns);
        append_delay_set(
            result.output0ToInput0DelaysSamples,
            result.sampleRate,
            output0_to_input0_delay_samples_columns,
            output0_to_input0_delay_seconds_columns);
        append_delay_set(
            result.output0ToInput1DelaysSamples,
            result.sampleRate,
            output0_to_input1_delay_samples_columns,
            output0_to_input1_delay_seconds_columns);
    }

    std::vector<std::string> headers = {
        "sample_rate_hz",
        "buffer_size_samples",
        "capture_completed",
        "input_underflows",
        "input_overflows",
        "output_underflows",
        "output_overflows"};
    std::vector<const std::vector<float> *> columns = {
        &sample_rates,
        &buffer_sizes,
        &capture_completed,
        &input_underflows,
        &input_overflows,
        &output_underflows,
        &output_overflows};

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        headers.push_back("delay_burst_" + std::to_string(burst_index + 1) + "_samples");
        columns.push_back(&delay_samples_columns[static_cast<size_t>(burst_index)]);
    }

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        headers.push_back("delay_input0_to_input1_burst_" + std::to_string(burst_index + 1) + "_samples");
        columns.push_back(&relay_delay_samples_columns[static_cast<size_t>(burst_index)]);
    }

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        headers.push_back("delay_output0_to_input0_burst_" + std::to_string(burst_index + 1) + "_samples");
        columns.push_back(&output0_to_input0_delay_samples_columns[static_cast<size_t>(burst_index)]);
    }

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        headers.push_back("delay_output0_to_input1_burst_" + std::to_string(burst_index + 1) + "_samples");
        columns.push_back(&output0_to_input1_delay_samples_columns[static_cast<size_t>(burst_index)]);
    }

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        headers.push_back("delay_burst_" + std::to_string(burst_index + 1) + "_seconds");
        columns.push_back(&delay_seconds_columns[static_cast<size_t>(burst_index)]);
    }

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        headers.push_back("delay_input0_to_input1_burst_" + std::to_string(burst_index + 1) + "_seconds");
        columns.push_back(&relay_delay_seconds_columns[static_cast<size_t>(burst_index)]);
    }

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        headers.push_back("delay_output0_to_input0_burst_" + std::to_string(burst_index + 1) + "_seconds");
        columns.push_back(&output0_to_input0_delay_seconds_columns[static_cast<size_t>(burst_index)]);
    }

    for (int burst_index = 0; burst_index < BURST_COUNT; ++burst_index)
    {
        headers.push_back("delay_output0_to_input1_burst_" + std::to_string(burst_index + 1) + "_seconds");
        columns.push_back(&output0_to_input1_delay_seconds_columns[static_cast<size_t>(burst_index)]);
    }

    return save_arrays_to_csv(csv_path, headers, columns);
}

bool run_timming_measurement(
    int device_index,
    double sample_rate,
    unsigned long buffer_size,
    TimingMeasurement &measurement)
{
    measurement.sampleRate = sample_rate;
    measurement.bufferSize = buffer_size;
    measurement.burstDelaysSamples.assign(static_cast<size_t>(BURST_COUNT), -1);
    measurement.output0ToInput0DelaysSamples.assign(static_cast<size_t>(BURST_COUNT), -1);
    measurement.output0ToInput1DelaysSamples.assign(static_cast<size_t>(BURST_COUNT), -1);
    measurement.captureCompleted = false;
    measurement.inputUnderflowCount = 0;
    measurement.inputOverflowCount = 0;
    measurement.outputUnderflowCount = 0;
    measurement.outputOverflowCount = 0;

    const int capture_samples =
        static_cast<int>(std::llround(CAPTURE_TIME_SECONDS * sample_rate));
    std::vector<float> saved_output0(static_cast<size_t>(capture_samples), 0.0f);
    std::vector<float> saved_input0(static_cast<size_t>(capture_samples), 0.0f);
    std::vector<float> saved_output1(static_cast<size_t>(capture_samples), 0.0f);
    std::vector<float> saved_input1(static_cast<size_t>(capture_samples), 0.0f);

    DetectTimmingData data{};
    data.savedOutput0 = saved_output0.data();
    data.savedInput0 = saved_input0.data();
    data.savedOutput1 = saved_output1.data();
    data.savedInput1 = saved_input1.data();
    data.maxFrames = capture_samples;
    data.sampleRate = sample_rate;

    PaStreamParameters input_parameters{};
    input_parameters.device = device_index;
    input_parameters.channelCount = CHANNELS;
    input_parameters.sampleFormat = SAMPLE_FORMAT;
    input_parameters.suggestedLatency = static_cast<double>(buffer_size) / sample_rate;
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = CHANNELS;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = static_cast<double>(buffer_size) / sample_rate;
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    if (!check_if_device_respects_input_and_output_stream_specs(
            &input_parameters,
            &output_parameters,
            sample_rate))
    {
        std::cerr << "Skipping sample_rate=" << sample_rate
                  << " buffer_size=" << buffer_size
                  << " because Pa_IsFormatSupported failed.\n";
        return false;
    }

    PaStream *stream = nullptr;
    PaError err = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        sample_rate,
        buffer_size,
        paClipOff,
        detect_timming_callback,
        &data);
    if (err != paNoError)
    {
        std::cerr << "Pa_OpenStream failed for sample_rate=" << sample_rate
                  << " buffer_size=" << buffer_size
                  << ": " << Pa_GetErrorText(err) << '\n';
        return false;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError)
    {
        std::cerr << "Pa_StartStream failed for sample_rate=" << sample_rate
                  << " buffer_size=" << buffer_size
                  << ": " << Pa_GetErrorText(err) << '\n';
        Pa_CloseStream(stream);
        return false;
    }

    while (Pa_IsStreamActive(stream) == 1)
    {
        Pa_Sleep(20);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Sleep(STREAM_COOLDOWN_MS);

    const std::string last_bursts_ploted_csv_path =
        std::string(PORTABLE_OUTPUT_DIR) + "/last_bursts_ploted.csv";
    if (save_last_bursts_ploted_csv(
            saved_output0,
            saved_input0,
            saved_output1,
            saved_input1,
            data.inputUnderflowCount,
            data.inputOverflowCount,
            data.outputUnderflowCount,
            data.outputOverflowCount,
            sample_rate,
            last_bursts_ploted_csv_path))
    {
        std::cout << "Saved last bursts plot CSV: " << last_bursts_ploted_csv_path << '\n';
        std::cout << "Plot command: "
                  << build_python_command(PORTABLE_PLOT_SCRIPT, last_bursts_ploted_csv_path)
                  << '\n';
    }
    else
    {
        std::cerr << "Failed to save last bursts plot CSV: "
                  << last_bursts_ploted_csv_path << '\n';
    }

    const std::vector<int> output0_onsets =
        infer_burst_onsets_samples(saved_output0, sample_rate, 0.05);
    const std::vector<int> input0_onsets =
        infer_burst_onsets_samples(saved_input0, sample_rate, 0.20);
    const std::vector<int> input1_onsets =
        infer_burst_onsets_samples(saved_input1, sample_rate, 0.20);

    // Primary delay:
    // compare the two captured inputs so the first acoustic / electrical leg
    // (output 0 -> input 0) is not baked into the main timing number.
    measurement.burstDelaysSamples =
        subtract_burst_onsets(input0_onsets, input1_onsets);

    // Keep the two end-to-end paths too, so we can still inspect the full chain.
    measurement.output0ToInput0DelaysSamples =
        subtract_burst_onsets(output0_onsets, input0_onsets);
    measurement.output0ToInput1DelaysSamples =
        subtract_burst_onsets(output0_onsets, input1_onsets);
    measurement.captureCompleted = true;
    measurement.inputUnderflowCount = data.inputUnderflowCount;
    measurement.inputOverflowCount = data.inputOverflowCount;
    measurement.outputUnderflowCount = data.outputUnderflowCount;
    measurement.outputOverflowCount = data.outputOverflowCount;
    return true;
}

void print_measurement_summary(const TimingMeasurement &measurement)
{
    std::cout << "sample_rate=" << measurement.sampleRate
              << " buffer_size=" << measurement.bufferSize
              << " relay_delays(input1-input0 samples)=";

    for (size_t i = 0; i < measurement.burstDelaysSamples.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << measurement.burstDelaysSamples[i];
    }

    std::cout << " | output0->input0=";
    for (size_t i = 0; i < measurement.output0ToInput0DelaysSamples.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << measurement.output0ToInput0DelaysSamples[i];
    }

    std::cout << " | output0->input1=";
    for (size_t i = 0; i < measurement.output0ToInput1DelaysSamples.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << measurement.output0ToInput1DelaysSamples[i];
    }

    std::cout << " | underflows(in/out)="
              << measurement.inputUnderflowCount
              << '/'
              << measurement.outputUnderflowCount
              << " | overflows(in/out)="
              << measurement.inputOverflowCount
              << '/'
              << measurement.outputOverflowCount
              << '\n';
}

} // namespace

int main()
{
    const std::string results_csv_path =
        std::string(PORTABLE_OUTPUT_DIR) + "/detect_timming_results.csv";

    const PaError init_err = Pa_Initialize();
    if (init_err != paNoError)
    {
        std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(init_err) << '\n';
        return 1;
    }

    const int device_index = select_device_by_name(DEVICE_NAME, true);
    if (device_index < 0)
    {
        Pa_Terminate();
        return 1;
    }

    const PaDeviceInfo *device_info = Pa_GetDeviceInfo(device_index);
    if (!device_info)
    {
        std::cerr << "Pa_GetDeviceInfo failed for device " << device_index << '\n';
        Pa_Terminate();
        return 1;
    }

    const PaHostApiInfo *host_api_info = Pa_GetHostApiInfo(device_info->hostApi);
    std::cout << "Selected device: [" << device_index << "] "
              << (device_info->name ? device_info->name : "(null)")
              << " | hostApi="
              << (host_api_info && host_api_info->name ? host_api_info->name : "(null)")
              << " | in=" << device_info->maxInputChannels
              << " out=" << device_info->maxOutputChannels
              << " defaultSR=" << device_info->defaultSampleRate
              << '\n';
    std::cout << "frequencies_to_test=";
    for (size_t i = 0; i < frequencies_to_test.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << frequencies_to_test[i];
    }
    std::cout << '\n';

    std::cout << "buffer_sizes_to_test=";
    for (size_t i = 0; i < buffer_sizes_to_test.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << buffer_sizes_to_test[i];
    }
    std::cout << '\n';

    const double estimated_total_seconds =
        static_cast<double>(frequencies_to_test.size() * buffer_sizes_to_test.size()) *
        (CAPTURE_TIME_SECONDS + static_cast<double>(STREAM_COOLDOWN_MS) / 1000.0);
    std::cout << "Estimated duration ~= " << estimated_total_seconds << " seconds\n";

    std::vector<TimingMeasurement> results;
    results.reserve(frequencies_to_test.size() * buffer_sizes_to_test.size());

    for (double sample_rate : frequencies_to_test)
    {
        for (unsigned long buffer_size : buffer_sizes_to_test)
        {
            std::cout << "Testing sample_rate=" << sample_rate
                      << " buffer_size=" << buffer_size << "...\n";

            TimingMeasurement measurement{};
            run_timming_measurement(
                device_index,
                sample_rate,
                buffer_size,
                measurement);
            print_measurement_summary(measurement);
            results.push_back(measurement);
        }
    }

    Pa_Terminate();

    if (!save_timming_results_csv(results, results_csv_path))
    {
        std::cerr << "Failed to save CSV: " << results_csv_path << '\n';
        return 1;
    }

    std::cout << "Saved CSV: " << results_csv_path << '\n';
    std::cout << "Analysis script command: "
              << build_python_command(
                     PORTABLE_DETECT_TIMMING_PLOT_SCRIPT,
                     results_csv_path)
              << '\n';
    std::cout << "Primary delay_burst_* columns now store input 1 - input 0 relay delay.\n";
    std::cout << "The CSV also keeps output0->input0 and output0->input1 delays for each burst.\n";

    return 0;
}
