/*
Build from repo root:
  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=multi_conv_benchmarking -DPORTABLE_USE_MOCK=OFF
  cmake --build Portable/build --target portable_multi_conv_benchmarking --parallel

If `Portable/build` does not exist yet, came from another machine, or you changed
`PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` line before
the `cmake --build ...` line.

Run:
  ./Portable/build/portable_multi_conv_benchmarking

Important config:
  mock/real audio is selected by `-DPORTABLE_USE_MOCK=ON/OFF` in the first command.
  #define CHANNELS 2
  #define DEVICE_NAME ...

To switch to mock audio, rerun the first command with `-DPORTABLE_USE_MOCK=ON`.
*/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
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

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 32
#endif

#ifndef FFT_PARTITION_SIZE
#define FFT_PARTITION_SIZE 128
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 48000
#endif

#ifndef PLAYBACK_TIME_SECONDS
#define PLAYBACK_TIME_SECONDS 1.0
#endif

#if MOCK
#define DEVICE_NAME "portable_mock_device"
#else
#define DEVICE_NAME "default"
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#ifndef CSV_TO_INFER_TOPOLOGY_FROM
#define CSV_TO_INFER_TOPOLOGY_FROM "infer_topology_selected_device_channels_2_frequency_domain_matrix"
#endif

#if MOCK
#include "portable/mockportaudio.h"
#include "portable/mock_devices.h"
#else
#include <portaudio.h>
#endif

#include "portable/audio_helpers.h"
#include "portable/csv_utils.h"
#include "portable/partitioned_fft_convolver.h"
#include "portable/plotting.h"

struct MultiConvData
{
    int frame_index = 0;
    int max_frames = 0;
    int staged_input_count = 0;

    int output_ring_capacity = 0;
    int output_ring_read_index = 0;
    int output_ring_write_index = 0;
    int output_ring_count = 0;

    std::vector<PartitionedFftConvolver> convolvers;
    std::vector<std::vector<float>> block_inputs;
    std::vector<std::vector<float>> block_outputs;
    std::vector<std::vector<float>> output_ring;

    std::vector<std::vector<float>> input_history;
    std::vector<std::vector<float>> output_history;
};

static std::vector<std::vector<float>> build_random_kernels()
{
    const int kernel_size = static_cast<int>(SAMPLE_RATE * 0.5f);
    std::vector<std::vector<float>> kernels(
        static_cast<size_t>(CHANNELS),
        std::vector<float>(static_cast<size_t>(kernel_size), 0.0f));

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    for (int ch = 0; ch < CHANNELS; ++ch)
    {
        for (int n = 0; n < kernel_size; ++n)
        {
            const float decay =
                std::exp(-4.0f * static_cast<float>(n) /
                         static_cast<float>(std::max(1, kernel_size)));
            kernels[static_cast<size_t>(ch)][static_cast<size_t>(n)] =
                0.02f * uni(rng) * decay;
        }
        kernels[static_cast<size_t>(ch)][0] += 0.6f;
    }

    return kernels;
}

static std::vector<std::vector<float>> compute_expected_outputs_naive(
    const std::vector<std::vector<float>> &input_history,
    const std::vector<std::vector<float>> &kernels)
{
    std::vector<std::vector<float>> expected(
        static_cast<size_t>(CHANNELS),
        std::vector<float>());

    if (input_history.size() != static_cast<size_t>(CHANNELS) ||
        kernels.size() != static_cast<size_t>(CHANNELS))
    {
        return expected;
    }

    const int total_frames = static_cast<int>(input_history[0].size());
    for (int ch = 0; ch < CHANNELS; ++ch)
    {
        expected[static_cast<size_t>(ch)].assign(
            static_cast<size_t>(total_frames),
            0.0f);

        const std::vector<float> &x = input_history[static_cast<size_t>(ch)];
        const std::vector<float> &h = kernels[static_cast<size_t>(ch)];
        const int h_size = static_cast<int>(h.size());

        for (int n = 0; n < total_frames; ++n)
        {
            double acc = 0.0;
            const int k_max = std::min(n, h_size - 1);
            for (int k = 0; k <= k_max; ++k)
            {
                acc += static_cast<double>(h[static_cast<size_t>(k)]) *
                       static_cast<double>(x[static_cast<size_t>(n - k)]);
            }
            expected[static_cast<size_t>(ch)][static_cast<size_t>(n)] =
                static_cast<float>(acc);
        }
    }

    return expected;
}

static std::vector<std::vector<float>> delay_outputs_for_comparison(
    const std::vector<std::vector<float>> &expected_output_history,
    int delay_samples)
{
    std::vector<std::vector<float>> delayed(
        static_cast<size_t>(CHANNELS),
        std::vector<float>());

    if (expected_output_history.size() != static_cast<size_t>(CHANNELS))
    {
        return delayed;
    }

    const int total_frames = static_cast<int>(expected_output_history[0].size());
    delay_samples = std::max(0, delay_samples);

    for (int ch = 0; ch < CHANNELS; ++ch)
    {
        delayed[static_cast<size_t>(ch)].assign(
            static_cast<size_t>(total_frames),
            0.0f);
        for (int n = delay_samples; n < total_frames; ++n)
        {
            delayed[static_cast<size_t>(ch)][static_cast<size_t>(n)] =
                expected_output_history[static_cast<size_t>(ch)]
                                      [static_cast<size_t>(n - delay_samples)];
        }
    }

    return delayed;
}

static int find_worst_channel_by_squared_error(
    const std::vector<std::vector<float>> &output_history,
    const std::vector<std::vector<float>> &expected_output_history,
    std::vector<double> &per_channel_sse)
{
    per_channel_sse.assign(static_cast<size_t>(CHANNELS), 0.0);

    int worst_channel = 0;
    double worst_sse = -std::numeric_limits<double>::infinity();

    for (int ch = 0; ch < CHANNELS; ++ch)
    {
        const std::vector<float> &y = output_history[static_cast<size_t>(ch)];
        const std::vector<float> &y_expected =
            expected_output_history[static_cast<size_t>(ch)];
        const int sample_count =
            std::min(static_cast<int>(y.size()), static_cast<int>(y_expected.size()));

        double sse = 0.0;
        for (int n = 0; n < sample_count; ++n)
        {
            const double diff =
                static_cast<double>(y[static_cast<size_t>(n)]) -
                static_cast<double>(y_expected[static_cast<size_t>(n)]);
            sse += diff * diff;
        }

        per_channel_sse[static_cast<size_t>(ch)] = sse;
        if (sse > worst_sse)
        {
            worst_sse = sse;
            worst_channel = ch;
        }
    }

    return worst_channel;
}

static int pa_multi_conv_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags,
    void *userData)
{
    MultiConvData *data = static_cast<MultiConvData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data || !output)
    {
        return paAbort;
    }

    const int frames_left = data->max_frames - data->frame_index;
    const int frames_to_process =
        std::max(0, std::min(static_cast<int>(framesPerBuffer), frames_left));

    for (int i = 0; i < frames_to_process; ++i)
    {
        const int n = data->frame_index + i;
        const int frame_offset = i * CHANNELS;

        for (int ch = 0; ch < CHANNELS; ++ch)
        {
            const float x = input ? input[frame_offset + ch] : 0.0f;
            data->block_inputs[static_cast<size_t>(ch)]
                             [static_cast<size_t>(data->staged_input_count)] = x;
            data->input_history[static_cast<size_t>(ch)][static_cast<size_t>(n)] = x;

            const float y =
                (data->output_ring_count > 0)
                    ? data->output_ring[static_cast<size_t>(ch)]
                                       [static_cast<size_t>(data->output_ring_read_index)]
                    : 0.0f;
            output[frame_offset + ch] = y;
            data->output_history[static_cast<size_t>(ch)][static_cast<size_t>(n)] = y;
        }

        if (data->output_ring_count > 0)
        {
            data->output_ring_read_index++;
            if (data->output_ring_read_index >= data->output_ring_capacity)
            {
                data->output_ring_read_index = 0;
            }
            data->output_ring_count--;
        }

        data->staged_input_count++;
        if (data->staged_input_count >= FFT_PARTITION_SIZE)
        {
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                data->convolvers[static_cast<size_t>(ch)].process_block(
                    data->block_inputs[static_cast<size_t>(ch)],
                    FFT_PARTITION_SIZE,
                    data->block_outputs[static_cast<size_t>(ch)]);
            }

            if (data->output_ring_count + FFT_PARTITION_SIZE <= data->output_ring_capacity)
            {
                for (int s = 0; s < FFT_PARTITION_SIZE; ++s)
                {
                    const int write_index =
                        (data->output_ring_write_index + s) % data->output_ring_capacity;
                    for (int ch = 0; ch < CHANNELS; ++ch)
                    {
                        data->output_ring[static_cast<size_t>(ch)][static_cast<size_t>(write_index)] =
                            data->block_outputs[static_cast<size_t>(ch)][static_cast<size_t>(s)];
                    }
                }

                data->output_ring_write_index =
                    (data->output_ring_write_index + FFT_PARTITION_SIZE) %
                    data->output_ring_capacity;
                data->output_ring_count += FFT_PARTITION_SIZE;
            }

            data->staged_input_count = 0;
        }
    }

    for (int i = frames_to_process; i < static_cast<int>(framesPerBuffer); ++i)
    {
        const int frame_offset = i * CHANNELS;
        for (int ch = 0; ch < CHANNELS; ++ch)
        {
            output[frame_offset + ch] = 0.0f;
        }
    }

    data->frame_index += frames_to_process;
    if (data->frame_index >= data->max_frames)
    {
        return paComplete;
    }

    return paContinue;
}

int main()
{
    const int total_frames =
        static_cast<int>(std::ceil(PLAYBACK_TIME_SECONDS * SAMPLE_RATE));

    PaError err = Pa_Initialize();
    if (err != paNoError)
    {
        std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(err) << '\n';
        return 1;
    }

#if MOCK
    register_mock_devices();
#endif

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

    PaStreamParameters input_parameters{};
    input_parameters.device = device_index;
    input_parameters.channelCount = CHANNELS;
    input_parameters.sampleFormat = SAMPLE_FORMAT;
    input_parameters.suggestedLatency = device_info->defaultLowInputLatency;
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = CHANNELS;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = device_info->defaultLowOutputLatency;
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    if (!check_if_device_respects_input_and_output_stream_specs(
            &input_parameters,
            &output_parameters,
            SAMPLE_RATE))
    {
        Pa_Terminate();
        return 1;
    }

    const std::vector<std::vector<float>> kernels = build_random_kernels();

    MultiConvData data{};
    data.frame_index = 0;
    data.max_frames = total_frames;
    data.staged_input_count = 0;
    data.output_ring_capacity = 8 * FFT_PARTITION_SIZE;
    data.output_ring_read_index = 0;
    data.output_ring_write_index = 0;
    data.output_ring_count = 0;
    data.convolvers.resize(static_cast<size_t>(CHANNELS));
    data.block_inputs.assign(
        static_cast<size_t>(CHANNELS),
        std::vector<float>(static_cast<size_t>(FFT_PARTITION_SIZE), 0.0f));
    data.block_outputs.assign(
        static_cast<size_t>(CHANNELS),
        std::vector<float>(static_cast<size_t>(FFT_PARTITION_SIZE), 0.0f));
    data.output_ring.assign(
        static_cast<size_t>(CHANNELS),
        std::vector<float>(static_cast<size_t>(data.output_ring_capacity), 0.0f));
    data.input_history.assign(
        static_cast<size_t>(CHANNELS),
        std::vector<float>(static_cast<size_t>(total_frames), 0.0f));
    data.output_history.assign(
        static_cast<size_t>(CHANNELS),
        std::vector<float>(static_cast<size_t>(total_frames), 0.0f));

    for (int ch = 0; ch < CHANNELS; ++ch)
    {
        if (!data.convolvers[static_cast<size_t>(ch)].initialize(
                kernels[static_cast<size_t>(ch)],
                FFT_PARTITION_SIZE))
        {
            std::cerr << "Failed to initialize FFT convolver for channel " << ch << '\n';
            Pa_Terminate();
            return 1;
        }
    }

    PaStream *stream = nullptr;
    err = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        pa_multi_conv_callback,
        &data);
    if (err != paNoError)
    {
        std::cerr << "Pa_OpenStream failed: " << Pa_GetErrorText(err) << '\n';
        Pa_Terminate();
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError)
    {
        std::cerr << "Pa_StartStream failed: " << Pa_GetErrorText(err) << '\n';
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    while (Pa_IsStreamActive(stream) == 1)
    {
        Pa_Sleep(20);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::vector<float> time_axis(static_cast<size_t>(total_frames), 0.0f);
    for (int i = 0; i < total_frames; ++i)
    {
        time_axis[static_cast<size_t>(i)] =
            static_cast<float>(i) / static_cast<float>(SAMPLE_RATE);
    }

    const std::vector<std::vector<float>> expected_output_history =
        compute_expected_outputs_naive(data.input_history, kernels);
    const std::vector<std::vector<float>> delayed_expected_output_history =
        delay_outputs_for_comparison(expected_output_history, FFT_PARTITION_SIZE);

    std::vector<double> per_channel_sse;
    const int worst_channel = find_worst_channel_by_squared_error(
        data.output_history,
        delayed_expected_output_history,
        per_channel_sse);

    const std::vector<float> &worst_input =
        data.input_history[static_cast<size_t>(worst_channel)];
    const std::vector<float> &worst_output =
        data.output_history[static_cast<size_t>(worst_channel)];
    const std::vector<float> &worst_expected =
        delayed_expected_output_history[static_cast<size_t>(worst_channel)];

    const std::string csv_path =
        std::string(PORTABLE_OUTPUT_DIR) +
        "/multi_conv_benchmarking_worst_channel_validation.csv";
    if (!save_arrays_to_csv(
            csv_path,
            {
                "time (s)",
                "input[worst]",
                "output[worst]",
                "expected_output[worst] (naive, delay-aligned)"
            },
            {&time_axis, &worst_input, &worst_output, &worst_expected}))
    {
        std::cerr << "Failed to save CSV: " << csv_path << '\n';
        return 1;
    }

    std::cout << "Saved: " << csv_path << '\n';
    plot_csv_with_portable_script(csv_path);
    std::cout << "Worst channel by SSE: " << worst_channel
              << " (SSE=" << per_channel_sse[static_cast<size_t>(worst_channel)]
              << ")\n";
    std::cout << "Build mode: " << (MOCK ? "mock" : "real") << '\n';
    std::cout << "CHANNELS=" << CHANNELS
              << " FRAMES_PER_BUFFER=" << FRAMES_PER_BUFFER
              << " FFT_PARTITION_SIZE=" << FFT_PARTITION_SIZE
              << " SAMPLE_RATE=" << SAMPLE_RATE << '\n';

    return 0;
}
