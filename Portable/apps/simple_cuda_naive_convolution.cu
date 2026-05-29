/*
Build from repo root:

  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=simple_cuda_naive_convolution -DPORTABLE_USE_MOCK=ON
  cmake --build Portable/build --target portable_simple_cuda_naive_convolution --parallel

Run:
  ./Portable/build/portable_simple_cuda_naive_convolution

To try real PortAudio instead, rerun the first command with `-DPORTABLE_USE_MOCK=OFF`.

This is a deliberately naive CUDA FIR example.
Each output/input channel pair owns one KERNEL_SIZE-tap kernel uploaded once to the GPU.
For each callback, the CPU builds one input window per input channel:
  [previous KERNEL_SIZE - 1 input samples][current FRAMES_PER_BUFFER input samples]
Then CUDA launches:
  <<<OUTPUT_CHANNELS, FRAMES_PER_BUFFER>>>
so one block owns one output channel and one thread owns one output sample.
Each thread sums contributions from every input channel.
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

#define INPUT_CHANNELS 5
#define OUTPUT_CHANNELS 5

#define DEVICE_NAME "default"

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 48000.0
#endif

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 64ul
#endif

#ifndef RUN_SECONDS
#define RUN_SECONDS 1.0
#endif

#ifndef KERNEL_SIZE
#define KERNEL_SIZE (1 << 15)
#endif

#ifndef INPUT0_FREQUENCY_HZ
#define INPUT0_FREQUENCY_HZ 440.0
#endif

#ifndef INPUT1_FREQUENCY_HZ
#define INPUT1_FREQUENCY_HZ 660.0
#endif

#ifndef INPUT0_AMPLITUDE
#define INPUT0_AMPLITUDE 0.8
#endif

#ifndef INPUT1_AMPLITUDE
#define INPUT1_AMPLITUDE 0.6
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#ifndef PORTABLE_PLOT_SCRIPT
#define PORTABLE_PLOT_SCRIPT "Portable/scripts/plot.py"
#endif

#ifndef PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT
#define PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT "Portable/scripts/show_callback_timing.py"
#endif

#include "portable/audio_helpers.h"
#include "portable/csv_utils.h"
#if MOCK
#include "portable/mock_device_registry.h"
#endif

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr int kInputChannelCount = INPUT_CHANNELS;
constexpr int kOutputChannelCount = OUTPUT_CHANNELS;
constexpr int kHistorySize = KERNEL_SIZE - 1;
constexpr int kWindowSize = KERNEL_SIZE - 1 + static_cast<int>(FRAMES_PER_BUFFER);

using Kernel = std::vector<float>;
using KernelBank = std::vector<Kernel>;
using KernelMatrix = std::vector<KernelBank>;

struct SimpleCudaNaiveConvolutionData
{
    // These device buffers persist for the whole stream.
    // Kernels are uploaded once.
    // The input window and output block are reused every callback.
    float *deviceKernels = nullptr;
    float *deviceInputWindows = nullptr;
    float *deviceOutputBlock = nullptr;

    // These host buffers are the CPU-side staging areas.
    // hostInputHistory keeps the last KERNEL_SIZE - 1 samples per input channel.
    // hostInputWindows is the concatenated history + current callback block.
    std::vector<float> hostKernelsFlat;
    std::vector<float> hostInputHistory;
    std::vector<float> hostInputWindows;
    std::vector<float> hostOutputBlock;

    int frameIndex = 0;
    int maxFrames = 0;
    double sampleRate = SAMPLE_RATE;
    bool hasFirstCallbackTime = false;
    double firstCallbackTimeSeconds = 0.0;

    bool hadCudaError = false;
    cudaError_t lastCudaError = cudaSuccess;
    const char *lastCudaStage = nullptr;

    std::vector<std::vector<float>> savedInputs;
    std::vector<std::vector<float>> savedOutputs;
    std::vector<float> savedCallbackIndex;
    std::vector<float> savedCallbackTimeSeconds;
    std::vector<float> savedCallbackDelaySeconds;
    std::vector<float> savedCallbackAllowedSeconds;
    std::vector<float> savedCallbackLoadPercent;
};

void remember_cuda_error(
    SimpleCudaNaiveConvolutionData *data,
    const char *stage,
    cudaError_t error)
{
    if (!data || data->hadCudaError)
    {
        return;
    }

    data->hadCudaError = true;
    data->lastCudaError = error;
    data->lastCudaStage = stage;
}

float sine_sample(
    double time_seconds,
    double frequency_hz,
    double amplitude,
    double phase_radians = 0.0)
{
    return static_cast<float>(
        amplitude * std::sin(2.0 * kPi * frequency_hz * time_seconds + phase_radians));
}

double frequency_for_input_channel(
    int input_channel)
{
    if (input_channel == 0)
    {
        return INPUT0_FREQUENCY_HZ;
    }
    if (input_channel == 1)
    {
        return INPUT1_FREQUENCY_HZ;
    }
    return 880.0 + 110.0 * static_cast<double>(input_channel - 2);
}

double amplitude_for_input_channel(
    int input_channel)
{
    if (input_channel == 0)
    {
        return INPUT0_AMPLITUDE;
    }
    if (input_channel == 1)
    {
        return INPUT1_AMPLITUDE;
    }
    return std::max(0.2, 0.5 - 0.05 * static_cast<double>(input_channel - 2));
}

double phase_for_input_channel(
    int input_channel)
{
    if (input_channel == 0)
    {
        return 0.0;
    }
    if (input_channel == 1)
    {
        return 0.3;
    }
    return 0.15 * static_cast<double>(input_channel);
}

size_t input_history_offset(
    int input_channel)
{
    return static_cast<size_t>(input_channel * kHistorySize);
}

size_t input_window_offset(
    int input_channel)
{
    return static_cast<size_t>(input_channel * kWindowSize);
}

size_t output_block_offset(
    int output_channel)
{
    return static_cast<size_t>(output_channel * FRAMES_PER_BUFFER);
}

size_t flattened_kernel_index(
    int output_channel,
    int input_channel,
    size_t tap)
{
    return static_cast<size_t>(
        (output_channel * kInputChannelCount + input_channel) * KERNEL_SIZE) + tap;
}

KernelMatrix make_naive_kernel_matrix()
{
    KernelMatrix kernel_matrix(
        static_cast<size_t>(kOutputChannelCount),
        KernelBank(
            static_cast<size_t>(kInputChannelCount),
            Kernel(KERNEL_SIZE, 0.0f)));

    for (int output_channel = 0; output_channel < kOutputChannelCount; ++output_channel)
    {
        double output_sum_abs = 0.0;
        for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
        {
            Kernel &kernel =
                kernel_matrix[static_cast<size_t>(output_channel)]
                           [static_cast<size_t>(input_channel)];
            const bool is_direct_path = output_channel == input_channel;
            const double mix_gain =
                is_direct_path
                    ? 1.0
                    : 0.20 + 0.04 * static_cast<double>((output_channel + input_channel) % 3);
            const double decay =
                is_direct_path
                    ? 6.0 + 1.3 * static_cast<double>(output_channel)
                    : 8.5 + 0.8 * static_cast<double>(output_channel) +
                          0.5 * static_cast<double>(input_channel);
            const double phase =
                0.35 * static_cast<double>(output_channel) +
                0.55 * static_cast<double>(input_channel);

            for (int tap = 0; tap < KERNEL_SIZE; ++tap)
            {
                const double position =
                    static_cast<double>(tap) / static_cast<double>(KERNEL_SIZE - 1);

                double coefficient = 0.0;
                if (is_direct_path)
                {
                    coefficient =
                        mix_gain *
                        std::exp(-decay * position) *
                        (0.70 + 0.30 * std::cos(
                                           (0.019 + 0.002 * static_cast<double>(input_channel)) *
                                               static_cast<double>(tap) +
                                           phase));
                }
                else
                {
                    coefficient =
                        mix_gain *
                        std::exp(-decay * position) *
                        (0.45 +
                         0.25 * std::sin(
                                    (0.011 + 0.001 * static_cast<double>(output_channel)) *
                                        static_cast<double>(tap) +
                                    phase) +
                         0.30 * std::cos(
                                    (0.023 + 0.0015 * static_cast<double>(input_channel)) *
                                        static_cast<double>(tap) +
                                    0.5 * phase));
                }

                kernel[static_cast<size_t>(tap)] = static_cast<float>(coefficient);
                output_sum_abs += std::fabs(coefficient);
            }
        }

        const double target_sum_abs =
            std::max(0.35, 0.95 - 0.12 * static_cast<double>(output_channel));
        const float scale =
            output_sum_abs > 0.0
                ? static_cast<float>(target_sum_abs / output_sum_abs)
                : 0.0f;

        for (Kernel &kernel : kernel_matrix[static_cast<size_t>(output_channel)])
        {
            for (float &coefficient : kernel)
            {
                coefficient *= scale;
            }
        }
    }

    return kernel_matrix;
}

std::vector<float> flatten_kernel_matrix(
    const KernelMatrix &kernel_matrix)
{
    std::vector<float> flat(
        static_cast<size_t>(kOutputChannelCount * kInputChannelCount * KERNEL_SIZE),
        0.0f);

    for (int output_channel = 0; output_channel < kOutputChannelCount; ++output_channel)
    {
        if (static_cast<size_t>(output_channel) >= kernel_matrix.size())
        {
            break;
        }

        const KernelBank &kernel_row = kernel_matrix[static_cast<size_t>(output_channel)];
        for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
        {
            if (static_cast<size_t>(input_channel) >= kernel_row.size())
            {
                break;
            }

            const Kernel &kernel = kernel_row[static_cast<size_t>(input_channel)];
            const size_t copy_count =
                std::min(kernel.size(), static_cast<size_t>(KERNEL_SIZE));
            for (size_t tap = 0; tap < copy_count; ++tap)
            {
                flat[flattened_kernel_index(output_channel, input_channel, tap)] =
                    kernel[tap];
            }
        }
    }

    return flat;
}

#if MOCK
void register_simple_mock_device(
    const KernelMatrix &kernel_matrix)
{
    clear_mock_devices();

    std::vector<MockInputGenerator> input_generators;
    input_generators.reserve(static_cast<size_t>(kInputChannelCount));

    for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
    {
        const double frequency_hz = frequency_for_input_channel(input_channel);
        const double amplitude = amplitude_for_input_channel(input_channel);
        const double phase = phase_for_input_channel(input_channel);
        input_generators.push_back(
            [frequency_hz, amplitude, phase](double time_seconds) -> float
        {
            return sine_sample(time_seconds, frequency_hz, amplitude, phase);
        });
    }

    register_device_that_expects_exact_convolution(
        DEVICE_NAME,
        SAMPLE_RATE,
        {SAMPLE_RATE},
        input_generators,
        kernel_matrix);
}
#endif

std::string build_python_command(
    const std::string &script_path,
    const std::string &csv_path)
{
    std::string command = "py -3 ";
    command += '"' + script_path + "\" \"" + csv_path + '"';
    return command;
}

bool save_capture_csv(
    const SimpleCudaNaiveConvolutionData &data,
    const std::string &csv_path)
{
    const size_t sample_count =
        !data.savedInputs.empty()
            ? data.savedInputs.front().size()
            : (!data.savedOutputs.empty() ? data.savedOutputs.front().size() : 0u);
    if (sample_count == 0)
    {
        return false;
    }

    const auto all_channels_have_expected_size =
        [sample_count](const std::vector<std::vector<float>> &channels) -> bool
    {
        for (const std::vector<float> &channel : channels)
        {
            if (channel.size() != sample_count)
            {
                return false;
            }
        }
        return true;
    };

    if (!all_channels_have_expected_size(data.savedInputs) ||
        !all_channels_have_expected_size(data.savedOutputs))
    {
        return false;
    }

    std::vector<float> time_axis(sample_count, 0.0f);
    for (size_t sample = 0; sample < sample_count; ++sample)
    {
        time_axis[sample] =
            static_cast<float>(static_cast<double>(sample) / data.sampleRate);
    }

    std::vector<std::string> headers = {"time (s)"};
    std::vector<const std::vector<float> *> columns = {&time_axis};

    for (size_t input_channel = 0; input_channel < data.savedInputs.size(); ++input_channel)
    {
        headers.push_back("input " + std::to_string(input_channel));
        columns.push_back(&data.savedInputs[input_channel]);
    }

    for (size_t output_channel = 0; output_channel < data.savedOutputs.size(); ++output_channel)
    {
        headers.push_back("output " + std::to_string(output_channel));
        columns.push_back(&data.savedOutputs[output_channel]);
    }

    return save_arrays_to_csv(csv_path, headers, columns);
}

void record_callback_timing(
    SimpleCudaNaiveConvolutionData *data,
    const PaStreamCallbackTimeInfo *timeInfo,
    unsigned long framesPerBuffer,
    double callback_delay_seconds)
{
    if (!data)
    {
        return;
    }

    const float callback_index =
        static_cast<float>(data->savedCallbackIndex.size());
    const double raw_callback_time_seconds =
        timeInfo
            ? static_cast<double>(timeInfo->currentTime)
            : static_cast<double>(
                  static_cast<double>(data->frameIndex) /
                  std::max(data->sampleRate, 1.0));
    if (!data->hasFirstCallbackTime)
    {
        data->hasFirstCallbackTime = true;
        data->firstCallbackTimeSeconds = raw_callback_time_seconds;
    }
    const float callback_time_seconds =
        static_cast<float>(raw_callback_time_seconds - data->firstCallbackTimeSeconds);
    const float allowed_delay_seconds =
        static_cast<float>(
            static_cast<double>(framesPerBuffer) /
            std::max(data->sampleRate, 1.0));
    const float callback_load_percent =
        allowed_delay_seconds > 0.0f
            ? static_cast<float>(
                  100.0 * callback_delay_seconds /
                  static_cast<double>(allowed_delay_seconds))
            : 0.0f;

    data->savedCallbackIndex.push_back(callback_index);
    data->savedCallbackTimeSeconds.push_back(callback_time_seconds);
    data->savedCallbackDelaySeconds.push_back(
        static_cast<float>(callback_delay_seconds));
    data->savedCallbackAllowedSeconds.push_back(allowed_delay_seconds);
    data->savedCallbackLoadPercent.push_back(callback_load_percent);
}

bool save_callback_timing_csv(
    const SimpleCudaNaiveConvolutionData &data,
    const std::string &csv_path)
{
    const size_t sample_count = data.savedCallbackIndex.size();
    if (data.savedCallbackTimeSeconds.size() != sample_count ||
        data.savedCallbackDelaySeconds.size() != sample_count ||
        data.savedCallbackAllowedSeconds.size() != sample_count ||
        data.savedCallbackLoadPercent.size() != sample_count)
    {
        return false;
    }

    return save_arrays_to_csv(
        csv_path,
        {
            "callback index",
            "callback time (s)",
            "callback delay (s)",
            "allowed delay (s)",
            "callback load (%)"},
        {
            &data.savedCallbackIndex,
            &data.savedCallbackTimeSeconds,
            &data.savedCallbackDelaySeconds,
            &data.savedCallbackAllowedSeconds,
            &data.savedCallbackLoadPercent});
}

// Grid layout:
//   blockIdx.x  -> output channel
//   threadIdx.x -> output sample index inside the current callback block
//
// Each thread performs the full direct-form FIR sum for one output sample,
// accumulating every input-channel contribution for its output channel.
__global__ void naive_convolution_kernel(
    const float *input_windows,
    const float *kernels,
    float *output_block)
{
    const int output_channel = static_cast<int>(blockIdx.x);
    const int output_sample = static_cast<int>(threadIdx.x);

    if (output_channel >= kOutputChannelCount ||
        output_sample >= static_cast<int>(FRAMES_PER_BUFFER))
    {
        return;
    }

    double accumulated = 0.0;
    const int newest_input_index = kHistorySize + output_sample;
    for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
    {
        const float *channel_window =
            input_windows + static_cast<size_t>(input_channel * kWindowSize);
        const float *channel_kernel =
            kernels +
            static_cast<size_t>(
                (output_channel * kInputChannelCount + input_channel) * KERNEL_SIZE);

        for (int tap = 0; tap < KERNEL_SIZE; ++tap)
        {
            accumulated +=
                static_cast<double>(channel_window[newest_input_index - tap]) *
                static_cast<double>(channel_kernel[tap]);
        }
    }

    output_block[static_cast<size_t>(output_channel * FRAMES_PER_BUFFER + output_sample)] =
        static_cast<float>(accumulated);
}

int simple_cuda_naive_convolution_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *timeInfo,
    PaStreamCallbackFlags,
    void *userData)
{
    SimpleCudaNaiveConvolutionData *data =
        static_cast<SimpleCudaNaiveConvolutionData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data || !output ||
        !data->deviceKernels ||
        !data->deviceInputWindows ||
        !data->deviceOutputBlock)
    {
        return paAbort;
    }

    const auto callback_start = std::chrono::steady_clock::now();
    const auto finish_callback =
        [data, timeInfo, framesPerBuffer, callback_start](int result) -> int
    {
        const auto callback_end = std::chrono::steady_clock::now();
        const double callback_delay_seconds =
            std::chrono::duration<double>(callback_end - callback_start).count();
        record_callback_timing(
            data,
            timeInfo,
            framesPerBuffer,
            callback_delay_seconds);
        return result;
    };

    if (framesPerBuffer != FRAMES_PER_BUFFER)
    {
        std::cerr << "Unexpected callback size: got " << framesPerBuffer
                  << " but this demo expects " << FRAMES_PER_BUFFER << '\n';
        return finish_callback(paAbort);
    }

    const int frames_left = data->maxFrames - data->frameIndex;
    const int frames_to_process =
        std::max(0, std::min(static_cast<int>(framesPerBuffer), frames_left));

    // Build one contiguous window per input channel:
    // [old history][current callback input]
    for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
    {
        float *window =
            data->hostInputWindows.data() + input_window_offset(input_channel);
        const float *history =
            data->hostInputHistory.data() + input_history_offset(input_channel);

        for (int history_index = 0; history_index < kHistorySize; ++history_index)
        {
            window[history_index] = history[history_index];
        }

        for (unsigned long frame = 0; frame < framesPerBuffer; ++frame)
        {
            float sample = 0.0f;
            if (static_cast<int>(frame) < frames_to_process)
            {
                sample =
                    input
                        ? input[static_cast<size_t>(frame * kInputChannelCount + input_channel)]
                        : 0.0f;
            }
            window[kHistorySize + static_cast<int>(frame)] = sample;
        }
    }

    cudaError_t error = cudaMemcpy(
        data->deviceInputWindows,
        data->hostInputWindows.data(),
        data->hostInputWindows.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemcpy input windows host->device", error);
        return finish_callback(paAbort);
    }

    naive_convolution_kernel<<<kOutputChannelCount, FRAMES_PER_BUFFER>>>(
        data->deviceInputWindows,
        data->deviceKernels,
        data->deviceOutputBlock);

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        remember_cuda_error(data, "kernel launch", error);
        return finish_callback(paAbort);
    }

    error = cudaMemcpy(
        data->hostOutputBlock.data(),
        data->deviceOutputBlock,
        data->hostOutputBlock.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemcpy output block device->host", error);
        return finish_callback(paAbort);
    }

    for (int frame = 0; frame < frames_to_process; ++frame)
    {
        const size_t captured_frame_index = static_cast<size_t>(data->frameIndex);
        for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
        {
            const size_t sample_index =
                input_window_offset(input_channel) +
                static_cast<size_t>(kHistorySize + frame);
            data->savedInputs[static_cast<size_t>(input_channel)][captured_frame_index] =
                data->hostInputWindows[sample_index];
        }

        for (int output_channel = 0; output_channel < kOutputChannelCount; ++output_channel)
        {
            const float output_sample_value =
                data->hostOutputBlock[
                    output_block_offset(output_channel) + static_cast<size_t>(frame)];
            output[static_cast<size_t>(frame * kOutputChannelCount + output_channel)] =
                output_sample_value;
            data->savedOutputs[static_cast<size_t>(output_channel)][captured_frame_index] =
                output_sample_value;
        }

        data->frameIndex++;
    }

    for (unsigned long frame = static_cast<unsigned long>(frames_to_process);
         frame < framesPerBuffer;
         ++frame)
    {
        for (int output_channel = 0; output_channel < kOutputChannelCount; ++output_channel)
        {
            output[static_cast<size_t>(frame * kOutputChannelCount + output_channel)] = 0.0f;
        }
    }

    // Keep only the last KERNEL_SIZE - 1 real input samples for the next callback.
    for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
    {
        const float *window =
            data->hostInputWindows.data() + input_window_offset(input_channel);
        float *history =
            data->hostInputHistory.data() + input_history_offset(input_channel);
        const int new_history_offset = frames_to_process;
        for (int history_index = 0; history_index < kHistorySize; ++history_index)
        {
            history[history_index] = window[new_history_offset + history_index];
        }
    }

    if (data->frameIndex >= data->maxFrames)
    {
        return finish_callback(paComplete);
    }

    return finish_callback(paContinue);
}

} // namespace

int main()
{
    const KernelMatrix kernel_matrix = make_naive_kernel_matrix();
#if MOCK
    register_simple_mock_device(kernel_matrix);
#endif

    int cuda_device_count = 0;
    cudaError_t cuda_error = cudaGetDeviceCount(&cuda_device_count);
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        return 1;
    }
    if (cuda_device_count <= 0)
    {
        std::cerr << "No CUDA device found.\n";
        return 1;
    }

    cuda_error = cudaSetDevice(0);
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaSetDevice(0) failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        return 1;
    }

    cudaDeviceProp device_properties{};
    cuda_error = cudaGetDeviceProperties(&device_properties, 0);
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        return 1;
    }

    if (FRAMES_PER_BUFFER > static_cast<unsigned long>(device_properties.maxThreadsPerBlock))
    {
        std::cerr << "FRAMES_PER_BUFFER=" << FRAMES_PER_BUFFER
                  << " exceeds maxThreadsPerBlock="
                  << device_properties.maxThreadsPerBlock << '\n';
        return 1;
    }

    std::cout << "CUDA device 0: " << device_properties.name << '\n';

    const PaError init_error = Pa_Initialize();
    if (init_error != paNoError)
    {
        std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(init_error) << '\n';
        return 1;
    }

    const int device_index = select_device_by_name(DEVICE_NAME, MOCK ? true : false);
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

    SimpleCudaNaiveConvolutionData data{};
    data.maxFrames = static_cast<int>(std::llround(RUN_SECONDS * SAMPLE_RATE));
    data.sampleRate = SAMPLE_RATE;
    data.hostKernelsFlat = flatten_kernel_matrix(kernel_matrix);
    data.hostInputHistory.assign(
        static_cast<size_t>(kInputChannelCount * kHistorySize),
        0.0f);
    data.hostInputWindows.assign(
        static_cast<size_t>(kInputChannelCount * kWindowSize),
        0.0f);
    data.hostOutputBlock.assign(
        static_cast<size_t>(kOutputChannelCount * FRAMES_PER_BUFFER),
        0.0f);
    data.savedInputs.assign(
        static_cast<size_t>(kInputChannelCount),
        std::vector<float>(static_cast<size_t>(data.maxFrames), 0.0f));
    data.savedOutputs.assign(
        static_cast<size_t>(kOutputChannelCount),
        std::vector<float>(static_cast<size_t>(data.maxFrames), 0.0f));
    const size_t estimated_callback_count =
        (static_cast<size_t>(data.maxFrames) +
         static_cast<size_t>(FRAMES_PER_BUFFER) - 1u) /
        static_cast<size_t>(FRAMES_PER_BUFFER);
    data.savedCallbackIndex.reserve(estimated_callback_count);
    data.savedCallbackTimeSeconds.reserve(estimated_callback_count);
    data.savedCallbackDelaySeconds.reserve(estimated_callback_count);
    data.savedCallbackAllowedSeconds.reserve(estimated_callback_count);
    data.savedCallbackLoadPercent.reserve(estimated_callback_count);

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data.deviceKernels),
        data.hostKernelsFlat.size() * sizeof(float));
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaMalloc(deviceKernels) failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        Pa_Terminate();
        return 1;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data.deviceInputWindows),
        data.hostInputWindows.size() * sizeof(float));
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaMalloc(deviceInputWindows) failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        cudaFree(data.deviceKernels);
        Pa_Terminate();
        return 1;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data.deviceOutputBlock),
        data.hostOutputBlock.size() * sizeof(float));
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaMalloc(deviceOutputBlock) failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        cudaFree(data.deviceInputWindows);
        cudaFree(data.deviceKernels);
        Pa_Terminate();
        return 1;
    }

    cuda_error = cudaMemcpy(
        data.deviceKernels,
        data.hostKernelsFlat.data(),
        data.hostKernelsFlat.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaMemcpy(deviceKernels) failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        cudaFree(data.deviceOutputBlock);
        cudaFree(data.deviceInputWindows);
        cudaFree(data.deviceKernels);
        Pa_Terminate();
        return 1;
    }

    PaStreamParameters input_parameters{};
    input_parameters.device = device_index;
    input_parameters.channelCount = kInputChannelCount;
    input_parameters.sampleFormat = SAMPLE_FORMAT;
    input_parameters.suggestedLatency = device_info->defaultLowInputLatency;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = kOutputChannelCount;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = device_info->defaultLowOutputLatency;

    if (!check_if_device_respects_input_and_output_stream_specs(
            &input_parameters,
            &output_parameters,
            SAMPLE_RATE))
    {
        cudaFree(data.deviceOutputBlock);
        cudaFree(data.deviceInputWindows);
        cudaFree(data.deviceKernels);
        Pa_Terminate();
        return 1;
    }

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        simple_cuda_naive_convolution_callback,
        &data);
    if (open_error != paNoError)
    {
        std::cerr << "Pa_OpenStream failed: " << Pa_GetErrorText(open_error) << '\n';
        cudaFree(data.deviceOutputBlock);
        cudaFree(data.deviceInputWindows);
        cudaFree(data.deviceKernels);
        Pa_Terminate();
        return 1;
    }

    std::cout << "Running device " << DEVICE_NAME
              << " for " << RUN_SECONDS
              << " seconds at " << SAMPLE_RATE
              << " Hz.\n";
    std::cout << "Naive CUDA launch: <<<" << kOutputChannelCount
              << ", " << FRAMES_PER_BUFFER
              << ">>> with one block per output channel and one thread per output sample.\n";
    std::cout << "Mixing " << kInputChannelCount
              << " input channels into " << kOutputChannelCount
              << " output channels.\n";
    std::cout << "Kernel size: " << KERNEL_SIZE << " taps per output/input pair.\n";

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        std::cerr << "Pa_StartStream failed: " << Pa_GetErrorText(start_error) << '\n';
        Pa_CloseStream(stream);
        cudaFree(data.deviceOutputBlock);
        cudaFree(data.deviceInputWindows);
        cudaFree(data.deviceKernels);
        Pa_Terminate();
        return 1;
    }

    while (Pa_IsStreamActive(stream) == 1)
    {
        Pa_Sleep(20);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);

    if (data.hadCudaError)
    {
        std::cerr << "CUDA failed during callback at "
                  << (data.lastCudaStage ? data.lastCudaStage : "(unknown stage)")
                  << ": " << cudaGetErrorString(data.lastCudaError) << '\n';
        cudaFree(data.deviceOutputBlock);
        cudaFree(data.deviceInputWindows);
        cudaFree(data.deviceKernels);
        Pa_Terminate();
        return 1;
    }

    const std::string capture_csv_path =
        std::string(PORTABLE_OUTPUT_DIR) + "/simple_cuda_naive_convolution_capture.csv";
    if (save_capture_csv(data, capture_csv_path))
    {
        std::cout << "Saved CSV: " << capture_csv_path << '\n';
        std::cout << "Plot command: "
                  << build_python_command(PORTABLE_PLOT_SCRIPT, capture_csv_path)
                  << '\n';
    }
    else
    {
        std::cerr << "Failed to save CSV: " << capture_csv_path << '\n';
    }

    const std::string callback_timing_csv_path =
        std::string(PORTABLE_OUTPUT_DIR) + "/simple_cuda_naive_convolution_callback_timing.csv";
    if (save_callback_timing_csv(data, callback_timing_csv_path))
    {
        std::cout << "Saved callback timing CSV: " << callback_timing_csv_path << '\n';
        std::cout << "Callback timing plot command: "
                  << build_python_command(
                         PORTABLE_CALLBACK_TIMING_VIEW_SCRIPT,
                         callback_timing_csv_path)
                  << '\n';
    }
    else
    {
        std::cerr << "Failed to save callback timing CSV: "
                  << callback_timing_csv_path << '\n';
    }

    std::cout << "Captured frames: " << data.frameIndex << '\n';
    std::cout << "Captured callbacks: " << data.savedCallbackDelaySeconds.size() << '\n';

    cudaFree(data.deviceOutputBlock);
    cudaFree(data.deviceInputWindows);
    cudaFree(data.deviceKernels);
    Pa_Terminate();
    return 0;
}
