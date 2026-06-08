/*
Build from repo root:
  direct machine:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=simple_cuda_less_naive_convolution -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=OFF -DPORTABLE_DEVICE_NAME="MADIface USB (24285073): Audio (hw:2,0)" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32
    cmake --build Portable/build --target portable_simple_cuda_less_naive_convolution --parallel
    ./Portable/build/portable_simple_cuda_less_naive_convolution

  JACK:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=simple_cuda_less_naive_convolution -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=ON -DPORTABLE_DEVICE_NAME="system" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32
    cmake --build Portable/build --target portable_simple_cuda_less_naive_convolution --parallel
    ./Portable/build/portable_simple_cuda_less_naive_convolution

To try mock audio instead, rerun either `cmake -S ...` command with `-DPORTABLE_USE_MOCK=ON`.

Build-time config:
  `INPUT_CHANNELS` and `OUTPUT_CHANNELS` are fixed in this file.
  `DEVICE_NAME`, `SAMPLE_RATE`, and `FRAMES_PER_BUFFER` come from the CMake command above.

Prepare JACK for direct hardware access:
  systemctl --user mask --runtime pipewire.service pipewire.socket pipewire-pulse.service pipewire-pulse.socket wireplumber.service
  systemctl --user stop pipewire.service pipewire.socket pipewire-pulse.service pipewire-pulse.socket wireplumber.service
  killall pipewire pipewire-pulse wireplumber

Start JACK:
  jackd -d alsa -d hw:2,0 -r 44100 -p 32 -n 3
  jack_lsp

Restore desktop audio afterwards:
  killall jackd
  systemctl --user unmask --runtime pipewire.service pipewire.socket pipewire-pulse.service pipewire-pulse.socket wireplumber.service
  systemctl --user start wireplumber.service pipewire.service pipewire-pulse.service

This is a less naive CUDA FIR example.
Each output/input channel pair owns one fixed KERNEL_SIZE-tap kernel.
The kernels are partitioned into FRAMES_PER_BUFFER-sample blocks and transformed
once before the stream starts.
For each callback, the app:
  1. copies one FRAMES_PER_BUFFER block per input channel to the GPU,
  2. performs batched R2C FFTs for the current input block,
  3. mixes all input-channel spectra through the precomputed kernel matrix,
  4. performs batched C2R inverse FFTs for the output channels,
  5. applies overlap-add to recover the current output block.
*/

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "portable/build_config.h"

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef MOCK
#error "MOCK must be defined by the build system for this target."
#endif

#define INPUT_CHANNELS 32
#define OUTPUT_CHANNELS 32

#ifndef DEVICE_NAME
#define DEVICE_NAME "default"
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 48000.0
#endif

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 256l
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
constexpr int kBlockSize = static_cast<int>(FRAMES_PER_BUFFER);
constexpr int kFftSize = 2 * kBlockSize;
constexpr int kFftBins = kFftSize / 2 + 1;
constexpr int kPartitionCount =
    (KERNEL_SIZE + kBlockSize - 1) / kBlockSize;

using Kernel = std::vector<float>;
using KernelBank = std::vector<Kernel>;
using KernelMatrix = std::vector<KernelBank>;

struct SimpleCudaLessNaiveConvolutionData
{
    cufftHandle inputForwardPlan = 0;
    cufftHandle kernelForwardPlan = 0;
    cufftHandle outputInversePlan = 0;
    cudaStream_t processingStream = nullptr;

    void *inputForwardWorkArea = nullptr;
    void *kernelForwardWorkArea = nullptr;
    void *outputInverseWorkArea = nullptr;

    cufftReal *deviceInputTimeBlock = nullptr;
    cufftComplex *deviceInputFreqBlock = nullptr;
    cufftComplex *deviceInputFreqHistory = nullptr;
    cufftReal *deviceKernelTimePartitions = nullptr;
    cufftComplex *deviceKernelFreqPartitions = nullptr;
    cufftComplex *deviceOutputFreqBlock = nullptr;
    cufftReal *deviceOutputTimeBlock = nullptr;
    float *deviceOverlapTail = nullptr;
    float *deviceOutputBlock = nullptr;

    std::vector<cufftReal> hostInputTimeBlock;
    std::vector<float> hostOutputBlock;

    int inputHistoryWritePartition = 0;
    int frameIndex = 0;
    int maxFrames = 0;
    double sampleRate = SAMPLE_RATE;
    bool hasFirstCallbackTime = false;
    double firstCallbackTimeSeconds = 0.0;

    bool hadCudaError = false;
    cudaError_t lastCudaError = cudaSuccess;
    const char *lastCudaStage = nullptr;

    bool hadCufftError = false;
    cufftResult lastCufftError = CUFFT_SUCCESS;
    const char *lastCufftStage = nullptr;

    std::vector<std::vector<float>> savedInputs;
    std::vector<std::vector<float>> savedOutputs;
    std::vector<float> savedCallbackIndex;
    std::vector<float> savedCallbackTimeSeconds;
    std::vector<float> savedCallbackDelaySeconds;
    std::vector<float> savedCallbackAllowedSeconds;
    std::vector<float> savedCallbackLoadPercent;
};

void remember_cuda_error(
    SimpleCudaLessNaiveConvolutionData *data,
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

void remember_cufft_error(
    SimpleCudaLessNaiveConvolutionData *data,
    const char *stage,
    cufftResult error)
{
    if (!data || data->hadCufftError)
    {
        return;
    }

    data->hadCufftError = true;
    data->lastCufftError = error;
    data->lastCufftStage = stage;
}

const char *cufft_result_string(
    cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
    case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
    default: return "CUFFT_UNKNOWN";
    }
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

__host__ __device__ inline size_t input_time_block_offset(
    int input_channel)
{
    return static_cast<size_t>(input_channel * kFftSize);
}

__host__ __device__ inline size_t input_freq_block_offset(
    int input_channel)
{
    return static_cast<size_t>(input_channel * kFftBins);
}

__host__ __device__ inline size_t input_history_freq_offset(
    int partition,
    int input_channel)
{
    return static_cast<size_t>((partition * kInputChannelCount + input_channel) * kFftBins);
}

__host__ __device__ inline size_t kernel_time_partition_offset(
    int output_channel,
    int input_channel,
    int partition)
{
    return static_cast<size_t>(
        ((output_channel * kInputChannelCount + input_channel) * kPartitionCount + partition) *
        kFftSize);
}

__host__ __device__ inline size_t kernel_freq_partition_offset(
    int output_channel,
    int input_channel,
    int partition)
{
    return static_cast<size_t>(
        ((output_channel * kInputChannelCount + input_channel) * kPartitionCount + partition) *
        kFftBins);
}

__host__ __device__ inline size_t output_freq_block_offset(
    int output_channel)
{
    return static_cast<size_t>(output_channel * kFftBins);
}

__host__ __device__ inline size_t output_time_block_offset(
    int output_channel)
{
    return static_cast<size_t>(output_channel * kFftSize);
}

__host__ __device__ inline size_t output_block_offset(
    int output_channel)
{
    return static_cast<size_t>(output_channel * kBlockSize);
}

KernelMatrix make_less_naive_kernel_matrix()
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

std::vector<cufftReal> make_kernel_time_partitions(
    const KernelMatrix &kernel_matrix)
{
    std::vector<cufftReal> partitions(
        static_cast<size_t>(kOutputChannelCount * kInputChannelCount * kPartitionCount * kFftSize),
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
            for (int partition = 0; partition < kPartitionCount; ++partition)
            {
                const size_t partition_offset =
                    kernel_time_partition_offset(output_channel, input_channel, partition);
                const size_t kernel_start =
                    static_cast<size_t>(partition * kBlockSize);
                if (kernel_start >= kernel.size())
                {
                    continue;
                }

                const size_t copy_count =
                    std::min(
                        static_cast<size_t>(kBlockSize),
                        kernel.size() - kernel_start);
                for (size_t sample = 0; sample < copy_count; ++sample)
                {
                    partitions[partition_offset + sample] =
                        kernel[kernel_start + sample];
                }
            }
        }
    }

    return partitions;
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
    const SimpleCudaLessNaiveConvolutionData &data,
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
    SimpleCudaLessNaiveConvolutionData *data,
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
    const SimpleCudaLessNaiveConvolutionData &data,
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

__global__ void store_input_frequency_history_kernel(
    const cufftComplex *input_frequency,
    cufftComplex *input_frequency_history,
    int write_partition)
{
    const int flat_index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_bins = kInputChannelCount * kFftBins;
    if (flat_index >= total_bins)
    {
        return;
    }

    const int input_channel = flat_index / kFftBins;
    const int bin = flat_index % kFftBins;
    input_frequency_history[
        input_history_freq_offset(write_partition, input_channel) +
        static_cast<size_t>(bin)] =
        input_frequency[
            input_freq_block_offset(input_channel) +
            static_cast<size_t>(bin)];
}

__global__ void accumulate_output_frequency_kernel(
    const cufftComplex *input_frequency_history,
    const cufftComplex *kernel_frequency_partitions,
    cufftComplex *output_frequency,
    int newest_partition)
{
    const int flat_index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_bins = kOutputChannelCount * kFftBins;
    if (flat_index >= total_bins)
    {
        return;
    }

    const int output_channel = flat_index / kFftBins;
    const int bin = flat_index % kFftBins;

    float accumulated_real = 0.0f;
    float accumulated_imag = 0.0f;

    for (int partition = 0; partition < kPartitionCount; ++partition)
    {
        int history_partition = newest_partition - partition;
        if (history_partition < 0)
        {
            history_partition += kPartitionCount;
        }

        for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
        {
            const cufftComplex input_value =
                input_frequency_history[
                    input_history_freq_offset(history_partition, input_channel) +
                    static_cast<size_t>(bin)];
            const cufftComplex kernel_value =
                kernel_frequency_partitions[
                    kernel_freq_partition_offset(output_channel, input_channel, partition) +
                    static_cast<size_t>(bin)];

            accumulated_real +=
                kernel_value.x * input_value.x - kernel_value.y * input_value.y;
            accumulated_imag +=
                kernel_value.x * input_value.y + kernel_value.y * input_value.x;
        }
    }

    cufftComplex accumulated_value;
    accumulated_value.x = accumulated_real;
    accumulated_value.y = accumulated_imag;
    output_frequency[
        output_freq_block_offset(output_channel) +
        static_cast<size_t>(bin)] =
        accumulated_value;
}

__global__ void overlap_add_and_scale_kernel(
    const cufftReal *output_time,
    float *overlap_tail,
    float *output_block)
{
    const int flat_index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_samples = kOutputChannelCount * kBlockSize;
    if (flat_index >= total_samples)
    {
        return;
    }

    const int output_channel = flat_index / kBlockSize;
    const int sample = flat_index % kBlockSize;
    const float scale = 1.0f / static_cast<float>(kFftSize);

    const size_t time_offset =
        output_time_block_offset(output_channel) +
        static_cast<size_t>(sample);
    const size_t tail_offset =
        output_block_offset(output_channel) +
        static_cast<size_t>(sample);

    const float current_sample =
        static_cast<float>(output_time[time_offset]) * scale +
        overlap_tail[tail_offset];
    output_block[tail_offset] = current_sample;
    overlap_tail[tail_offset] =
        static_cast<float>(output_time[time_offset + static_cast<size_t>(kBlockSize)]) * scale;
}

void release_processing_resources(
    SimpleCudaLessNaiveConvolutionData &data)
{
    if (data.inputForwardPlan != 0)
    {
        cufftDestroy(data.inputForwardPlan);
        data.inputForwardPlan = 0;
    }
    if (data.kernelForwardPlan != 0)
    {
        cufftDestroy(data.kernelForwardPlan);
        data.kernelForwardPlan = 0;
    }
    if (data.outputInversePlan != 0)
    {
        cufftDestroy(data.outputInversePlan);
        data.outputInversePlan = 0;
    }

    cudaFree(data.inputForwardWorkArea);
    cudaFree(data.kernelForwardWorkArea);
    cudaFree(data.outputInverseWorkArea);
    cudaFree(data.deviceInputTimeBlock);
    cudaFree(data.deviceInputFreqBlock);
    cudaFree(data.deviceInputFreqHistory);
    cudaFree(data.deviceKernelTimePartitions);
    cudaFree(data.deviceKernelFreqPartitions);
    cudaFree(data.deviceOutputFreqBlock);
    cudaFree(data.deviceOutputTimeBlock);
    cudaFree(data.deviceOverlapTail);
    cudaFree(data.deviceOutputBlock);

    data.inputForwardWorkArea = nullptr;
    data.kernelForwardWorkArea = nullptr;
    data.outputInverseWorkArea = nullptr;
    data.deviceInputTimeBlock = nullptr;
    data.deviceInputFreqBlock = nullptr;
    data.deviceInputFreqHistory = nullptr;
    data.deviceKernelTimePartitions = nullptr;
    data.deviceKernelFreqPartitions = nullptr;
    data.deviceOutputFreqBlock = nullptr;
    data.deviceOutputTimeBlock = nullptr;
    data.deviceOverlapTail = nullptr;
    data.deviceOutputBlock = nullptr;

    if (data.processingStream)
    {
        cudaStreamDestroy(data.processingStream);
        data.processingStream = nullptr;
    }
}

bool setup_r2c_plan_many(
    cufftHandle *plan,
    int batch_count,
    cudaStream_t stream,
    void **work_area,
    size_t *work_size)
{
    if (!plan || !work_area || !work_size)
    {
        return false;
    }

    *plan = 0;
    *work_area = nullptr;
    *work_size = 0;

    cufftResult result = cufftCreate(plan);
    if (result != CUFFT_SUCCESS)
    {
        return false;
    }

    result = cufftSetAutoAllocation(*plan, 0);
    if (result != CUFFT_SUCCESS)
    {
        cufftDestroy(*plan);
        *plan = 0;
        return false;
    }

    int n[1] = {kFftSize};
    int inembed[1] = {kFftSize};
    int onembed[1] = {kFftBins};
    result = cufftMakePlanMany(
        *plan,
        1,
        n,
        inembed,
        1,
        kFftSize,
        onembed,
        1,
        kFftBins,
        CUFFT_R2C,
        batch_count,
        work_size);
    if (result != CUFFT_SUCCESS)
    {
        cufftDestroy(*plan);
        *plan = 0;
        return false;
    }

    if (*work_size > 0)
    {
        cudaError_t cuda_error =
            cudaMalloc(work_area, *work_size);
        if (cuda_error != cudaSuccess)
        {
            cufftDestroy(*plan);
            *plan = 0;
            *work_area = nullptr;
            *work_size = 0;
            return false;
        }

        result = cufftSetWorkArea(*plan, *work_area);
        if (result != CUFFT_SUCCESS)
        {
            cudaFree(*work_area);
            *work_area = nullptr;
            *work_size = 0;
            cufftDestroy(*plan);
            *plan = 0;
            return false;
        }
    }

    result = cufftSetStream(*plan, stream);
    if (result != CUFFT_SUCCESS)
    {
        cudaFree(*work_area);
        *work_area = nullptr;
        *work_size = 0;
        cufftDestroy(*plan);
        *plan = 0;
        return false;
    }

    return true;
}

bool setup_c2r_plan_many(
    cufftHandle *plan,
    int batch_count,
    cudaStream_t stream,
    void **work_area,
    size_t *work_size)
{
    if (!plan || !work_area || !work_size)
    {
        return false;
    }

    *plan = 0;
    *work_area = nullptr;
    *work_size = 0;

    cufftResult result = cufftCreate(plan);
    if (result != CUFFT_SUCCESS)
    {
        return false;
    }

    result = cufftSetAutoAllocation(*plan, 0);
    if (result != CUFFT_SUCCESS)
    {
        cufftDestroy(*plan);
        *plan = 0;
        return false;
    }

    int n[1] = {kFftSize};
    int inembed[1] = {kFftBins};
    int onembed[1] = {kFftSize};
    result = cufftMakePlanMany(
        *plan,
        1,
        n,
        inembed,
        1,
        kFftBins,
        onembed,
        1,
        kFftSize,
        CUFFT_C2R,
        batch_count,
        work_size);
    if (result != CUFFT_SUCCESS)
    {
        cufftDestroy(*plan);
        *plan = 0;
        return false;
    }

    if (*work_size > 0)
    {
        cudaError_t cuda_error =
            cudaMalloc(work_area, *work_size);
        if (cuda_error != cudaSuccess)
        {
            cufftDestroy(*plan);
            *plan = 0;
            *work_area = nullptr;
            *work_size = 0;
            return false;
        }

        result = cufftSetWorkArea(*plan, *work_area);
        if (result != CUFFT_SUCCESS)
        {
            cudaFree(*work_area);
            *work_area = nullptr;
            *work_size = 0;
            cufftDestroy(*plan);
            *plan = 0;
            return false;
        }
    }

    result = cufftSetStream(*plan, stream);
    if (result != CUFFT_SUCCESS)
    {
        cudaFree(*work_area);
        *work_area = nullptr;
        *work_size = 0;
        cufftDestroy(*plan);
        *plan = 0;
        return false;
    }

    return true;
}

bool initialize_processing(
    SimpleCudaLessNaiveConvolutionData *data)
{
    if (!data)
    {
        return false;
    }

    cudaError_t cuda_error =
        cudaStreamCreate(&data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaStreamCreate", cuda_error);
        return false;
    }

    data->hostInputTimeBlock.assign(
        static_cast<size_t>(kInputChannelCount * kFftSize),
        0.0f);
    data->hostOutputBlock.assign(
        static_cast<size_t>(kOutputChannelCount * kBlockSize),
        0.0f);

    const size_t input_time_count =
        static_cast<size_t>(kInputChannelCount * kFftSize);
    const size_t input_freq_count =
        static_cast<size_t>(kInputChannelCount * kFftBins);
    const size_t input_history_freq_count =
        static_cast<size_t>(kPartitionCount * kInputChannelCount * kFftBins);
    const size_t kernel_batch_count =
        static_cast<size_t>(kOutputChannelCount * kInputChannelCount * kPartitionCount);
    const size_t kernel_time_count =
        kernel_batch_count * static_cast<size_t>(kFftSize);
    const size_t kernel_freq_count =
        kernel_batch_count * static_cast<size_t>(kFftBins);
    const size_t output_freq_count =
        static_cast<size_t>(kOutputChannelCount * kFftBins);
    const size_t output_time_count =
        static_cast<size_t>(kOutputChannelCount * kFftSize);
    const size_t overlap_count =
        static_cast<size_t>(kOutputChannelCount * kBlockSize);

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceInputTimeBlock),
        input_time_count * sizeof(cufftReal));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceInputTimeBlock)", cuda_error);
        return false;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceInputFreqBlock),
        input_freq_count * sizeof(cufftComplex));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceInputFreqBlock)", cuda_error);
        return false;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceInputFreqHistory),
        input_history_freq_count * sizeof(cufftComplex));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceInputFreqHistory)", cuda_error);
        return false;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceKernelTimePartitions),
        kernel_time_count * sizeof(cufftReal));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceKernelTimePartitions)", cuda_error);
        return false;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceKernelFreqPartitions),
        kernel_freq_count * sizeof(cufftComplex));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceKernelFreqPartitions)", cuda_error);
        return false;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceOutputFreqBlock),
        output_freq_count * sizeof(cufftComplex));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceOutputFreqBlock)", cuda_error);
        return false;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceOutputTimeBlock),
        output_time_count * sizeof(cufftReal));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceOutputTimeBlock)", cuda_error);
        return false;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceOverlapTail),
        overlap_count * sizeof(float));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceOverlapTail)", cuda_error);
        return false;
    }

    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&data->deviceOutputBlock),
        overlap_count * sizeof(float));
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMalloc(deviceOutputBlock)", cuda_error);
        return false;
    }

    return true;
}

bool finish_initialization(
    SimpleCudaLessNaiveConvolutionData *data,
    const KernelMatrix &kernel_matrix)
{
    if (!data)
    {
        return false;
    }

    const size_t kernel_batch_count =
        static_cast<size_t>(kOutputChannelCount * kInputChannelCount * kPartitionCount);

    size_t input_forward_work_size = 0;
    if (!setup_r2c_plan_many(
            &data->inputForwardPlan,
            kInputChannelCount,
            data->processingStream,
            &data->inputForwardWorkArea,
            &input_forward_work_size))
    {
        remember_cufft_error(data, "setup_r2c_plan_many(input)", CUFFT_SETUP_FAILED);
        return false;
    }

    size_t kernel_forward_work_size = 0;
    if (!setup_r2c_plan_many(
            &data->kernelForwardPlan,
            static_cast<int>(kernel_batch_count),
            data->processingStream,
            &data->kernelForwardWorkArea,
            &kernel_forward_work_size))
    {
        remember_cufft_error(data, "setup_r2c_plan_many(kernel)", CUFFT_SETUP_FAILED);
        return false;
    }

    size_t output_inverse_work_size = 0;
    if (!setup_c2r_plan_many(
            &data->outputInversePlan,
            kOutputChannelCount,
            data->processingStream,
            &data->outputInverseWorkArea,
            &output_inverse_work_size))
    {
        remember_cufft_error(data, "setup_c2r_plan_many(output)", CUFFT_SETUP_FAILED);
        return false;
    }

    const std::vector<cufftReal> host_kernel_time_partitions =
        make_kernel_time_partitions(kernel_matrix);
    cudaError_t cuda_error = cudaMemcpyAsync(
        data->deviceKernelTimePartitions,
        host_kernel_time_partitions.data(),
        host_kernel_time_partitions.size() * sizeof(cufftReal),
        cudaMemcpyHostToDevice,
        data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemcpyAsync(deviceKernelTimePartitions)", cuda_error);
        return false;
    }

    cufftResult cufft_error = cufftExecR2C(
        data->kernelForwardPlan,
        data->deviceKernelTimePartitions,
        data->deviceKernelFreqPartitions);
    if (cufft_error != CUFFT_SUCCESS)
    {
        remember_cufft_error(data, "cufftExecR2C(kernelForwardPlan)", cufft_error);
        return false;
    }

    cuda_error = cudaMemsetAsync(
        data->deviceInputFreqHistory,
        0,
        static_cast<size_t>(kPartitionCount * kInputChannelCount * kFftBins) *
            sizeof(cufftComplex),
        data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemsetAsync(deviceInputFreqHistory)", cuda_error);
        return false;
    }

    cuda_error = cudaMemsetAsync(
        data->deviceOverlapTail,
        0,
        static_cast<size_t>(kOutputChannelCount * kBlockSize) * sizeof(float),
        data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemsetAsync(deviceOverlapTail)", cuda_error);
        return false;
    }

    cuda_error = cudaMemsetAsync(
        data->deviceInputTimeBlock,
        0,
        static_cast<size_t>(kInputChannelCount * kFftSize) * sizeof(cufftReal),
        data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemsetAsync(deviceInputTimeBlock)", cuda_error);
        return false;
    }

    cuda_error = cudaMemsetAsync(
        data->deviceOutputFreqBlock,
        0,
        static_cast<size_t>(kOutputChannelCount * kFftBins) * sizeof(cufftComplex),
        data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemsetAsync(deviceOutputFreqBlock)", cuda_error);
        return false;
    }

    cufft_error = cufftExecR2C(
        data->inputForwardPlan,
        data->deviceInputTimeBlock,
        data->deviceInputFreqBlock);
    if (cufft_error != CUFFT_SUCCESS)
    {
        remember_cufft_error(data, "cufftExecR2C(inputForwardPlan warmup)", cufft_error);
        return false;
    }

    cufft_error = cufftExecC2R(
        data->outputInversePlan,
        data->deviceOutputFreqBlock,
        data->deviceOutputTimeBlock);
    if (cufft_error != CUFFT_SUCCESS)
    {
        remember_cufft_error(data, "cufftExecC2R(outputInversePlan warmup)", cufft_error);
        return false;
    }

    cuda_error = cudaStreamSynchronize(data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaStreamSynchronize(initialization)", cuda_error);
        return false;
    }

    cudaFree(data->deviceKernelTimePartitions);
    data->deviceKernelTimePartitions = nullptr;
    return true;
}

int simple_cuda_less_naive_convolution_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *timeInfo,
    PaStreamCallbackFlags,
    void *userData)
{
    SimpleCudaLessNaiveConvolutionData *data =
        static_cast<SimpleCudaLessNaiveConvolutionData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data || !output ||
        !data->deviceInputTimeBlock ||
        !data->deviceInputFreqBlock ||
        !data->deviceInputFreqHistory ||
        !data->deviceKernelFreqPartitions ||
        !data->deviceOutputFreqBlock ||
        !data->deviceOutputTimeBlock ||
        !data->deviceOverlapTail ||
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

    std::fill(data->hostInputTimeBlock.begin(), data->hostInputTimeBlock.end(), 0.0f);
    for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
    {
        cufftReal *channel_block =
            data->hostInputTimeBlock.data() + input_time_block_offset(input_channel);
        for (int frame = 0; frame < frames_to_process; ++frame)
        {
            channel_block[static_cast<size_t>(frame)] =
                input
                    ? input[static_cast<size_t>(frame * kInputChannelCount + input_channel)]
                    : 0.0f;
        }
    }

    cudaError_t cuda_error = cudaMemcpyAsync(
        data->deviceInputTimeBlock,
        data->hostInputTimeBlock.data(),
        data->hostInputTimeBlock.size() * sizeof(cufftReal),
        cudaMemcpyHostToDevice,
        data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemcpyAsync input block host->device", cuda_error);
        return finish_callback(paAbort);
    }

    cufftResult cufft_error = cufftExecR2C(
        data->inputForwardPlan,
        data->deviceInputTimeBlock,
        data->deviceInputFreqBlock);
    if (cufft_error != CUFFT_SUCCESS)
    {
        remember_cufft_error(data, "cufftExecR2C(inputForwardPlan)", cufft_error);
        return finish_callback(paAbort);
    }

    const int frequency_thread_count = 128;
    const int input_frequency_total = kInputChannelCount * kFftBins;
    store_input_frequency_history_kernel<<<
        static_cast<unsigned int>((input_frequency_total + frequency_thread_count - 1) /
                                  frequency_thread_count),
        frequency_thread_count,
        0,
        data->processingStream>>>(
        data->deviceInputFreqBlock,
        data->deviceInputFreqHistory,
        data->inputHistoryWritePartition);
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "store_input_frequency_history_kernel launch", cuda_error);
        return finish_callback(paAbort);
    }

    const int output_frequency_total = kOutputChannelCount * kFftBins;
    accumulate_output_frequency_kernel<<<
        static_cast<unsigned int>((output_frequency_total + frequency_thread_count - 1) /
                                  frequency_thread_count),
        frequency_thread_count,
        0,
        data->processingStream>>>(
        data->deviceInputFreqHistory,
        data->deviceKernelFreqPartitions,
        data->deviceOutputFreqBlock,
        data->inputHistoryWritePartition);
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "accumulate_output_frequency_kernel launch", cuda_error);
        return finish_callback(paAbort);
    }

    cufft_error = cufftExecC2R(
        data->outputInversePlan,
        data->deviceOutputFreqBlock,
        data->deviceOutputTimeBlock);
    if (cufft_error != CUFFT_SUCCESS)
    {
        remember_cufft_error(data, "cufftExecC2R(outputInversePlan)", cufft_error);
        return finish_callback(paAbort);
    }

    const int overlap_thread_count = 128;
    const int overlap_total = kOutputChannelCount * kBlockSize;
    overlap_add_and_scale_kernel<<<
        static_cast<unsigned int>((overlap_total + overlap_thread_count - 1) /
                                  overlap_thread_count),
        overlap_thread_count,
        0,
        data->processingStream>>>(
        data->deviceOutputTimeBlock,
        data->deviceOverlapTail,
        data->deviceOutputBlock);
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "overlap_add_and_scale_kernel launch", cuda_error);
        return finish_callback(paAbort);
    }

    cuda_error = cudaMemcpyAsync(
        data->hostOutputBlock.data(),
        data->deviceOutputBlock,
        data->hostOutputBlock.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaMemcpyAsync output block device->host", cuda_error);
        return finish_callback(paAbort);
    }

    cuda_error = cudaStreamSynchronize(data->processingStream);
    if (cuda_error != cudaSuccess)
    {
        remember_cuda_error(data, "cudaStreamSynchronize(callback)", cuda_error);
        return finish_callback(paAbort);
    }

    for (int frame = 0; frame < frames_to_process; ++frame)
    {
        const size_t captured_frame_index = static_cast<size_t>(data->frameIndex);
        for (int input_channel = 0; input_channel < kInputChannelCount; ++input_channel)
        {
            data->savedInputs[static_cast<size_t>(input_channel)][captured_frame_index] =
                data->hostInputTimeBlock[
                    input_time_block_offset(input_channel) + static_cast<size_t>(frame)];
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

    data->inputHistoryWritePartition++;
    if (data->inputHistoryWritePartition >= kPartitionCount)
    {
        data->inputHistoryWritePartition = 0;
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
    const KernelMatrix kernel_matrix = make_less_naive_kernel_matrix();
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

    SimpleCudaLessNaiveConvolutionData data{};
    data.maxFrames = static_cast<int>(std::llround(RUN_SECONDS * SAMPLE_RATE));
    data.sampleRate = SAMPLE_RATE;
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

    if (!initialize_processing(&data) ||
        !finish_initialization(&data, kernel_matrix))
    {
        if (data.hadCudaError)
        {
            std::cerr << "CUDA setup failed at "
                      << (data.lastCudaStage ? data.lastCudaStage : "(unknown stage)")
                      << ": " << cudaGetErrorString(data.lastCudaError) << '\n';
        }
        if (data.hadCufftError)
        {
            std::cerr << "cuFFT setup failed at "
                      << (data.lastCufftStage ? data.lastCufftStage : "(unknown stage)")
                      << ": " << cufft_result_string(data.lastCufftError)
                      << " (" << static_cast<int>(data.lastCufftError) << ")\n";
        }
        release_processing_resources(data);
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
        release_processing_resources(data);
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
        simple_cuda_less_naive_convolution_callback,
        &data);
    if (open_error != paNoError)
    {
        std::cerr << "Pa_OpenStream failed: " << Pa_GetErrorText(open_error) << '\n';
        release_processing_resources(data);
        Pa_Terminate();
        return 1;
    }

    std::cout << "Running device " << DEVICE_NAME
              << " for " << RUN_SECONDS
              << " seconds at " << SAMPLE_RATE
              << " Hz.\n";
    std::cout << "Less naive CUDA path: partitioned FFT convolution with "
              << kPartitionCount << " partitions of "
              << kBlockSize << " samples per output/input pair.\n";
    std::cout << "Mixing " << kInputChannelCount
              << " input channels into " << kOutputChannelCount
              << " output channels.\n";
    std::cout << "Kernel size: " << KERNEL_SIZE
              << " taps, FFT size: " << kFftSize
              << ", FFT bins: " << kFftBins << ".\n";

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        std::cerr << "Pa_StartStream failed: " << Pa_GetErrorText(start_error) << '\n';
        Pa_CloseStream(stream);
        release_processing_resources(data);
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
        release_processing_resources(data);
        Pa_Terminate();
        return 1;
    }

    if (data.hadCufftError)
    {
        std::cerr << "cuFFT failed during callback at "
                  << (data.lastCufftStage ? data.lastCufftStage : "(unknown stage)")
                  << ": " << cufft_result_string(data.lastCufftError)
                  << " (" << static_cast<int>(data.lastCufftError) << ")\n";
        release_processing_resources(data);
        Pa_Terminate();
        return 1;
    }

    const std::string capture_csv_path =
        std::string(PORTABLE_OUTPUT_DIR) + "/simple_cuda_less_naive_convolution_capture.csv";
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
        std::string(PORTABLE_OUTPUT_DIR) + "/simple_cuda_less_naive_convolution_callback_timing.csv";
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

    release_processing_resources(data);
    Pa_Terminate();
    return 0;
}
