/*
Build from repo root:

  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=simple_cuda_portaudio -DPORTABLE_USE_MOCK=ON
  cmake --build Portable/build --target portable_simple_cuda_portaudio --parallel

If `Portable/build` does not exist yet, came from another machine, or you changed
`PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` line before
the `cmake --build ...` line.

Run:
  ./Portable/build/portable_simple_cuda_portaudio

To try real PortAudio instead, rerun the first command with `-DPORTABLE_USE_MOCK=OFF`.

This is intentionally a very small mock PortAudio + CUDA example.
The mock device generates two sine-wave input channels.
Inside the callback we copy one audio frame at a time to the GPU,
launch <<<1, 2>>>, and copy the two processed samples back.
That is deliberately simple, not efficient.
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

#define CHANNELS 2

#define DEVICE_NAME "simple_cuda_portaudio_mock"

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
#include "portable/mock_device_registry.h"

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct SimpleCudaPortAudioData
{
    // These two buffers live on the GPU.
    // They only hold one audio frame worth of channel data:
    // input[0], input[1] and output[0], output[1].
    float *deviceInput = nullptr;
    float *deviceOutput = nullptr;

    int frameIndex = 0;
    int maxFrames = 0;
    double sampleRate = SAMPLE_RATE;
    bool hasFirstCallbackTime = false;
    double firstCallbackTimeSeconds = 0.0;

    // The callback cannot throw, so we remember the first CUDA error
    // and stop the stream as soon as it happens.
    bool hadCudaError = false;
    cudaError_t lastCudaError = cudaSuccess;
    const char *lastCudaStage = nullptr;

    // These host-side vectors let us inspect what happened after the run.
    std::vector<float> savedInput0;
    std::vector<float> savedInput1;
    std::vector<float> savedOutput0;
    std::vector<float> savedOutput1;
    std::vector<float> savedCallbackIndex;
    std::vector<float> savedCallbackTimeSeconds;
    std::vector<float> savedCallbackDelaySeconds;
    std::vector<float> savedCallbackAllowedSeconds;
    std::vector<float> savedCallbackLoadPercent;
};

void remember_cuda_error(
    SimpleCudaPortAudioData *data,
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

// This kernel is intentionally tiny:
// - one block
// - two threads
// - one thread owns one channel
//
// Thread 0 scales channel 0 by 1/2.
// Thread 1 scales channel 1 by 1/3.
__global__ void scale_two_channels_kernel(
    const float *input,
    float *output)
{
    const int channel = static_cast<int>(threadIdx.x);
    if (channel == 0)
    {
        output[0] = input[0] * 0.5f;
    }
    else if (channel == 1)
    {
        output[1] = input[1] / 3.0f;
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

void register_simple_mock_device()
{
    clear_mock_devices();

    std::vector<MockInputGenerator> input_generators;
    input_generators.reserve(CHANNELS);

    input_generators.push_back([](double time_seconds) -> float
    {
        return sine_sample(
            time_seconds,
            INPUT0_FREQUENCY_HZ,
            INPUT0_AMPLITUDE);
    });

    input_generators.push_back([](double time_seconds) -> float
    {
        return sine_sample(
            time_seconds,
            INPUT1_FREQUENCY_HZ,
            INPUT1_AMPLITUDE,
            0.3);
    });

    register_device_that_expects_exact_convolution(
        DEVICE_NAME,
        SAMPLE_RATE,
        {SAMPLE_RATE},
        input_generators,
        {
            {
                {0.5f},
                {0.0f},
            },
            {
                {0.0f},
                {1.0f / 3.0f},
            },
        });
}

std::string build_python_command(
    const std::string &script_path,
    const std::string &csv_path)
{
    std::string command = "py -3 ";
    command += '"' + script_path + "\" \"" + csv_path + '"';
    return command;
}

bool save_capture_csv(
    const SimpleCudaPortAudioData &data,
    const std::string &csv_path)
{
    const size_t sample_count = data.savedInput0.size();
    if (data.savedInput1.size() != sample_count ||
        data.savedOutput0.size() != sample_count ||
        data.savedOutput1.size() != sample_count)
    {
        return false;
    }

    std::vector<float> time_axis(sample_count, 0.0f);
    for (size_t i = 0; i < sample_count; ++i)
    {
        time_axis[i] = static_cast<float>(static_cast<double>(i) / SAMPLE_RATE);
    }

    return save_arrays_to_csv(
        csv_path,
        {
            "time (s)",
            "input 0",
            "input 1",
            "output 0",
            "output 1"},
        {
            &time_axis,
            &data.savedInput0,
            &data.savedInput1,
            &data.savedOutput0,
            &data.savedOutput1});
}

void record_callback_timing(
    SimpleCudaPortAudioData *data,
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
    const SimpleCudaPortAudioData &data,
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

int simple_cuda_portaudio_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *timeInfo,
    PaStreamCallbackFlags,
    void *userData)
{
    SimpleCudaPortAudioData *data =
        static_cast<SimpleCudaPortAudioData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data || !output || !data->deviceInput || !data->deviceOutput)
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

    const int frames_left = data->maxFrames - data->frameIndex;
    const int frames_to_process =
        std::max(0, std::min(static_cast<int>(framesPerBuffer), frames_left));

    for (int frame = 0; frame < frames_to_process; ++frame)
    {
        const float host_input[CHANNELS] = {
            input ? input[frame * CHANNELS + 0] : 0.0f,
            input ? input[frame * CHANNELS + 1] : 0.0f};
        float host_output[CHANNELS] = {0.0f, 0.0f};

        // Step 1:
        // Copy the current two-channel audio frame from host RAM to GPU RAM.
        cudaError_t error = cudaMemcpy(
            data->deviceInput,
            host_input,
            sizeof(host_input),
            cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            remember_cuda_error(data, "cudaMemcpy host->device", error);
            return finish_callback(paAbort);
        }

        // Step 2:
        // Launch exactly one block with exactly two threads.
        // Thread 0 handles channel 0.
        // Thread 1 handles channel 1.
        scale_two_channels_kernel<<<1, CHANNELS>>>(data->deviceInput, data->deviceOutput);

        // Step 3:
        // Check whether the kernel launch itself was accepted by CUDA.
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            remember_cuda_error(data, "kernel launch", error);
            return finish_callback(paAbort);
        }

        // Step 4:
        // Copy the two processed samples back to the CPU.
        // This blocking copy also makes sure the kernel has finished.
        error = cudaMemcpy(
            host_output,
            data->deviceOutput,
            sizeof(host_output),
            cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            remember_cuda_error(data, "cudaMemcpy device->host", error);
            return finish_callback(paAbort);
        }

        output[frame * CHANNELS + 0] = host_output[0];
        output[frame * CHANNELS + 1] = host_output[1];

        data->savedInput0[static_cast<size_t>(data->frameIndex)] = host_input[0];
        data->savedInput1[static_cast<size_t>(data->frameIndex)] = host_input[1];
        data->savedOutput0[static_cast<size_t>(data->frameIndex)] = host_output[0];
        data->savedOutput1[static_cast<size_t>(data->frameIndex)] = host_output[1];
        data->frameIndex++;
    }

    for (unsigned long frame = static_cast<unsigned long>(frames_to_process);
         frame < framesPerBuffer;
         ++frame)
    {
        output[frame * CHANNELS + 0] = 0.0f;
        output[frame * CHANNELS + 1] = 0.0f;
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
    register_simple_mock_device();

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

    SimpleCudaPortAudioData data{};
    data.maxFrames = static_cast<int>(std::llround(RUN_SECONDS * SAMPLE_RATE));
    data.sampleRate = SAMPLE_RATE;
    data.savedInput0.assign(static_cast<size_t>(data.maxFrames), 0.0f);
    data.savedInput1.assign(static_cast<size_t>(data.maxFrames), 0.0f);
    data.savedOutput0.assign(static_cast<size_t>(data.maxFrames), 0.0f);
    data.savedOutput1.assign(static_cast<size_t>(data.maxFrames), 0.0f);
    const size_t estimated_callback_count =
        (static_cast<size_t>(data.maxFrames) +
         static_cast<size_t>(FRAMES_PER_BUFFER) - 1u) /
        static_cast<size_t>(FRAMES_PER_BUFFER);
    data.savedCallbackIndex.reserve(estimated_callback_count);
    data.savedCallbackTimeSeconds.reserve(estimated_callback_count);
    data.savedCallbackDelaySeconds.reserve(estimated_callback_count);
    data.savedCallbackAllowedSeconds.reserve(estimated_callback_count);
    data.savedCallbackLoadPercent.reserve(estimated_callback_count);

    // We allocate the tiny CUDA buffers once up front.
    // The callback will then reuse them for every audio frame.
    cuda_error = cudaMalloc(reinterpret_cast<void **>(&data.deviceInput), CHANNELS * sizeof(float));
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaMalloc(deviceInput) failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        Pa_Terminate();
        return 1;
    }

    cuda_error = cudaMalloc(reinterpret_cast<void **>(&data.deviceOutput), CHANNELS * sizeof(float));
    if (cuda_error != cudaSuccess)
    {
        std::cerr << "cudaMalloc(deviceOutput) failed: "
                  << cudaGetErrorString(cuda_error) << '\n';
        cudaFree(data.deviceInput);
        Pa_Terminate();
        return 1;
    }

    PaStreamParameters input_parameters{};
    input_parameters.device = device_index;
    input_parameters.channelCount = CHANNELS;
    input_parameters.sampleFormat = SAMPLE_FORMAT;
    input_parameters.suggestedLatency = device_info->defaultLowInputLatency;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = CHANNELS;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = device_info->defaultLowOutputLatency;

    if (!check_if_device_respects_input_and_output_stream_specs(
            &input_parameters,
            &output_parameters,
            SAMPLE_RATE))
    {
        cudaFree(data.deviceOutput);
        cudaFree(data.deviceInput);
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
        simple_cuda_portaudio_callback,
        &data);
    if (open_error != paNoError)
    {
        std::cerr << "Pa_OpenStream failed: " << Pa_GetErrorText(open_error) << '\n';
        cudaFree(data.deviceOutput);
        cudaFree(data.deviceInput);
        Pa_Terminate();
        return 1;
    }

    std::cout << "Running mock device " << DEVICE_NAME
              << " for " << RUN_SECONDS
              << " seconds at " << SAMPLE_RATE
              << " Hz.\n";
    std::cout << "The callback sends each audio frame to CUDA with <<<1, 2>>>.\n";

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        std::cerr << "Pa_StartStream failed: " << Pa_GetErrorText(start_error) << '\n';
        Pa_CloseStream(stream);
        cudaFree(data.deviceOutput);
        cudaFree(data.deviceInput);
        Pa_Terminate();
        return 1;
    }

    Pa_CloseStream(stream);

    if (data.hadCudaError)
    {
        std::cerr << "CUDA failed during callback at "
                  << (data.lastCudaStage ? data.lastCudaStage : "(unknown stage)")
                  << ": " << cudaGetErrorString(data.lastCudaError) << '\n';
        cudaFree(data.deviceOutput);
        cudaFree(data.deviceInput);
        Pa_Terminate();
        return 1;
    }

    const std::string capture_csv_path =
        std::string(PORTABLE_OUTPUT_DIR) + "/simple_cuda_portaudio_capture.csv";
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
        std::string(PORTABLE_OUTPUT_DIR) + "/simple_cuda_portaudio_callback_timing.csv";
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
    std::cout << "Expected mapping: output 0 = input 0 / 2, output 1 = input 1 / 3\n";

    cudaFree(data.deviceOutput);
    cudaFree(data.deviceInput);
    Pa_Terminate();
    return 0;
}
