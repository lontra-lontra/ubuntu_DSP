/*
Build from repo root:
  cd <repo-root>

  cmake -S Portable -B Portable/build -G Ninja
  cmake --build Portable/build --target portable_infer_topology_from_infered_topology_and_save_it

Run:
  ./Portable/build/portable_infer_topology_from_infered_topology_and_save_it

Important config is in this file:
  #define MOCK TRUE or FALSE
  #define CHANNELS 2
  #define DEVICE_NAME ...

The commands stay the same when you switch MOCK.
*/

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define MOCK TRUE

#define CHANNELS 2

#define DEVICE_NAME "infered_from_topology"

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 32
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 48000
#endif

#ifndef SWEEP_TIME_SECONDS
#define SWEEP_TIME_SECONDS 4.0f
#endif

#ifndef CHIRP_PADDING
#define CHIRP_PADDING 0.5f
#endif

#ifndef CHIRP_WINDOW_SECONDS
#define CHIRP_WINDOW_SECONDS 0.5f
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#ifndef PORTABLE_TOPOLOGY_VIEW_SCRIPT
#define PORTABLE_TOPOLOGY_VIEW_SCRIPT "Portable/scripts/show_topology_matrix.py"
#endif

#if MOCK
#include "portable/mockportaudio.h"
#include "portable/mock_devices.h"
#else
#include <portaudio.h>
#endif

#include "portable/audio_helpers.h"
#include "portable/csv_utils.h"
#include "portable/impulse_response_analysis.h"
#include "portable/usefull_singnals.h"

struct SweepCaptureData
{
    float *recorded = nullptr;
    float *played = nullptr;
    int frame_index = 0;
    int max_frames = 0;
    int active_output_channel = 0;
};

static float chirp_silence_time()
{
    const float max_padding =
        0.5f * SWEEP_TIME_SECONDS - 1.0f / static_cast<float>(SAMPLE_RATE);

    if (max_padding <= 0.0f)
    {
        return 0.0f;
    }

    return std::min(CHIRP_PADDING, max_padding);
}

static float chirp_window_time()
{
    const float active_time = SWEEP_TIME_SECONDS - 2.0f * chirp_silence_time();
    const float max_window = 0.5f * active_time;

    if (max_window <= 0.0f)
    {
        return 0.0f;
    }

    return std::min(CHIRP_WINDOW_SECONDS, max_window);
}

static float output_sample_for_channel(
    int channel,
    float t,
    int active_output_channel)
{
    if (channel != active_output_channel)
    {
        return 0.0f;
    }

    return portable_classic_chirp(
        t,
        SWEEP_TIME_SECONDS,
        chirp_silence_time(),
        20.0f,
        20000.0f,
        chirp_window_time());
}

static int pa_sweep_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags,
    void *userData)
{
    SweepCaptureData *data = static_cast<SweepCaptureData *>(userData);
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
        const int n = data->frame_index;
        const float t = static_cast<float>(n) / static_cast<float>(SAMPLE_RATE);

        for (int ch = 0; ch < CHANNELS; ++ch)
        {
            const float sample =
                output_sample_for_channel(ch, t, data->active_output_channel);
            output[i * CHANNELS + ch] = sample;
            data->played[n * CHANNELS + ch] = sample;
            data->recorded[n * CHANNELS + ch] =
                input ? input[i * CHANNELS + ch] : 0.0f;
        }

        data->frame_index++;
    }

    for (int i = frames_to_process; i < static_cast<int>(framesPerBuffer); ++i)
    {
        for (int ch = 0; ch < CHANNELS; ++ch)
        {
            output[i * CHANNELS + ch] = 0.0f;
        }
    }

    if (data->frame_index >= data->max_frames)
    {
        return paComplete;
    }

    return paContinue;
}

static bool save_time_domain_matrix_csv(
    const std::vector<std::vector<std::vector<float>>> &topology_time,
    int impulse_samples,
    const std::string &csv_path)
{
    std::vector<float> time_axis(static_cast<size_t>(impulse_samples), 0.0f);
    for (int i = 0; i < impulse_samples; ++i)
    {
        time_axis[static_cast<size_t>(i)] =
            static_cast<float>(i) / static_cast<float>(SAMPLE_RATE);
    }

    std::vector<std::string> headers;
    headers.reserve(1 + CHANNELS * CHANNELS);
    headers.push_back("time (s)");

    std::vector<const std::vector<float> *> columns;
    columns.reserve(1 + CHANNELS * CHANNELS);
    columns.push_back(&time_axis);

    for (int input_channel = 0; input_channel < CHANNELS; ++input_channel)
    {
        for (int output_channel = 0; output_channel < CHANNELS; ++output_channel)
        {
            headers.push_back(
                "h input " + std::to_string(input_channel) +
                " <- output " + std::to_string(output_channel));
            columns.push_back(
                &topology_time[static_cast<size_t>(input_channel)]
                              [static_cast<size_t>(output_channel)]);
        }
    }

    return save_arrays_to_csv(csv_path, headers, columns);
}

static bool save_frequency_domain_matrix_csv(
    const std::vector<std::vector<std::vector<std::complex<float>>>> &topology_frequency,
    int sweep_samples,
    const std::string &csv_path)
{
    const int fft_bins = sweep_samples / 2 + 1;
    std::vector<float> frequency_axis(static_cast<size_t>(fft_bins), 0.0f);
    for (int k = 0; k < fft_bins; ++k)
    {
        frequency_axis[static_cast<size_t>(k)] =
            static_cast<float>(k) * static_cast<float>(SAMPLE_RATE) /
            static_cast<float>(sweep_samples);
    }

    std::vector<std::vector<float>> storage;
    storage.reserve(static_cast<size_t>(CHANNELS * CHANNELS * 2));
    std::vector<std::string> headers;
    headers.reserve(1 + CHANNELS * CHANNELS * 2);
    headers.push_back("frequency (Hz)");

    std::vector<const std::vector<float> *> columns;
    columns.reserve(1 + CHANNELS * CHANNELS * 2);
    columns.push_back(&frequency_axis);

    for (int input_channel = 0; input_channel < CHANNELS; ++input_channel)
    {
        for (int output_channel = 0; output_channel < CHANNELS; ++output_channel)
        {
            const std::vector<std::complex<float>> &response =
                topology_frequency[static_cast<size_t>(input_channel)]
                                  [static_cast<size_t>(output_channel)];

            storage.push_back(std::vector<float>(static_cast<size_t>(fft_bins), 0.0f));
            storage.push_back(std::vector<float>(static_cast<size_t>(fft_bins), 0.0f));

            std::vector<float> &real_part = storage[storage.size() - 2];
            std::vector<float> &imag_part = storage[storage.size() - 1];

            for (int k = 0; k < fft_bins && k < static_cast<int>(response.size()); ++k)
            {
                real_part[static_cast<size_t>(k)] = response[static_cast<size_t>(k)].real();
                imag_part[static_cast<size_t>(k)] = response[static_cast<size_t>(k)].imag();
            }

            headers.push_back(
                "Re H input " + std::to_string(input_channel) +
                " <- output " + std::to_string(output_channel));
            columns.push_back(&real_part);

            headers.push_back(
                "Im H input " + std::to_string(input_channel) +
                " <- output " + std::to_string(output_channel));
            columns.push_back(&imag_part);
        }
    }

    return save_arrays_to_csv(csv_path, headers, columns);
}

int main()
{
    const int sweep_samples =
        static_cast<int>(std::round(SWEEP_TIME_SECONDS * static_cast<float>(SAMPLE_RATE)));
    const int impulse_samples =
        std::min(sweep_samples, static_cast<int>(0.5f * static_cast<float>(SAMPLE_RATE)));

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

#if !MOCK
    const PaHostApiInfo *host_api_info = Pa_GetHostApiInfo(device_info->hostApi);
#endif
    std::cout << "Selected device: [" << device_index << "] "
              << (device_info->name ? device_info->name : "(null)")
#if !MOCK
              << " | hostApi="
              << (host_api_info && host_api_info->name ? host_api_info->name : "(null)")
#endif
              << " | in=" << device_info->maxInputChannels
              << " out=" << device_info->maxOutputChannels
              << " defaultSR=" << device_info->defaultSampleRate
              << '\n';

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

    std::vector<std::vector<std::vector<float>>> topology_time(
        static_cast<size_t>(CHANNELS),
        std::vector<std::vector<float>>(
            static_cast<size_t>(CHANNELS),
            std::vector<float>(static_cast<size_t>(impulse_samples), 0.0f)));

    const int fft_bins = sweep_samples / 2 + 1;
    std::vector<std::vector<std::vector<std::complex<float>>>> topology_frequency(
        static_cast<size_t>(CHANNELS),
        std::vector<std::vector<std::complex<float>>>(
            static_cast<size_t>(CHANNELS),
            std::vector<std::complex<float>>(
                static_cast<size_t>(fft_bins),
                std::complex<float>(0.0f, 0.0f))));

    for (int output_channel = 0; output_channel < CHANNELS; ++output_channel)
    {
        std::cout << "Running chirp on output channel " << output_channel << "...\n";

        std::vector<float> played(static_cast<size_t>(sweep_samples * CHANNELS), 0.0f);
        std::vector<float> recorded(static_cast<size_t>(sweep_samples * CHANNELS), 0.0f);

        SweepCaptureData data{};
        data.recorded = recorded.data();
        data.played = played.data();
        data.frame_index = 0;
        data.max_frames = sweep_samples;
        data.active_output_channel = output_channel;

        PaStream *stream = nullptr;
        err = Pa_OpenStream(
            &stream,
            &input_parameters,
            &output_parameters,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
            paClipOff,
            pa_sweep_callback,
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
            Pa_Sleep(50);
        }

        Pa_StopStream(stream);
        Pa_CloseStream(stream);

        std::vector<float> played_out(static_cast<size_t>(sweep_samples), 0.0f);
        for (int i = 0; i < sweep_samples; ++i)
        {
            played_out[static_cast<size_t>(i)] =
                played[static_cast<size_t>(i * CHANNELS + output_channel)];
        }

        for (int input_channel = 0; input_channel < CHANNELS; ++input_channel)
        {
            std::vector<float> recorded_in(static_cast<size_t>(sweep_samples), 0.0f);
            for (int i = 0; i < sweep_samples; ++i)
            {
                recorded_in[static_cast<size_t>(i)] =
                    recorded[static_cast<size_t>(i * CHANNELS + input_channel)];
            }

            const std::vector<std::complex<float>> frequency_response =
                estimate_frequency_response_half(played_out, recorded_in);
            topology_frequency[static_cast<size_t>(input_channel)]
                              [static_cast<size_t>(output_channel)] =
                frequency_response;

            const std::vector<float> impulse_response =
                frequency_response_half_to_time_domain(frequency_response, sweep_samples);

            const int copy_samples =
                std::min(impulse_samples, static_cast<int>(impulse_response.size()));
            for (int i = 0; i < copy_samples; ++i)
            {
                topology_time[static_cast<size_t>(input_channel)]
                             [static_cast<size_t>(output_channel)]
                             [static_cast<size_t>(i)] =
                    impulse_response[static_cast<size_t>(i)];
            }
        }
    }

    Pa_Terminate();

    const std::string base_path =
        std::string(PORTABLE_OUTPUT_DIR) +
        "/infered_topology_ch_" + std::to_string(CHANNELS);
    const std::string time_csv_path = base_path + "_time_domain_matrix.csv";
    const std::string frequency_csv_path = base_path + "_frequency_domain_matrix.csv";

    if (!save_time_domain_matrix_csv(topology_time, impulse_samples, time_csv_path))
    {
        std::cerr << "Failed to save time-domain topology CSV: " << time_csv_path << '\n';
        return 1;
    }

    if (!save_frequency_domain_matrix_csv(topology_frequency, sweep_samples, frequency_csv_path))
    {
        std::cerr << "Failed to save frequency-domain topology CSV: " << frequency_csv_path << '\n';
        return 1;
    }

    std::cout << "Saved time-domain matrix: " << time_csv_path << '\n';
    std::cout << "Saved frequency-domain matrix: " << frequency_csv_path << '\n';
    std::cout << "Device index: " << device_index << '\n';
    std::cout << "CHANNELS=" << CHANNELS
              << " SAMPLE_RATE=" << SAMPLE_RATE
              << " SWEEP_TIME_SECONDS=" << SWEEP_TIME_SECONDS
              << " CHIRP_SILENCE_SECONDS=" << chirp_silence_time()
              << " CHIRP_WINDOW_SECONDS=" << chirp_window_time()
              << '\n';

    std::ostringstream topology_command;
#ifdef _WIN32
    topology_command << "py -3 ";
#else
    topology_command << "python3 ";
#endif
    topology_command << '"' << PORTABLE_TOPOLOGY_VIEW_SCRIPT << "\" "
                     << '"' << frequency_csv_path << '"';
    std::cout << "Topology viewer command: " << topology_command.str() << '\n';

    return 0;
}
