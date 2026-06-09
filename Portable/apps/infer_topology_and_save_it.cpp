/*
Build from repo root:
  direct machine:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=infer_topology_and_save_it -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=OFF -DPORTABLE_DEVICE_NAME="MADIface USB (24285073): Audio (hw:2,0)" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32
    cmake --build Portable/build --target portable_infer_topology_and_save_it --parallel
    ./Portable/build/portable_infer_topology_and_save_it

  JACK:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=infer_topology_and_save_it -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=ON -DPORTABLE_DEVICE_NAME="system" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32
    cmake --build Portable/build --target portable_infer_topology_and_save_it --parallel
    ./Portable/build/portable_infer_topology_and_save_it

To try mock audio instead, rerun either `cmake -S ...` command with `-DPORTABLE_USE_MOCK=ON`.

Build-time config:
  `CHANNELS` is fixed at 32 in this file.
  `DEVICE_NAME`, `SAMPLE_RATE`, and `FRAMES_PER_BUFFER` come from the CMake command above.

Prepare JACK for direct hardware access:
  systemctl --user mask --runtime pipewire.service pipewire.socket pipewire-pulse.service pipewire-pulse.socket wireplumber.service
  systemctl --user stop pipewire.service pipewire.socket pipewire-pulse.service pipewire-pulse.socket wireplumber.service
  killall pipewire pipewire-pulse wireplumber

Start JACK in another terminal and leave it running:
  jackd -d alsa -d hw:2,0 -r 44100 -p 32 -n 3

Restore desktop audio afterwards:
  killall jackd
  systemctl --user unmask --runtime pipewire.service pipewire.socket pipewire-pulse.service pipewire-pulse.socket wireplumber.service
  systemctl --user start wireplumber.service pipewire.service pipewire-pulse.service
*/

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <sstream>
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

#ifndef AMPLITUDE
#define AMPLITUDE 0.3
#endif

#define CHANNELS 32

#ifndef DEVICE_NAME
#define DEVICE_NAME "system"
#endif

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 32
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 44100
#endif

#ifndef SILENCE_BETWEEN_CHIRPS_SECONDS
#define SILENCE_BETWEEN_CHIRPS_SECONDS 1.0f
#endif

#ifndef CHIRP_INCREASE_AND_DECREASE_TIME_SECONDS
#define CHIRP_INCREASE_AND_DECREASE_TIME_SECONDS 0.5f
#endif

#ifndef PURE_CHIRP_TIME_SECONDS
#define PURE_CHIRP_TIME_SECONDS 1.0f
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#ifndef PORTABLE_PLOT_SCRIPT
#define PORTABLE_PLOT_SCRIPT "Portable/scripts/plot.py"
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

static void print_output_sweep_summary(
    const std::vector<float> &played,
    const std::vector<float> &recorded,
    int sweep_samples,
    int active_output_channel)
{
    if (sweep_samples <= 0)
    {
        return;
    }

    float played_active_peak = 0.0f;
    float played_other_peak = 0.0f;
    float strongest_recorded_peak = 0.0f;
    int strongest_input_channel = -1;

    for (int frame = 0; frame < sweep_samples; ++frame)
    {
        for (int channel = 0; channel < CHANNELS; ++channel)
        {
            const float played_sample =
                played[static_cast<size_t>(frame * CHANNELS + channel)];
            if (channel == active_output_channel)
            {
                played_active_peak =
                    std::max(played_active_peak, std::fabs(played_sample));
            }
            else
            {
                played_other_peak =
                    std::max(played_other_peak, std::fabs(played_sample));
            }
        }
    }

    for (int input_channel = 0; input_channel < CHANNELS; ++input_channel)
    {
        float input_peak = 0.0f;
        for (int frame = 0; frame < sweep_samples; ++frame)
        {
            input_peak = std::max(
                input_peak,
                std::fabs(
                    recorded[static_cast<size_t>(frame * CHANNELS + input_channel)]));
        }

        if (input_peak > strongest_recorded_peak)
        {
            strongest_recorded_peak = input_peak;
            strongest_input_channel = input_channel;
        }
    }

    std::cout << "Output sweep summary:"
              << " out_ch=" << active_output_channel
              << " played_active_peak=" << played_active_peak
              << " played_other_peak=" << played_other_peak
              << " strongest_input_ch=" << strongest_input_channel
              << " strongest_recorded_peak=" << strongest_recorded_peak
              << '\n';

    if (played_active_peak <= 1.0e-6f)
    {
        std::cerr << "Warning: active output channel "
                  << active_output_channel
                  << " was generated near zero inside the app.\n";
    }

    if (strongest_recorded_peak <= 1.0e-6f)
    {
        std::cerr << "Warning: output channel "
                  << active_output_channel
                  << " produced near-zero response on every recorded input channel.\n";
    }
}

static void print_matching_device_entries(const char *target_name)
{
    if (!target_name || !target_name[0])
    {
        return;
    }

    const int device_count = Pa_GetDeviceCount();
    if (device_count <= 0)
    {
        return;
    }

    int match_count = 0;
    for (int device_index = 0; device_index < device_count; ++device_index)
    {
        const PaDeviceInfo *device_info = Pa_GetDeviceInfo(device_index);
        if (!device_info || !device_info->name)
        {
            continue;
        }

        if (std::string(device_info->name) != std::string(target_name))
        {
            continue;
        }

        match_count++;
#if !MOCK
        const PaHostApiInfo *host_api_info =
            Pa_GetHostApiInfo(device_info->hostApi);
#endif
        std::cout << "Matching device entry: [" << device_index << "] "
                  << device_info->name
                  << " | in=" << device_info->maxInputChannels
                  << " out=" << device_info->maxOutputChannels
                  << " defaultSR=" << device_info->defaultSampleRate
#if !MOCK
                  << " hostApi="
                  << (host_api_info && host_api_info->name
                          ? host_api_info->name
                          : "(null)")
#endif
                  << '\n';
    }

    if (match_count > 1)
    {
        std::cout << "Warning: DEVICE_NAME matched "
                  << match_count
                  << " PortAudio entries. The app uses the first exact match.\n";
    }
}

static float chirp_total_period_seconds()
{
    return SILENCE_BETWEEN_CHIRPS_SECONDS +
           2.0f * CHIRP_INCREASE_AND_DECREASE_TIME_SECONDS +
           PURE_CHIRP_TIME_SECONDS;
}

static float chirp_silence_time()
{
    return 0.5f * SILENCE_BETWEEN_CHIRPS_SECONDS;
}

static float chirp_window_time()
{
    return CHIRP_INCREASE_AND_DECREASE_TIME_SECONDS;
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
        chirp_total_period_seconds(),
        chirp_silence_time(),
        20.0f,
        20000.0f,
        chirp_window_time()) * AMPLITUDE;
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

static bool save_raw_capture_csv(
    const std::vector<float> &played,
    const std::vector<float> &recorded,
    int sweep_samples,
    const std::string &csv_path)
{
    if (static_cast<int>(played.size()) != sweep_samples * CHANNELS ||
        static_cast<int>(recorded.size()) != sweep_samples * CHANNELS)
    {
        return false;
    }

    std::vector<float> time_axis(static_cast<size_t>(sweep_samples), 0.0f);
    for (int i = 0; i < sweep_samples; ++i)
    {
        time_axis[static_cast<size_t>(i)] =
            static_cast<float>(i) / static_cast<float>(SAMPLE_RATE);
    }

    std::vector<std::vector<float>> storage;
    storage.reserve(static_cast<size_t>(CHANNELS * 2));
    std::vector<std::string> headers;
    headers.reserve(1 + CHANNELS * 2);
    headers.push_back("time (s)");

    std::vector<const std::vector<float> *> columns;
    columns.reserve(1 + CHANNELS * 2);
    columns.push_back(&time_axis);

    for (int channel = 0; channel < CHANNELS; ++channel)
    {
        storage.push_back(std::vector<float>(static_cast<size_t>(sweep_samples), 0.0f));
        std::vector<float> &played_channel = storage.back();
        for (int i = 0; i < sweep_samples; ++i)
        {
            played_channel[static_cast<size_t>(i)] =
                played[static_cast<size_t>(i * CHANNELS + channel)];
        }

        headers.push_back("played ch " + std::to_string(channel));
        columns.push_back(&played_channel);
    }

    for (int channel = 0; channel < CHANNELS; ++channel)
    {
        storage.push_back(std::vector<float>(static_cast<size_t>(sweep_samples), 0.0f));
        std::vector<float> &recorded_channel = storage.back();
        for (int i = 0; i < sweep_samples; ++i)
        {
            recorded_channel[static_cast<size_t>(i)] =
                recorded[static_cast<size_t>(i * CHANNELS + channel)];
        }

        headers.push_back("recorded ch " + std::to_string(channel));
        columns.push_back(&recorded_channel);
    }

    return save_arrays_to_csv(csv_path, headers, columns);
}

static std::string build_python_command(
    const std::string &script_path,
    const std::string &csv_path)
{
    std::ostringstream command;
#ifdef _WIN32
    command << "py -3 ";
#else
    command << "python3 ";
#endif
    command << '"' << script_path << "\" "
            << '"' << csv_path << '"';
    return command.str();
}

int main()
{
    const int sweep_samples =
        static_cast<int>(std::round(
            chirp_total_period_seconds() * static_cast<float>(SAMPLE_RATE)));
    const int impulse_samples =
        std::min(sweep_samples, static_cast<int>(0.5f * static_cast<float>(SAMPLE_RATE)));
    const std::string base_path =
        std::string(PORTABLE_OUTPUT_DIR) +
        "/infered_topology_ch_" + std::to_string(CHANNELS);

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
    print_matching_device_entries(DEVICE_NAME);

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
    std::vector<std::string> raw_csv_paths;
    raw_csv_paths.reserve(static_cast<size_t>(CHANNELS));

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

        print_output_sweep_summary(
            played,
            recorded,
            sweep_samples,
            output_channel);

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

        const std::string raw_csv_path =
            base_path +
            "_raw_capture_output_" + std::to_string(output_channel) + ".csv";
        if (!save_raw_capture_csv(played, recorded, sweep_samples, raw_csv_path))
        {
            std::cerr << "Failed to save raw capture CSV: " << raw_csv_path << '\n';
            Pa_Terminate();
            return 1;
        }

        raw_csv_paths.push_back(raw_csv_path);
        std::cout << "Saved raw capture: " << raw_csv_path << '\n';
    }

    Pa_Terminate();

    const std::string time_csv_path = base_path + "_time_domain_matrix.csv";
    const std::string frequency_csv_path = base_path + "_frequency_domain_matrix.csv";

    if (!save_time_domain_matrix_csv(topology_time, impulse_samples, time_csv_path))
    {
        std::cerr << "Failed to save time-domain topology CSV: " << time_csv_path << '\n';
        return 1;
    }

    if (!save_frequency_domain_matrix_csv(topology_frequency, sweep_samples, frequency_csv_path))
    {
        std::cerr << "Failed to save frequency-domain topology CSV: " << frequency_csv_path
                  << '\n';
        return 1;
    }

    std::cout << "Saved time-domain matrix: " << time_csv_path << '\n';
    std::cout << "Saved frequency-domain matrix: " << frequency_csv_path << '\n';
    std::cout << "Device index: " << device_index << '\n';
    std::cout << "CHANNELS=" << CHANNELS
              << " SAMPLE_RATE=" << SAMPLE_RATE
              << " CHIRP_TOTAL_PERIOD_SECONDS=" << chirp_total_period_seconds()
              << " SILENCE_BETWEEN_CHIRPS_SECONDS="
              << SILENCE_BETWEEN_CHIRPS_SECONDS
              << " CHIRP_INCREASE_AND_DECREASE_TIME_SECONDS="
              << CHIRP_INCREASE_AND_DECREASE_TIME_SECONDS
              << " PURE_CHIRP_TIME_SECONDS=" << PURE_CHIRP_TIME_SECONDS
              << '\n';

    for (size_t output_channel = 0; output_channel < raw_csv_paths.size(); ++output_channel)
    {
        std::cout << "Raw plot command (output " << output_channel << "): "
                  << build_python_command(PORTABLE_PLOT_SCRIPT, raw_csv_paths[output_channel])
                  << '\n';
    }

    std::cout << "Impulse plot command: "
              << build_python_command(PORTABLE_PLOT_SCRIPT, time_csv_path) << '\n';
    std::cout << "Topology viewer command: "
              << build_python_command(PORTABLE_TOPOLOGY_VIEW_SCRIPT, frequency_csv_path)
              << '\n';

    return 0;
}
