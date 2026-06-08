/*
Build from repo root:
  direct machine:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=sound_device_query -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=OFF -DPORTABLE_DEVICE_NAME="MADIface USB (24285073): Audio (hw:2,0)" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32
    cmake --build Portable/build --target portable_sound_device_query --parallel
    ./Portable/build/portable_sound_device_query

  JACK:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=sound_device_query -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=ON -DPORTABLE_DEVICE_NAME="system" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32
    cmake --build Portable/build --target portable_sound_device_query --parallel
    ./Portable/build/portable_sound_device_query

To try mock audio instead, rerun either `cmake -S ...` command with `-DPORTABLE_USE_MOCK=ON`.

Build-time config:
  `CHANNELS` is fixed in this file.
  `DEVICE_NAME`, `SAMPLE_RATE`, and `FRAMES_PER_BUFFER` come from the CMake command above.
  `SAMPLE_FORMAT` still lives in this file.
  app-specific timing config still lives in this file.

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
*/

#include <algorithm>
#include <cmath>
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

#ifndef DEVICE_NAME
#define DEVICE_NAME "MADIface USB (24285073): Audio (hw:1,0)"
#endif

#ifndef CHANNELS
#define CHANNELS 32
#endif

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 256
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 44100
#endif

#ifndef PLAY_AND_LISTEN_SECONDS
#define PLAY_AND_LISTEN_SECONDS 5.0
#endif

#ifndef LISTEN_SECONDS
#define LISTEN_SECONDS 2.0
#endif

#ifndef TONE_FREQUENCY_HZ
#define TONE_FREQUENCY_HZ 400.0
#endif

#ifndef TONE_AMPLITUDE
#define TONE_AMPLITUDE 0.8
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "sysdefault"
#endif

#ifndef PORTABLE_PLOT_SCRIPT
#define PORTABLE_PLOT_SCRIPT "Portable/scripts/plot.py"
#endif

#if MOCK
#include "portable/mockportaudio.h"
#include "portable/mock_devices.h"
#else
#include <portaudio.h>
#if defined(PA_USE_ASIO)
#include <pa_asio.h>
#endif
#endif

#include "portable/audio_helpers.h"
#include "portable/csv_utils.h"
#include "portable/plotting.h"

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct SoundDeviceQueryCaptureData
{
    float *captured = nullptr;
    float *played_reference = nullptr;
    int frame_index = 0;
    int max_frames = 0;
    int play_and_listen_frames = 0;
    int output_channels = 0;
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
};

int active_output_channel_for_frame(
    int absolute_frame,
    int play_and_listen_frames,
    int output_channels)
{
    if (absolute_frame < 0 ||
        absolute_frame >= play_and_listen_frames ||
        play_and_listen_frames <= 0 ||
        output_channels <= 0)
    {
        return -1;
    }

    const long long scaled_frame =
        static_cast<long long>(absolute_frame) *
        static_cast<long long>(output_channels);
    const int active_channel = static_cast<int>(
        scaled_frame / static_cast<long long>(play_and_listen_frames));
    return std::max(0, std::min(output_channels - 1, active_channel));
}

int sound_device_query_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    SoundDeviceQueryCaptureData *data =
        static_cast<SoundDeviceQueryCaptureData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data || !data->captured || !data->played_reference)
    {
        return paAbort;
    }

    if ((statusFlags & paInputUnderflow) != 0)
    {
        data->input_underflow_count++;
    }
    if ((statusFlags & paInputOverflow) != 0)
    {
        data->input_overflow_count++;
    }
    if ((statusFlags & paOutputUnderflow) != 0)
    {
        data->output_underflow_count++;
    }
    if ((statusFlags & paOutputOverflow) != 0)
    {
        data->output_overflow_count++;
    }

    const int frames_left = data->max_frames - data->frame_index;
    const int frames_to_process =
        std::max(0, std::min(static_cast<int>(framesPerBuffer), frames_left));

    for (int i = 0; i < frames_to_process; ++i)
    {
        const int absolute_frame = data->frame_index + i;
        float tone_sample = 0.0f;
        const int active_output_channel = active_output_channel_for_frame(
            absolute_frame,
            data->play_and_listen_frames,
            data->output_channels);

        if (active_output_channel >= 0)
        {
            const double t =
                static_cast<double>(absolute_frame) /
                static_cast<double>(SAMPLE_RATE);
            tone_sample = static_cast<float>(
                TONE_AMPLITUDE * std::sin(2.0 * kPi * TONE_FREQUENCY_HZ * t));
        }

        data->played_reference[static_cast<size_t>(absolute_frame)] = tone_sample;

        for (int ch = 0; ch < CHANNELS; ++ch)
        {
            data->captured[static_cast<size_t>(absolute_frame * CHANNELS + ch)] =
                input ? input[static_cast<size_t>(i * CHANNELS + ch)] : 0.0f;
        }

        if (output)
        {
            for (int ch = 0; ch < data->output_channels; ++ch)
            {
                output[static_cast<size_t>(i * data->output_channels + ch)] =
                    ch == active_output_channel ? tone_sample : 0.0f;
            }
        }
    }

    if (output)
    {
        for (int i = frames_to_process; i < static_cast<int>(framesPerBuffer); ++i)
        {
            for (int ch = 0; ch < data->output_channels; ++ch)
            {
                output[static_cast<size_t>(i * data->output_channels + ch)] =
                    0.0f;
            }
        }
    }

    data->frame_index += frames_to_process;
    return data->frame_index >= data->max_frames ? paComplete : paContinue;
}

void print_requested_config()
{
    std::cout << "Requested config:"
              << " DEVICE_NAME=" << DEVICE_NAME
              << " CHANNELS=" << CHANNELS
              << " SAMPLE_RATE=" << SAMPLE_RATE
              << " FRAMES_PER_BUFFER=" << FRAMES_PER_BUFFER
              << " PLAY_AND_LISTEN_SECONDS=" << PLAY_AND_LISTEN_SECONDS
              << " LISTEN_SECONDS=" << LISTEN_SECONDS
              << " TONE_FREQUENCY_HZ=" << TONE_FREQUENCY_HZ
              << '\n';
}

void print_selected_device_summary(
    int device_index,
    const PaDeviceInfo *device_info)
{
    if (!device_info)
    {
        return;
    }

#if !MOCK
    const PaHostApiInfo *host_api_info = Pa_GetHostApiInfo(device_info->hostApi);
#endif

    std::cout << "Selected device: [" << device_index << "] "
              << (device_info->name ? device_info->name : "(null)")
              << " | in=" << device_info->maxInputChannels
              << " out=" << device_info->maxOutputChannels
              << " defaultSR=" << device_info->defaultSampleRate
#if !MOCK
              << " hostApi="
              << (host_api_info && host_api_info->name ? host_api_info->name : "(null)")
#endif
              << '\n';

    const std::vector<double> input_rates =
        get_possible_sample_rates(device_index, true);
    if (!input_rates.empty())
    {
        std::cout << "Supported input rates:";
        for (double rate : input_rates)
        {
            std::cout << ' ' << rate;
        }
        std::cout << '\n';
    }
}

bool looks_like_generic_alsa_plugin_device(
    const PaDeviceInfo *device_info)
{
    if (!device_info || !device_info->name)
    {
        return false;
    }

#if !MOCK
    const PaHostApiInfo *host_api_info = Pa_GetHostApiInfo(device_info->hostApi);
    if (!host_api_info || !host_api_info->name ||
        std::string(host_api_info->name) != "ALSA")
    {
        return false;
    }
#endif

    const std::string name = portable_ascii_lower(device_info->name);
    return name == "default" ||
           name == "sysdefault" ||
           name == "front" ||
           name == "rear" ||
           name == "side" ||
           name == "dmix" ||
           name == "dsnoop" ||
           name == "surround40" ||
           name == "surround41" ||
           name == "surround50" ||
           name == "surround51" ||
           name == "surround71";
}

void maybe_warn_about_generic_alsa_plugin_device(
    const PaDeviceInfo *device_info)
{
    if (!looks_like_generic_alsa_plugin_device(device_info))
    {
        return;
    }

    std::cout
        << "Warning: selected ALSA plugin device '"
        << (device_info && device_info->name ? device_info->name : "(null)")
        << "'. These virtual PCMs can expose synthetic channel counts and "
        << "often do not behave like the underlying hardware for multichannel "
        << "full-duplex tests.\n"
        << "Prefer a hardware match such as 'MADIface USB', 'hw:1,0', or a "
        << "PortAudio device name that includes '(hw:...)' or '(plughw:...)'.\n";
}

void show_failure_context(
    int device_index,
    const PaDeviceInfo *device_info,
    const std::string &context,
    PaError err)
{
    std::cerr << context;
    if (err != paNoError)
    {
        std::cerr << ": " << Pa_GetErrorText(err);
    }
    std::cerr << '\n';

    print_requested_config();
    print_selected_device_summary(device_index, device_info);
    maybe_warn_about_generic_alsa_plugin_device(device_info);

    std::cerr << "Available devices for comparison:\n";
    list_all_devices();
}

bool save_capture_csv(
    const std::vector<float> &played_reference,
    const std::vector<float> &captured,
    int frame_count,
    int play_and_listen_frames,
    int output_channels,
    const std::string &csv_path)
{
    constexpr int kTopChannelCount = 5;
    std::vector<float> time_seconds(static_cast<size_t>(frame_count), 0.0f);
    std::vector<float> output_reference(static_cast<size_t>(frame_count), 0.0f);
    std::vector<float> active_output_channel_reference(
        static_cast<size_t>(frame_count),
        -1.0f);
    std::vector<std::vector<float>> channels(
        static_cast<size_t>(CHANNELS),
        std::vector<float>(static_cast<size_t>(frame_count), 0.0f));
    std::vector<double> mean_square_by_channel(
        static_cast<size_t>(CHANNELS),
        0.0);

    for (int frame = 0; frame < frame_count; ++frame)
    {
        time_seconds[static_cast<size_t>(frame)] =
            static_cast<float>(frame) / static_cast<float>(SAMPLE_RATE);
        output_reference[static_cast<size_t>(frame)] =
            frame < static_cast<int>(played_reference.size())
                ? played_reference[static_cast<size_t>(frame)]
                : 0.0f;
        active_output_channel_reference[static_cast<size_t>(frame)] =
            static_cast<float>(
                active_output_channel_for_frame(
                    frame,
                    play_and_listen_frames,
                    output_channels));

        for (int ch = 0; ch < CHANNELS; ++ch)
        {
            const float sample =
                captured[static_cast<size_t>(frame * CHANNELS + ch)];
            channels[static_cast<size_t>(ch)][static_cast<size_t>(frame)] = sample;
            mean_square_by_channel[static_cast<size_t>(ch)] +=
                static_cast<double>(sample) * static_cast<double>(sample);
        }
    }

    std::vector<int> ranked_channels(static_cast<size_t>(CHANNELS), 0);
    for (int ch = 0; ch < CHANNELS; ++ch)
    {
        ranked_channels[static_cast<size_t>(ch)] = ch;
        if (frame_count > 0)
        {
            mean_square_by_channel[static_cast<size_t>(ch)] /=
                static_cast<double>(frame_count);
        }
    }

    std::sort(
        ranked_channels.begin(),
        ranked_channels.end(),
        [&mean_square_by_channel](int lhs, int rhs)
        {
            const double lhs_mean_square =
                mean_square_by_channel[static_cast<size_t>(lhs)];
            const double rhs_mean_square =
                mean_square_by_channel[static_cast<size_t>(rhs)];
            if (lhs_mean_square != rhs_mean_square)
            {
                return lhs_mean_square > rhs_mean_square;
            }
            return lhs < rhs;
        });

    const int selected_channel_count =
        std::min(kTopChannelCount, CHANNELS);

    std::vector<std::string> headers;
    headers.reserve(static_cast<size_t>(selected_channel_count) + 3);
    headers.push_back("time_seconds");
    headers.push_back("output_reference");
    headers.push_back("active_output_channel");

    std::vector<const std::vector<float> *> columns;
    columns.reserve(static_cast<size_t>(selected_channel_count) + 3);
    columns.push_back(&time_seconds);
    columns.push_back(&output_reference);
    columns.push_back(&active_output_channel_reference);

    std::cout << "Plotting top input channels by mean square:";
    for (int rank = 0; rank < selected_channel_count; ++rank)
    {
        const int ch = ranked_channels[static_cast<size_t>(rank)];
        headers.push_back("input_ch" + std::to_string(ch));
        columns.push_back(&channels[static_cast<size_t>(ch)]);
        std::cout << " ch" << ch
                  << "=" << mean_square_by_channel[static_cast<size_t>(ch)];
    }
    std::cout << '\n';

    return save_arrays_to_csv(csv_path, headers, columns);
}

#if !MOCK && defined(PA_USE_ASIO)
PaError maybe_set_asio_sample_rate(
    PaStream *stream,
    const PaDeviceInfo *device_info)
{
    if (!stream || !device_info)
    {
        return paBadStreamPtr;
    }

    const PaHostApiInfo *host_api_info = Pa_GetHostApiInfo(device_info->hostApi);
    if (!host_api_info || host_api_info->type != paASIO)
    {
        return paNoError;
    }

    return PaAsio_SetStreamSampleRate(stream, SAMPLE_RATE);
}
#endif

} // namespace

int main()
{
    if (CHANNELS <= 0)
    {
        std::cerr << "CHANNELS must be positive.\n";
        return 1;
    }
    if (FRAMES_PER_BUFFER <= 0)
    {
        std::cerr << "FRAMES_PER_BUFFER must be positive.\n";
        return 1;
    }

    const int capture_frames = static_cast<int>(std::llround(
        (PLAY_AND_LISTEN_SECONDS + LISTEN_SECONDS) *
        static_cast<double>(SAMPLE_RATE)));
    if (capture_frames <= 0)
    {
        std::cerr
            << "(PLAY_AND_LISTEN_SECONDS + LISTEN_SECONDS) * SAMPLE_RATE "
            << "must be positive.\n";
        return 1;
    }

    const int play_and_listen_frames = static_cast<int>(std::llround(
        PLAY_AND_LISTEN_SECONDS * static_cast<double>(SAMPLE_RATE)));

    print_requested_config();

    const PaError init_error = Pa_Initialize();
    if (init_error != paNoError)
    {
        std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(init_error)
                  << '\n';
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
        std::cerr << "Pa_GetDeviceInfo failed for device " << device_index
                  << '\n';
        Pa_Terminate();
        return 1;
    }

    print_selected_device_summary(device_index, device_info);

    if (device_info->maxInputChannels < CHANNELS)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested channel count exceeds the device input capability",
            paInvalidChannelCount);
        Pa_Terminate();
        return 1;
    }
    if (device_info->maxOutputChannels <= 0)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested playback phase requires at least one output channel",
            paInvalidChannelCount);
        Pa_Terminate();
        return 1;
    }

    const int output_channels =
        std::max(1, std::min(CHANNELS, device_info->maxOutputChannels));

    PaStreamParameters input_parameters{};
    input_parameters.device = device_index;
    input_parameters.channelCount = CHANNELS;
    input_parameters.sampleFormat = SAMPLE_FORMAT;
    input_parameters.suggestedLatency = std::max(
        device_info->defaultLowInputLatency,
        static_cast<double>(FRAMES_PER_BUFFER) /
            static_cast<double>(SAMPLE_RATE));
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = output_channels;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = std::max(
        device_info->defaultLowOutputLatency,
        static_cast<double>(FRAMES_PER_BUFFER) /
            static_cast<double>(SAMPLE_RATE));
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    const PaError format_error =
        Pa_IsFormatSupported(
            &input_parameters,
            &output_parameters,
            SAMPLE_RATE);
    if (format_error != paFormatIsSupported)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested play-and-listen configuration is not supported",
            format_error);
        Pa_Terminate();
        return 1;
    }

    if (output_channels != CHANNELS)
    {
        std::cout << "Using " << output_channels
                  << " output channels for the tone phase.\n";
    }

    std::vector<float> played_reference(
        static_cast<size_t>(capture_frames),
        0.0f);
    std::vector<float> captured(
        static_cast<size_t>(capture_frames * CHANNELS),
        0.0f);

    SoundDeviceQueryCaptureData data{};
    data.captured = captured.data();
    data.played_reference = played_reference.data();
    data.max_frames = capture_frames;
    data.play_and_listen_frames = play_and_listen_frames;
    data.output_channels = output_channels;

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        sound_device_query_callback,
        &data);
    if (open_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_OpenStream failed for the requested play-and-listen configuration",
            open_error);
        Pa_Terminate();
        return 1;
    }

#if !MOCK && defined(PA_USE_ASIO)
    const PaError asio_rate_error =
        maybe_set_asio_sample_rate(stream, device_info);
    if (asio_rate_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "PaAsio_SetStreamSampleRate failed",
            asio_rate_error);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
#endif

    const double seconds_per_output_channel =
        output_channels > 0
            ? PLAY_AND_LISTEN_SECONDS / static_cast<double>(output_channels)
            : 0.0;

    std::cout << "Scanning one output channel at a time with "
              << TONE_FREQUENCY_HZ
              << " Hz for " << PLAY_AND_LISTEN_SECONDS
              << " seconds total (" << seconds_per_output_channel
              << " seconds per output), then listening for "
              << LISTEN_SECONDS << " seconds...\n";

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_StartStream failed for the requested play-and-listen configuration",
            start_error);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    while (true)
    {
        const int active = Pa_IsStreamActive(stream);
        if (active == 1)
        {
            Pa_Sleep(20);
            continue;
        }
        if (active == 0)
        {
            break;
        }

        std::cerr << "Pa_IsStreamActive failed: "
                  << Pa_GetErrorText(active) << '\n';
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    const std::string csv_path =
        std::string(PORTABLE_OUTPUT_DIR) + "/sound_device_query_capture.csv";
    if (!save_capture_csv(
            played_reference,
            captured,
            capture_frames,
            play_and_listen_frames,
            output_channels,
            csv_path))
    {
        std::cerr << "Failed to save capture CSV: " << csv_path << '\n';
        return 1;
    }

    std::cout << "Saved capture CSV: " << csv_path << '\n';
    std::cout << "Captured frames=" << data.frame_index
              << " inputUnderflows=" << data.input_underflow_count
              << " inputOverflows=" << data.input_overflow_count
              << " outputUnderflows=" << data.output_underflow_count
              << " outputOverflows=" << data.output_overflow_count
              << '\n';

    plot_csv_with_portable_script(csv_path);
    return 0;
}
