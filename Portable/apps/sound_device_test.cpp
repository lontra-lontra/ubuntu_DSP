/*
Build from repo root:

  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=sound_device_test -DPORTABLE_USE_MOCK=OFF
  cmake --build Portable/build --target portable_sound_device_test --parallel

If `Portable/build` does not exist yet, came from another machine, or you changed
`PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` line before
the `cmake --build ...` line.

Run:
  ./Portable/build/portable_sound_device_test

To try mock audio instead, rerun the first command with `-DPORTABLE_USE_MOCK=ON`.

Important config:
  local device config in this file:
  #define DEVICE_NAME ...
  #define CHANNELS ...
  #define SAMPLE_RATE ...
  #define FRAMES_PER_BUFFER ...
  #define SAMPLE_FORMAT ...

This app plays a 1000 Hz tone for 1 second on one output channel at a time,
with 1 second of silence between channels, and repeats until you stop it.
*/

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
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

#ifndef TONE_SECONDS
#define TONE_SECONDS 1.0
#endif

#ifndef PAUSE_SECONDS
#define PAUSE_SECONDS 1.0
#endif

#ifndef TONE_FREQUENCY_HZ
#define TONE_FREQUENCY_HZ 1000.0
#endif

#ifndef TONE_AMPLITUDE
#define TONE_AMPLITUDE 0.8
#endif

#if MOCK
#include "portable/mockportaudio.h"
#include "portable/mock_devices.h"
#else
#include <portaudio.h>
#endif

#include "portable/audio_helpers.h"

namespace
{

constexpr double kPi = 3.14159265358979323846;
std::atomic<bool> g_keep_running{true};

void handle_interrupt_signal(int)
{
    g_keep_running.store(false);
}

void print_requested_config()
{
    std::cout << "Requested config:"
              << " DEVICE_NAME=" << DEVICE_NAME
              << " CHANNELS=" << CHANNELS
              << " SAMPLE_RATE=" << SAMPLE_RATE
              << " FRAMES_PER_BUFFER=" << FRAMES_PER_BUFFER
              << " TONE_SECONDS=" << TONE_SECONDS
              << " PAUSE_SECONDS=" << PAUSE_SECONDS
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

    const std::vector<double> output_rates =
        get_possible_sample_rates(device_index, false);
    if (!output_rates.empty())
    {
        std::cout << "Supported output rates:";
        for (double rate : output_rates)
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
        << "output tests.\n"
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

void fill_tone_buffer(
    std::vector<float> &buffer,
    int output_channels,
    int active_output_channel,
    int tone_frames)
{
    std::fill(buffer.begin(), buffer.end(), 0.0f);

    for (int frame = 0; frame < tone_frames; ++frame)
    {
        const double t =
            static_cast<double>(frame) / static_cast<double>(SAMPLE_RATE);
        const float sample = static_cast<float>(
            TONE_AMPLITUDE * std::sin(2.0 * kPi * TONE_FREQUENCY_HZ * t));
        buffer[static_cast<size_t>(frame * output_channels + active_output_channel)] =
            sample;
    }
}

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

    const int tone_frames = static_cast<int>(std::llround(
        TONE_SECONDS * static_cast<double>(SAMPLE_RATE)));
    if (tone_frames <= 0)
    {
        std::cerr << "TONE_SECONDS * SAMPLE_RATE must be positive.\n";
        return 1;
    }

    const int pause_frames = std::max(
        0,
        static_cast<int>(std::llround(
            PAUSE_SECONDS * static_cast<double>(SAMPLE_RATE))));

    print_requested_config();

    const PaError init_error = Pa_Initialize();
    if (init_error != paNoError)
    {
        std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(init_error)
                  << '\n';
        return 1;
    }

    const int device_index = select_device_by_name(DEVICE_NAME, false);
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
    maybe_warn_about_generic_alsa_plugin_device(device_info);

    if (device_info->maxOutputChannels <= 0)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested playback test requires at least one output channel",
            paInvalidChannelCount);
        Pa_Terminate();
        return 1;
    }

    const int output_channels =
        std::max(1, std::min(CHANNELS, device_info->maxOutputChannels));

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
        Pa_IsFormatSupported(nullptr, &output_parameters, SAMPLE_RATE);
    if (format_error != paFormatIsSupported)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested playback configuration is not supported",
            format_error);
        Pa_Terminate();
        return 1;
    }

    if (output_channels != CHANNELS)
    {
        std::cout << "Using " << output_channels
                  << " output channels for this test.\n";
    }

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        nullptr,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        nullptr,
        nullptr);
    if (open_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_OpenStream failed for the requested playback configuration",
            open_error);
        Pa_Terminate();
        return 1;
    }

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_StartStream failed for the requested playback configuration",
            start_error);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    std::vector<float> tone_buffer(
        static_cast<size_t>(tone_frames * output_channels),
        0.0f);
    std::vector<float> silence_buffer(
        static_cast<size_t>(pause_frames * output_channels),
        0.0f);

    std::signal(SIGINT, handle_interrupt_signal);
    std::cout << "Playing a " << TONE_FREQUENCY_HZ
              << " Hz tone for " << TONE_SECONDS
              << " seconds on each output channel with "
              << PAUSE_SECONDS
              << " seconds between channels. Press Ctrl+C to stop.\n";

    while (g_keep_running.load())
    {
        for (int active_output_channel = 0;
             active_output_channel < output_channels && g_keep_running.load();
             ++active_output_channel)
        {
            std::cout << "Playing output channel "
                      << active_output_channel << " of "
                      << output_channels - 1 << '\n';

            fill_tone_buffer(
                tone_buffer,
                output_channels,
                active_output_channel,
                tone_frames);

            const PaError write_tone_error =
                Pa_WriteStream(stream, tone_buffer.data(), tone_frames);
            if (write_tone_error != paNoError)
            {
                show_failure_context(
                    device_index,
                    device_info,
                    "Pa_WriteStream failed while playing the tone",
                    write_tone_error);
                Pa_AbortStream(stream);
                Pa_CloseStream(stream);
                Pa_Terminate();
                return 1;
            }

            if (pause_frames > 0)
            {
                const PaError write_pause_error =
                    Pa_WriteStream(stream, silence_buffer.data(), pause_frames);
                if (write_pause_error != paNoError)
                {
                    show_failure_context(
                        device_index,
                        device_info,
                        "Pa_WriteStream failed while playing the pause",
                        write_pause_error);
                    Pa_AbortStream(stream);
                    Pa_CloseStream(stream);
                    Pa_Terminate();
                    return 1;
                }
            }
        }
    }

    const PaError stop_error = Pa_StopStream(stream);
    if (stop_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_StopStream failed after playback",
            stop_error);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "Stopped output sweep.\n";
    return 0;
}
