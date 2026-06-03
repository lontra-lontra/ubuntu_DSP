/*
Build from repo root:
  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=400hz_on_all_channels -DPORTABLE_USE_MOCK=OFF
  cmake --build Portable/build --target portable_400hz_on_all_channels --parallel

If `Portable/build` does not exist yet, came from another machine, or you changed
`PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` line before
the `cmake --build ...` line.

Run:
  ./Portable/build/portable_400hz_on_all_channels

Important config:
  mock/real audio is selected by `-DPORTABLE_USE_MOCK=ON/OFF` in the first command.
  #define CHANNELS 32
  #define DEVICE_NAME ...

To switch to mock audio, rerun the first command with `-DPORTABLE_USE_MOCK=ON`.
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

#define CHANNELS 32

#define DEVICE_NAME "MADIface USB (24285073): Audio (hw:1,0)"

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 128
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 44100
#endif

#ifndef TONE_FREQUENCY_HZ
#define TONE_FREQUENCY_HZ 200.0
#endif

#ifndef TONE_SECONDS
#define TONE_SECONDS 0.5
#endif

#ifndef SILENCE_SECONDS
#define SILENCE_SECONDS 0.5
#endif

#ifndef TONE_AMPLITUDE
#define TONE_AMPLITUDE 0.5
#endif

#if MOCK
#include "portable/mock_devices.h"
#include "portable/mockportaudio.h"
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
              << " TONE_FREQUENCY_HZ=" << TONE_FREQUENCY_HZ
              << " TONE_SECONDS=" << TONE_SECONDS
              << " SILENCE_SECONDS=" << SILENCE_SECONDS
              << " TONE_AMPLITUDE=" << TONE_AMPLITUDE
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

    std::cerr << "Available devices for comparison:\n";
    list_all_devices();
}

void fill_tone_buffer(
    std::vector<float> &buffer,
    int output_channels,
    int tone_frames)
{
    for (int frame = 0; frame < tone_frames; ++frame)
    {
        const double t =
            static_cast<double>(frame) / static_cast<double>(SAMPLE_RATE);
        const float sample = static_cast<float>(
            TONE_AMPLITUDE * std::sin(2.0 * kPi * TONE_FREQUENCY_HZ * t));

        for (int ch = 0; ch < output_channels; ++ch)
        {
            buffer[static_cast<size_t>(frame * output_channels + ch)] = sample;
        }
    }
}

} // namespace

int main()
{
    const int tone_frames = std::max(
        1,
        static_cast<int>(std::llround(
            TONE_SECONDS * static_cast<double>(SAMPLE_RATE))));
    const int silence_frames = std::max(
        0,
        static_cast<int>(std::llround(
            SILENCE_SECONDS * static_cast<double>(SAMPLE_RATE))));

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

    if (device_info->maxOutputChannels <= 0)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested playback requires at least one output channel",
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
        static_cast<size_t>(silence_frames * output_channels),
        0.0f);
    fill_tone_buffer(tone_buffer, output_channels, tone_frames);

    std::signal(SIGINT, handle_interrupt_signal);
    std::cout << "Playing " << TONE_FREQUENCY_HZ
              << " Hz on all " << output_channels
              << " output channels for " << TONE_SECONDS
              << " s, then silence for " << SILENCE_SECONDS
              << " s. Press Ctrl+C to stop.\n";

    while (g_keep_running.load())
    {
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

        if (silence_frames > 0)
        {
            const PaError write_silence_error =
                Pa_WriteStream(stream, silence_buffer.data(), silence_frames);
            if (write_silence_error != paNoError)
            {
                show_failure_context(
                    device_index,
                    device_info,
                    "Pa_WriteStream failed while playing the silence",
                    write_silence_error);
                Pa_AbortStream(stream);
                Pa_CloseStream(stream);
                Pa_Terminate();
                return 1;
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

    std::cout << "Stopped playback.\n";
    return 0;
}
