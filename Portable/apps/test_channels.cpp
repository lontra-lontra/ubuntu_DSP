/*
Build from repo root:
  direct machine:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=test_channels -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=OFF -DPORTABLE_DEVICE_NAME="MADIface USB (24285073): Audio (hw:2,0)" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32
    cmake --build Portable/build --target portable_test_channels --parallel
    ./Portable/build/portable_test_channels

  JACK:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=test_channels -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=ON -DPORTABLE_DEVICE_NAME="system" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32
    cmake --build Portable/build --target portable_test_channels --parallel
    ./Portable/build/portable_test_channels

This app records the full 32-channel input stream while it plays:
  - 0.2 s of 300 Hz at amplitude 0.8 on output channel 0
  - 0.2 s of silence
  - then the same pattern on output channel 1
  - and so on through output channel 31

It saves one compact CSV containing a time axis plus all 32 recorded input
channels summarized as short-window peak traces, then attempts to plot it.

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
//#define DEVICE_NAME "MADIface USB (24285073): Audio (hw:2,0)"
#define DEVICE_NAME "system"
#endif

#ifndef CHANNELS
#define CHANNELS 32
#endif

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 1024
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 44100
#endif

#ifndef TONE_SECONDS
#define TONE_SECONDS 0.2
#endif

#ifndef SILENCE_SECONDS
#define SILENCE_SECONDS 0.4
#endif

#ifndef TONE_FREQUENCY_HZ
#define TONE_FREQUENCY_HZ 300.0
#endif

#ifndef TONE_AMPLITUDE
#define TONE_AMPLITUDE 0.1
#endif

#ifndef PLOT_WINDOW_SAMPLES
#define PLOT_WINDOW_SAMPLES 32
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#ifndef PORTABLE_PLOT_SCRIPT
#define PORTABLE_PLOT_SCRIPT "Portable/scripts/plot.py"
#endif

#if MOCK
#include "portable/mock_devices.h"
#include "portable/mockportaudio.h"
#else
#include <portaudio.h>
#endif

#include "portable/audio_helpers.h"
#include "portable/csv_utils.h"
#include "portable/plotting.h"

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct TestChannelsCaptureData
{
    float *recorded = nullptr;
    int frame_index = 0;
    int max_frames = 0;
    int tone_frames = 0;
    int segment_frames = 0;
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
    int callback_warning_count = 0;
    int null_input_buffer_count = 0;
    std::string first_runtime_warning;
};

std::string describe_status_flags(PaStreamCallbackFlags statusFlags)
{
    std::string description;

    const auto append_flag = [&description](const char *label)
    {
        if (!description.empty())
        {
            description += ", ";
        }
        description += label;
    };

    if ((statusFlags & paInputUnderflow) != 0)
    {
        append_flag("input underflow");
    }
    if ((statusFlags & paInputOverflow) != 0)
    {
        append_flag("input overflow");
    }
    if ((statusFlags & paOutputUnderflow) != 0)
    {
        append_flag("output underflow");
    }
    if ((statusFlags & paOutputOverflow) != 0)
    {
        append_flag("output overflow");
    }
    if ((statusFlags & paPrimingOutput) != 0)
    {
        append_flag("output priming");
    }

    if (description.empty())
    {
        description = "unknown callback status flag issue";
    }

    return description;
}

void print_requested_config()
{
    std::cout << "Requested config:"
              << " DEVICE_NAME=" << DEVICE_NAME
              << " CHANNELS=" << CHANNELS
              << " SAMPLE_RATE=" << SAMPLE_RATE
              << " FRAMES_PER_BUFFER=" << FRAMES_PER_BUFFER
              << " TONE_SECONDS=" << TONE_SECONDS
              << " SILENCE_SECONDS=" << SILENCE_SECONDS
              << " TONE_FREQUENCY_HZ=" << TONE_FREQUENCY_HZ
              << " TONE_AMPLITUDE=" << TONE_AMPLITUDE
              << " PLOT_WINDOW_SAMPLES=" << PLOT_WINDOW_SAMPLES
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

int active_output_channel_for_frame(
    int absolute_frame,
    int segment_frames)
{
    if (absolute_frame < 0 || segment_frames <= 0)
    {
        return -1;
    }

    const int active_output_channel = absolute_frame / segment_frames;
    if (active_output_channel < 0 || active_output_channel >= CHANNELS)
    {
        return -1;
    }

    return active_output_channel;
}

float output_sample_for_frame(
    int absolute_frame,
    int output_channel,
    int tone_frames,
    int segment_frames)
{
    const int active_output_channel =
        active_output_channel_for_frame(absolute_frame, segment_frames);
    if (active_output_channel < 0 || output_channel != active_output_channel)
    {
        return 0.0f;
    }

    const int frame_in_segment = absolute_frame % segment_frames;
    if (frame_in_segment < 0 || frame_in_segment >= tone_frames)
    {
        return 0.0f;
    }

    const double t =
        static_cast<double>(frame_in_segment) /
        static_cast<double>(SAMPLE_RATE);
    return static_cast<float>(
        TONE_AMPLITUDE * std::sin(2.0 * kPi * TONE_FREQUENCY_HZ * t));
}

int test_channels_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    TestChannelsCaptureData *data =
        static_cast<TestChannelsCaptureData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data || !data->recorded || !output)
    {
        std::cerr << "FATAL: callback received invalid state"
                  << " data=" << (data ? "ok" : "null")
                  << " recorded=" << ((data && data->recorded) ? "ok" : "null")
                  << " output=" << (output ? "ok" : "null")
                  << '\n';
        return paAbort;
    }

    if ((statusFlags & paPrimingOutput) != 0)
    {
        std::cerr << "WARNING: PortAudio reported output priming in callback"
                  << " at frame_index=" << data->frame_index
                  << " framesPerBuffer=" << framesPerBuffer
                  << '\n';
        data->callback_warning_count++;
        if (data->first_runtime_warning.empty())
        {
            data->first_runtime_warning = "PortAudio reported output priming.";
        }
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

    if (statusFlags != 0)
    {
        const std::string flag_description = describe_status_flags(statusFlags);
        std::cerr << "WARNING: PortAudio callback reported " << flag_description
                  << " at frame_index=" << data->frame_index
                  << " framesPerBuffer=" << framesPerBuffer
                  << '\n';
        data->callback_warning_count++;
        if (data->first_runtime_warning.empty())
        {
            data->first_runtime_warning =
                "PortAudio callback status flags: " + flag_description;
        }
    }

    if (!input)
    {
        std::cerr << "WARNING: callback received null input buffer at frame_index="
                  << data->frame_index
                  << "; recording zeros for this callback."
                  << '\n';
        data->null_input_buffer_count++;
        if (data->first_runtime_warning.empty())
        {
            data->first_runtime_warning =
                "PortAudio callback received null input buffer.";
        }
    }

    const int frames_left = data->max_frames - data->frame_index;
    const int frames_to_process =
        std::max(0, std::min(static_cast<int>(framesPerBuffer), frames_left));

    if (frames_to_process <= 0)
    {
        std::cerr << "WARNING: callback reached non-positive frames_to_process="
                  << frames_to_process
                  << " with frame_index=" << data->frame_index
                  << " max_frames=" << data->max_frames
                  << ". Completing callback early."
                  << '\n';
        data->callback_warning_count++;
        if (data->first_runtime_warning.empty())
        {
            data->first_runtime_warning =
                "Callback reached non-positive frames_to_process before normal completion.";
        }
        return paComplete;
    }

    for (int i = 0; i < frames_to_process; ++i)
    {
        const int absolute_frame = data->frame_index + i;
        for (int channel = 0; channel < CHANNELS; ++channel)
        {
            output[static_cast<size_t>(i * CHANNELS + channel)] =
                output_sample_for_frame(
                    absolute_frame,
                    channel,
                    data->tone_frames,
                    data->segment_frames);
            data->recorded[static_cast<size_t>(absolute_frame * CHANNELS + channel)] =
                input ? input[static_cast<size_t>(i * CHANNELS + channel)] : 0.0f;
        }
    }

    for (int i = frames_to_process; i < static_cast<int>(framesPerBuffer); ++i)
    {
        for (int channel = 0; channel < CHANNELS; ++channel)
        {
            output[static_cast<size_t>(i * CHANNELS + channel)] = 0.0f;
        }
    }

    data->frame_index += frames_to_process;
    if (data->frame_index >= data->max_frames)
    {
        return paComplete;
    }

    return paContinue;
}

void print_segment_summaries(
    const std::vector<float> &recorded,
    int total_frames,
    int tone_frames,
    int segment_frames)
{
    std::cout << "Per-output strongest recorded input during each tone:\n";

    for (int output_channel = 0; output_channel < CHANNELS; ++output_channel)
    {
        const int tone_start_frame = output_channel * segment_frames;
        const int tone_end_frame =
            std::min(tone_start_frame + tone_frames, total_frames);

        float strongest_peak = 0.0f;
        int strongest_input_channel = -1;

        for (int input_channel = 0; input_channel < CHANNELS; ++input_channel)
        {
            float input_peak = 0.0f;
            for (int frame = tone_start_frame; frame < tone_end_frame; ++frame)
            {
                input_peak = std::max(
                    input_peak,
                    std::fabs(
                        recorded[static_cast<size_t>(frame * CHANNELS + input_channel)]));
            }

            if (input_peak > strongest_peak)
            {
                strongest_peak = input_peak;
                strongest_input_channel = input_channel;
            }
        }

        std::cout << "  out " << output_channel
                  << " -> strongest input " << strongest_input_channel
                  << " peak=" << strongest_peak
                  << '\n';
    }
}

bool save_compact_input_csv(
    const std::vector<float> &recorded,
    int total_frames,
    int tone_frames,
    int segment_frames,
    const std::string &csv_path)
{
    if (total_frames <= 0 ||
        tone_frames <= 0 ||
        segment_frames <= 0 ||
        static_cast<int>(recorded.size()) != total_frames * CHANNELS)
    {
        return false;
    }

    const int window_samples = std::max(1, PLOT_WINDOW_SAMPLES);
    const int window_count =
        (total_frames + window_samples - 1) / window_samples;

    std::vector<float> time_axis(static_cast<size_t>(window_count), 0.0f);
    for (int window_index = 0; window_index < window_count; ++window_index)
    {
        const int start_frame = window_index * window_samples;
        time_axis[static_cast<size_t>(window_index)] =
            static_cast<float>(start_frame) / static_cast<float>(SAMPLE_RATE);
    }

    std::vector<std::vector<float>> storage;
    storage.reserve(static_cast<size_t>(CHANNELS + 1));

    std::vector<std::string> headers;
    headers.reserve(2 + CHANNELS);
    headers.push_back("time (s)");

    std::vector<const std::vector<float> *> columns;
    columns.reserve(2 + CHANNELS);
    columns.push_back(&time_axis);

    storage.push_back(std::vector<float>(static_cast<size_t>(window_count), 0.0f));
    std::vector<float> &output_max_trace = storage.back();

    for (int window_index = 0; window_index < window_count; ++window_index)
    {
        const int start_frame = window_index * window_samples;
        const int end_frame = std::min(start_frame + window_samples, total_frames);
        const int frame_count = std::max(1, end_frame - start_frame);
        const int representative_frame =
            start_frame + (frame_count - 1) / 2;
        const int active_output_channel =
            active_output_channel_for_frame(representative_frame, segment_frames);

        float sampled_output = 0.0f;
        if (active_output_channel >= 0)
        {
            sampled_output = output_sample_for_frame(
                representative_frame,
                active_output_channel,
                tone_frames,
                segment_frames);
        }

        output_max_trace[static_cast<size_t>(window_index)] = sampled_output;
    }

    headers.push_back("output_max");
    columns.push_back(&output_max_trace);

    for (int input_channel = 0; input_channel < CHANNELS; ++input_channel)
    {
        storage.push_back(std::vector<float>(static_cast<size_t>(window_count), 0.0f));
        std::vector<float> &input_trace = storage.back();

        for (int window_index = 0; window_index < window_count; ++window_index)
        {
            const int start_frame = window_index * window_samples;
            const int end_frame = std::min(start_frame + window_samples, total_frames);
            float peak = 0.0f;

            for (int frame = start_frame; frame < end_frame; ++frame)
            {
                peak = std::max(
                    peak,
                    std::fabs(
                        recorded[static_cast<size_t>(frame * CHANNELS + input_channel)]));
            }

            input_trace[static_cast<size_t>(window_index)] = peak;
        }

        headers.push_back("recorded_ch_" + std::to_string(input_channel));
        columns.push_back(&input_trace);
    }

    return save_arrays_to_csv(csv_path, headers, columns);
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
    if (PLOT_WINDOW_SAMPLES <= 0)
    {
        std::cerr << "PLOT_WINDOW_SAMPLES must be positive.\n";
        return 1;
    }

    const int tone_frames = std::max(
        1,
        static_cast<int>(std::llround(
            TONE_SECONDS * static_cast<double>(SAMPLE_RATE))));
    const int silence_frames = std::max(
        0,
        static_cast<int>(std::llround(
            SILENCE_SECONDS * static_cast<double>(SAMPLE_RATE))));
    const int segment_frames = tone_frames + silence_frames;
    const int total_frames = CHANNELS * segment_frames;

    if (segment_frames <= 0 || total_frames <= 0)
    {
        std::cerr << "Test duration must be positive.\n";
        return 1;
    }

    print_requested_config();

    const PaError init_error = Pa_Initialize();
    if (init_error != paNoError)
    {
        std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(init_error)
                  << '\n';
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
        std::cerr << "Pa_GetDeviceInfo failed for device " << device_index
                  << '\n';
        Pa_Terminate();
        return 1;
    }

    print_selected_device_summary(device_index, device_info);

    if (device_info->maxInputChannels < CHANNELS ||
        device_info->maxOutputChannels < CHANNELS)
    {
        show_failure_context(
            device_index,
            device_info,
            "Selected device does not expose the requested full-duplex channel count",
            paInvalidChannelCount);
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

    std::vector<float> recorded(
        static_cast<size_t>(total_frames * CHANNELS),
        0.0f);

    TestChannelsCaptureData data{};
    data.recorded = recorded.data();
    data.frame_index = 0;
    data.max_frames = total_frames;
    data.tone_frames = tone_frames;
    data.segment_frames = segment_frames;

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        test_channels_callback,
        &data);
    if (open_error != paNoError)
    {
        std::cerr << "Pa_OpenStream failed: " << Pa_GetErrorText(open_error)
                  << '\n';
        Pa_Terminate();
        return 1;
    }

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        std::cerr << "Pa_StartStream failed: " << Pa_GetErrorText(start_error)
                  << '\n';
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    while (true)
    {
        const PaError active_state = Pa_IsStreamActive(stream);
        if (active_state < 0)
        {
            std::cerr << "FATAL: Pa_IsStreamActive failed: "
                      << Pa_GetErrorText(active_state)
                      << '\n';
            Pa_AbortStream(stream);
            Pa_CloseStream(stream);
            Pa_Terminate();
            return 1;
        }
        if (active_state == 0)
        {
            break;
        }
        Pa_Sleep(50);
    }

    const PaError stop_error = Pa_StopStream(stream);
    if (stop_error != paNoError)
    {
        std::cerr << "FATAL: Pa_StopStream failed: "
                  << Pa_GetErrorText(stop_error)
                  << '\n';
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    const PaError close_error = Pa_CloseStream(stream);
    if (close_error != paNoError)
    {
        std::cerr << "FATAL: Pa_CloseStream failed: "
                  << Pa_GetErrorText(close_error)
                  << '\n';
        Pa_Terminate();
        return 1;
    }
    Pa_Terminate();

    std::cout << "Status flags summary:"
              << " input_underflows=" << data.input_underflow_count
              << " input_overflows=" << data.input_overflow_count
              << " output_underflows=" << data.output_underflow_count
              << " output_overflows=" << data.output_overflow_count
              << " callback_warnings=" << data.callback_warning_count
              << " null_input_buffers=" << data.null_input_buffer_count
              << '\n';

    if (!data.first_runtime_warning.empty())
    {
        std::cout << "First runtime warning: "
                  << data.first_runtime_warning
                  << '\n';
    }

    print_segment_summaries(recorded, total_frames, tone_frames, segment_frames);

    const std::string csv_path =
        std::string(PORTABLE_OUTPUT_DIR) +
        "/test_channels_ch_" + std::to_string(CHANNELS) +
        "_input_peaks.csv";
    if (!save_compact_input_csv(
            recorded,
            total_frames,
            tone_frames,
            segment_frames,
            csv_path))
    {
        std::cerr << "Failed to save compact input CSV: " << csv_path << '\n';
        return 1;
    }

    std::cout << "Saved compact input CSV: " << csv_path << '\n';
    plot_csv_with_portable_script(csv_path);
    return 0;
}
