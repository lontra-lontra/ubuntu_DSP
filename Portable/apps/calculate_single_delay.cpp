/*
Build from repo root:
  direct machine:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=calculate_single_delay -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=OFF -DPORTABLE_DEVICE_NAME="MADIface USB (24285073): Audio (hw:2,0)" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32 -DPORTABLE_INPUT_PORT=5 -DPORTABLE_OUTPUT_PORT=5 -DPORTABLE_CORRELATION_THRESHOLD=0.95
    cmake --build Portable/build --target portable_calculate_single_delay --parallel
    ./Portable/build/portable_calculate_single_delay

  JACK:
    cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=calculate_single_delay -DPORTABLE_USE_MOCK=OFF -DPORTABLE_ENABLE_JACK=ON -DPORTABLE_DEVICE_NAME="system" -DPORTABLE_SAMPLE_RATE=44100 -DPORTABLE_FRAMES_PER_BUFFER=32 -DPORTABLE_INPUT_PORT=5 -DPORTABLE_OUTPUT_PORT=5 -DPORTABLE_CORRELATION_THRESHOLD=0.95
    cmake --build Portable/build --target portable_calculate_single_delay --parallel
    ./Portable/build/portable_calculate_single_delay

Build-time config:
  `DEVICE_NAME`, `SAMPLE_RATE`, `FRAMES_PER_BUFFER`, `INPUT_PORT_1BASED`,
  `OUTPUT_PORT_1BASED`, and `CORRELATION_THRESHOLD` come from the CMake command above.

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
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
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

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 44100
#endif

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 8
#endif

#ifndef CHIRP_START_FREQUENCY_HZ
#define CHIRP_START_FREQUENCY_HZ 1000.0
#endif

#ifndef CHIRP_END_FREQUENCY_HZ
#define CHIRP_END_FREQUENCY_HZ 18000.0
#endif

#ifndef CHIRP_AMPLITUDE
#define CHIRP_AMPLITUDE 0.8
#endif

#ifndef CHIRP_SECONDS
#define CHIRP_SECONDS 0.005
#endif

#ifndef LEADING_SILENCE_SECONDS
#define LEADING_SILENCE_SECONDS 0.05
#endif

#ifndef TRAILING_CAPTURE_SECONDS
#define TRAILING_CAPTURE_SECONDS 0.05
#endif

#ifndef INTERNAL_PREROLL_SECONDS
#define INTERNAL_PREROLL_SECONDS 0.05
#endif

#ifndef INPUT_PORT_1BASED
#define INPUT_PORT_1BASED 5
#endif

#ifndef OUTPUT_PORT_1BASED
#define OUTPUT_PORT_1BASED 5
#endif

#ifndef CORRELATION_THRESHOLD
#define CORRELATION_THRESHOLD 0.95
#endif

#ifndef MINIMUM_INPUT_PEAK
#define MINIMUM_INPUT_PEAK 1.0e-6
#endif

#ifndef DEVICE_NAME
#if MOCK
#define DEVICE_NAME "default"
#else
#define DEVICE_NAME "MADIface USB (24285073): Audio (hw:2,0)"
#endif
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#if MOCK
#include "portable/mockportaudio.h"
#include "portable/mock_devices.h"
#else
#include <portaudio.h>
#endif

#include "portable/audio_helpers.h"
#include "portable/csv_utils.h"
#include "portable/plotting.h"

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kInteractionSeconds =
    LEADING_SILENCE_SECONDS + CHIRP_SECONDS + TRAILING_CAPTURE_SECONDS;

std::atomic<bool> g_keep_running{true};

struct Options
{
    int input_port_1based = INPUT_PORT_1BASED;
    int output_port_1based = OUTPUT_PORT_1BASED;
    double threshold = CORRELATION_THRESHOLD;
};

struct CaptureData
{
    const float *reference_output = nullptr;
    float *captured_input = nullptr;
    int input_channel_index = 0;
    int output_channel_index = 0;
    int stream_channel_count = 0;
    int frame_index = 0;
    int max_frames = 0;
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
};

struct CorrelationResult
{
    int delay_samples = -1;
    double delay_ms = -1.0;
    double score = std::numeric_limits<double>::quiet_NaN();
    double absolute_score = std::numeric_limits<double>::quiet_NaN();
};

struct InteractionCapture
{
    std::vector<float> full_reference_output;
    std::vector<float> full_captured_input;
    std::vector<float> interaction_reference_output;
    std::vector<float> interaction_captured_input;
    std::vector<float> aligned_input;
    CorrelationResult correlation;
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
};

void handle_interrupt_signal(int)
{
    g_keep_running.store(false);
}

double quiet_nan()
{
    return std::numeric_limits<double>::quiet_NaN();
}

bool is_finite_number(double value)
{
    return std::isfinite(value);
}

double sample_offset_to_milliseconds(double sample_count)
{
    return 1000.0 * sample_count / static_cast<double>(SAMPLE_RATE);
}

void print_requested_config(const Options &options)
{
    std::cout << "Requested config:"
              << " DEVICE_NAME=" << DEVICE_NAME
              << " input_port_1based=" << options.input_port_1based
              << " output_port_1based=" << options.output_port_1based
              << " threshold=" << options.threshold
              << " SAMPLE_RATE=" << SAMPLE_RATE
              << " FRAMES_PER_BUFFER=" << FRAMES_PER_BUFFER
              << " CHIRP_START_FREQUENCY_HZ=" << CHIRP_START_FREQUENCY_HZ
              << " CHIRP_END_FREQUENCY_HZ=" << CHIRP_END_FREQUENCY_HZ
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

int select_measurement_device()
{
#if MOCK
    return select_device_by_name(DEVICE_NAME, true);
#else
    return select_device_by_name(DEVICE_NAME, true);
#endif
}

float chirp_sample_at_time(double time_seconds)
{
    if (time_seconds < 0.0 || time_seconds >= kInteractionSeconds)
    {
        return 0.0f;
    }

    if (time_seconds < LEADING_SILENCE_SECONDS)
    {
        return 0.0f;
    }

    if (time_seconds >= LEADING_SILENCE_SECONDS + CHIRP_SECONDS)
    {
        return 0.0f;
    }

    const double chirp_time = time_seconds - LEADING_SILENCE_SECONDS;
    const double chirp_progress =
        CHIRP_SECONDS > 0.0 ? chirp_time / CHIRP_SECONDS : 0.0;
    const double chirp_slope =
        CHIRP_SECONDS > 0.0
            ? (CHIRP_END_FREQUENCY_HZ - CHIRP_START_FREQUENCY_HZ) /
                  CHIRP_SECONDS
            : 0.0;
    const double phase =
        2.0 * kPi *
        ((CHIRP_START_FREQUENCY_HZ * chirp_time) +
         (0.5 * chirp_slope * chirp_time * chirp_time));
    const double window =
        0.5 - (0.5 * std::cos(2.0 * kPi * chirp_progress));

    return static_cast<float>(CHIRP_AMPLITUDE * window * std::sin(phase));
}

int capture_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    CaptureData *data = static_cast<CaptureData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data || !data->reference_output || !data->captured_input || !output)
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
        const float output_sample = data->reference_output[absolute_frame];
        data->captured_input[absolute_frame] =
            input ? input[i * data->stream_channel_count + data->input_channel_index]
                  : 0.0f;

        for (int channel = 0; channel < data->stream_channel_count; ++channel)
        {
            output[i * data->stream_channel_count + channel] =
                channel == data->output_channel_index ? output_sample : 0.0f;
        }
    }

    for (int i = frames_to_process; i < static_cast<int>(framesPerBuffer); ++i)
    {
        for (int channel = 0; channel < data->stream_channel_count; ++channel)
        {
            output[i * data->stream_channel_count + channel] = 0.0f;
        }
    }

    data->frame_index += frames_to_process;
    return data->frame_index >= data->max_frames ? paComplete : paContinue;
}

std::vector<float> build_reference_output_with_preroll(
    int total_frames,
    int pre_roll_frames,
    int interaction_frames)
{
    std::vector<float> reference_output(
        static_cast<size_t>(total_frames),
        0.0f);

    for (int frame = 0; frame < interaction_frames; ++frame)
    {
        const int absolute_frame = pre_roll_frames + frame;
        if (absolute_frame >= total_frames)
        {
            break;
        }

        const double time_seconds =
            static_cast<double>(frame) / static_cast<double>(SAMPLE_RATE);
        reference_output[static_cast<size_t>(absolute_frame)] =
            chirp_sample_at_time(time_seconds);
    }

    return reference_output;
}

std::vector<float> slice_signal(
    const std::vector<float> &signal,
    int start_frame,
    int frame_count)
{
    const int clamped_start =
        std::max(0, std::min(start_frame, static_cast<int>(signal.size())));
    const int clamped_end =
        std::max(
            clamped_start,
            std::min(
                clamped_start + frame_count,
                static_cast<int>(signal.size())));

    return std::vector<float>(
        signal.begin() + clamped_start,
        signal.begin() + clamped_end);
}

double max_absolute_value(const std::vector<float> &signal)
{
    double peak = 0.0;
    for (float sample : signal)
    {
        peak = std::max(peak, std::fabs(static_cast<double>(sample)));
    }
    return peak;
}

std::vector<float> shift_signal_left_by_samples(
    const std::vector<float> &signal,
    double shift_samples)
{
    std::vector<float> shifted(signal.size(), 0.0f);
    if (signal.empty() || !is_finite_number(shift_samples))
    {
        return shifted;
    }

    for (size_t destination_index = 0; destination_index < signal.size();
         ++destination_index)
    {
        const double source_index =
            static_cast<double>(destination_index) + shift_samples;
        if (source_index < 0.0 ||
            source_index > static_cast<double>(signal.size() - 1))
        {
            continue;
        }

        const size_t lower_index =
            static_cast<size_t>(std::floor(source_index));
        const size_t upper_index =
            std::min(lower_index + 1, signal.size() - 1);
        const double fraction =
            source_index - static_cast<double>(lower_index);
        const double lower_value =
            static_cast<double>(signal[lower_index]);
        const double upper_value =
            static_cast<double>(signal[upper_index]);

        shifted[destination_index] = static_cast<float>(
            ((1.0 - fraction) * lower_value) +
            (fraction * upper_value));
    }

    return shifted;
}

std::vector<double> build_squared_prefix_sums(const std::vector<float> &signal)
{
    std::vector<double> prefix_sums(signal.size() + 1, 0.0);
    for (size_t i = 0; i < signal.size(); ++i)
    {
        const double value = static_cast<double>(signal[i]);
        prefix_sums[i + 1] = prefix_sums[i] + (value * value);
    }
    return prefix_sums;
}

int chirp_start_frame_in_interaction()
{
    return static_cast<int>(std::llround(
        LEADING_SILENCE_SECONDS * static_cast<double>(SAMPLE_RATE)));
}

CorrelationResult infer_correlation_delay(
    const std::vector<float> &reference_output,
    const std::vector<float> &captured_input,
    int search_start_frame)
{
    CorrelationResult result;

    const int clamped_start = std::max(
        0,
        std::min(
            search_start_frame,
            std::min(
                static_cast<int>(reference_output.size()),
                static_cast<int>(captured_input.size()))));
    const int chirp_frame_count = std::max(
        1,
        static_cast<int>(std::llround(
            CHIRP_SECONDS * static_cast<double>(SAMPLE_RATE))));
    if (clamped_start >= static_cast<int>(reference_output.size()) ||
        clamped_start >= static_cast<int>(captured_input.size()))
    {
        return result;
    }

    const std::vector<float> reference_template = slice_signal(
        reference_output,
        clamped_start,
        chirp_frame_count);
    const std::vector<float> captured_tail = slice_signal(
        captured_input,
        clamped_start,
        static_cast<int>(captured_input.size()) - clamped_start);
    if (static_cast<int>(reference_template.size()) != chirp_frame_count ||
        captured_tail.size() < reference_template.size())
    {
        return result;
    }

    const double reference_peak = max_absolute_value(reference_template);
    const double captured_peak = max_absolute_value(captured_tail);
    if (reference_peak < MINIMUM_INPUT_PEAK ||
        captured_peak < MINIMUM_INPUT_PEAK)
    {
        return result;
    }

    double reference_energy = 0.0;
    for (float sample : reference_template)
    {
        const double value = static_cast<double>(sample);
        reference_energy += value * value;
    }
    if (reference_energy <= 0.0)
    {
        return result;
    }

    const std::vector<double> captured_squared_prefix_sums =
        build_squared_prefix_sums(captured_tail);
    const int max_lag =
        static_cast<int>(captured_tail.size()) - chirp_frame_count;

    double best_absolute_score = -1.0;
    double best_score = quiet_nan();
    int best_lag = -1;

    for (int lag = 0; lag <= max_lag; ++lag)
    {
        double dot_product = 0.0;
        for (int frame = 0; frame < chirp_frame_count; ++frame)
        {
            dot_product +=
                static_cast<double>(reference_template[static_cast<size_t>(frame)]) *
                static_cast<double>(captured_tail[static_cast<size_t>(lag + frame)]);
        }

        const double captured_energy =
            captured_squared_prefix_sums[static_cast<size_t>(lag + chirp_frame_count)] -
            captured_squared_prefix_sums[static_cast<size_t>(lag)];
        if (captured_energy <= 0.0)
        {
            continue;
        }

        const double denominator =
            std::sqrt(reference_energy * captured_energy);
        if (denominator <= 0.0)
        {
            continue;
        }

        const double score = dot_product / denominator;
        const double absolute_score = std::fabs(score);
        if (absolute_score > best_absolute_score)
        {
            best_absolute_score = absolute_score;
            best_score = score;
            best_lag = lag;
        }
    }

    if (best_lag < 0)
    {
        return result;
    }

    result.delay_samples = best_lag;
    result.delay_ms =
        sample_offset_to_milliseconds(static_cast<double>(best_lag));
    result.score = best_score;
    result.absolute_score = best_absolute_score;
    return result;
}

bool save_capture_csv(
    const std::string &csv_path,
    const std::vector<float> &reference_output,
    const std::vector<float> &captured_input,
    const std::vector<float> &aligned_input,
    const CorrelationResult &correlation)
{
    std::vector<float> time_ms(reference_output.size(), 0.0f);
    std::vector<float> delay_ms_column(
        reference_output.size(),
        static_cast<float>(correlation.delay_ms));
    std::vector<float> delay_samples_column(
        reference_output.size(),
        static_cast<float>(correlation.delay_samples));
    std::vector<float> score_column(
        reference_output.size(),
        static_cast<float>(correlation.score));
    for (size_t i = 0; i < reference_output.size(); ++i)
    {
        time_ms[i] = static_cast<float>(
            sample_offset_to_milliseconds(static_cast<double>(i)));
    }

    return save_arrays_to_csv(
        csv_path,
        {
            "time_ms",
            "reference_output",
            "captured_input",
            "aligned_input",
            "correlation_delay_samples",
            "correlation_delay_ms",
            "correlation_score",
        },
        {
            &time_ms,
            &reference_output,
            &captured_input,
            &aligned_input,
            &delay_samples_column,
            &delay_ms_column,
            &score_column,
        });
}

std::filesystem::path plot_script_path()
{
    return std::filesystem::path(PORTABLE_OUTPUT_DIR).parent_path() /
           "scripts" /
           "plot_calculate_single_delay.py";
}

std::string build_capture_plot_command(const std::string &csv_path)
{
    return build_plot_command(
        "python3",
        plot_script_path().string(),
        {csv_path});
}

PaError abort_stream_if_supported(PaStream *stream)
{
#if MOCK
    return Pa_StopStream(stream);
#else
    return Pa_AbortStream(stream);
#endif
}

bool is_already_stopped_stream_error(PaError error)
{
#if MOCK
    (void)error;
    return false;
#else
    return error == paStreamIsStopped;
#endif
}

bool run_single_capture(
    PaStream *stream,
    CaptureData *capture_data,
    InteractionCapture *interaction_capture)
{
    if (!stream || !capture_data || !interaction_capture)
    {
        return false;
    }

    std::fill(
        interaction_capture->full_captured_input.begin(),
        interaction_capture->full_captured_input.end(),
        0.0f);

    capture_data->reference_output =
        interaction_capture->full_reference_output.data();
    capture_data->captured_input =
        interaction_capture->full_captured_input.data();
    capture_data->frame_index = 0;
    capture_data->max_frames =
        static_cast<int>(interaction_capture->full_reference_output.size());
    capture_data->input_underflow_count = 0;
    capture_data->input_overflow_count = 0;
    capture_data->output_underflow_count = 0;
    capture_data->output_overflow_count = 0;

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        std::cerr << "Pa_StartStream failed: " << Pa_GetErrorText(start_error)
                  << '\n';
        return false;
    }

    while (g_keep_running.load())
    {
        const PaError active_state = Pa_IsStreamActive(stream);
        if (active_state == 0)
        {
            break;
        }
        if (active_state < 0)
        {
            std::cerr << "Pa_IsStreamActive failed: "
                      << Pa_GetErrorText(active_state) << '\n';
            abort_stream_if_supported(stream);
            return false;
        }
        Pa_Sleep(2);
    }

    if (!g_keep_running.load())
    {
        abort_stream_if_supported(stream);
        return false;
    }

    const PaError stop_error = Pa_StopStream(stream);
    if (stop_error != paNoError && !is_already_stopped_stream_error(stop_error))
    {
        std::cerr << "Pa_StopStream failed: " << Pa_GetErrorText(stop_error)
                  << '\n';
        return false;
    }

    interaction_capture->input_underflow_count =
        capture_data->input_underflow_count;
    interaction_capture->input_overflow_count =
        capture_data->input_overflow_count;
    interaction_capture->output_underflow_count =
        capture_data->output_underflow_count;
    interaction_capture->output_overflow_count =
        capture_data->output_overflow_count;
    return true;
}

} // namespace

int main()
{
    std::signal(SIGINT, handle_interrupt_signal);

    Options options;

    if (options.input_port_1based <= 0 || options.output_port_1based <= 0)
    {
        std::cerr << "input_port and output_port must be 1-based positive integers.\n";
        return 1;
    }
    if (!is_finite_number(options.threshold) ||
        options.threshold < 0.0 || options.threshold > 1.0)
    {
        std::cerr << "threshold must be in [0, 1].\n";
        return 1;
    }
    const int stream_channel_count =
        std::max(options.input_port_1based, options.output_port_1based);
    const int input_channel_index = options.input_port_1based - 1;
    const int output_channel_index = options.output_port_1based - 1;
    const int interaction_frames = static_cast<int>(std::llround(
        kInteractionSeconds * static_cast<double>(SAMPLE_RATE)));
    const int pre_roll_frames = static_cast<int>(std::llround(
        INTERNAL_PREROLL_SECONDS * static_cast<double>(SAMPLE_RATE)));
    const int total_capture_frames = pre_roll_frames + interaction_frames;

    print_requested_config(options);

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

    const int device_index = select_measurement_device();
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

    if (device_info->maxInputChannels < stream_channel_count ||
        device_info->maxOutputChannels < stream_channel_count)
    {
        std::cerr << "Selected device does not expose "
                  << stream_channel_count
                  << " full-duplex channels.\n";
        list_all_devices();
        Pa_Terminate();
        return 1;
    }

    PaStreamParameters input_parameters{};
    input_parameters.device = device_index;
    input_parameters.channelCount = stream_channel_count;
    input_parameters.sampleFormat = paFloat32;
    input_parameters.suggestedLatency = device_info->defaultLowInputLatency;
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = stream_channel_count;
    output_parameters.sampleFormat = paFloat32;
    output_parameters.suggestedLatency = device_info->defaultLowOutputLatency;
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    const PaError format_error = Pa_IsFormatSupported(
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE);
    if (format_error != paFormatIsSupported)
    {
        std::cerr << "Requested stream configuration is not supported: "
                  << Pa_GetErrorText(format_error) << '\n';
        list_all_devices();
        Pa_Terminate();
        return 1;
    }

    InteractionCapture interaction_capture;
    interaction_capture.full_reference_output =
        build_reference_output_with_preroll(
            total_capture_frames,
            pre_roll_frames,
            interaction_frames);
    interaction_capture.full_captured_input.assign(
        static_cast<size_t>(total_capture_frames),
        0.0f);

    CaptureData capture_data{};
    capture_data.reference_output =
        interaction_capture.full_reference_output.data();
    capture_data.captured_input =
        interaction_capture.full_captured_input.data();
    capture_data.input_channel_index = input_channel_index;
    capture_data.output_channel_index = output_channel_index;
    capture_data.stream_channel_count = stream_channel_count;
    capture_data.max_frames = total_capture_frames;

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        capture_callback,
        &capture_data);
    if (open_error != paNoError)
    {
        std::cerr << "Pa_OpenStream failed: " << Pa_GetErrorText(open_error)
                  << '\n';
        Pa_Terminate();
        return 1;
    }

    const bool capture_ok =
        run_single_capture(stream, &capture_data, &interaction_capture);
    if (stream)
    {
        Pa_CloseStream(stream);
    }
    Pa_Terminate();

    if (!capture_ok)
    {
        std::cerr << "Single-shot capture failed.\n";
        std::cout << -1 << '\n';
        return 1;
    }

    interaction_capture.interaction_reference_output = slice_signal(
        interaction_capture.full_reference_output,
        pre_roll_frames,
        interaction_frames);
    interaction_capture.interaction_captured_input = slice_signal(
        interaction_capture.full_captured_input,
        pre_roll_frames,
        interaction_frames);

    const CorrelationResult correlation = infer_correlation_delay(
        interaction_capture.interaction_reference_output,
        interaction_capture.interaction_captured_input,
        chirp_start_frame_in_interaction());
    interaction_capture.correlation = correlation;
    interaction_capture.aligned_input =
        correlation.delay_samples >= 0
            ? shift_signal_left_by_samples(
                  interaction_capture.interaction_captured_input,
                  static_cast<double>(correlation.delay_samples))
            : interaction_capture.interaction_captured_input;

    const std::filesystem::path output_dir(PORTABLE_OUTPUT_DIR);
    const std::string csv_path =
        (output_dir / "calculate_single_delay_capture.csv").string();
    if (!save_capture_csv(
            csv_path,
            interaction_capture.interaction_reference_output,
            interaction_capture.interaction_captured_input,
            interaction_capture.aligned_input,
            interaction_capture.correlation))
    {
        std::cerr << "Failed to save plot-ready CSV: " << csv_path << '\n';
    }

    const std::string plot_command = build_capture_plot_command(csv_path);
    std::cout << "Saved CSV: " << csv_path << '\n'
              << "Plot command:\n"
              << "  " << plot_command << '\n';

    if (interaction_capture.input_underflow_count != 0 ||
        interaction_capture.input_overflow_count != 0 ||
        interaction_capture.output_underflow_count != 0 ||
        interaction_capture.output_overflow_count != 0)
    {
        std::cerr << "WARNING: PortAudio status flags were not clean:"
                  << " input_underflows=" << interaction_capture.input_underflow_count
                  << " input_overflows=" << interaction_capture.input_overflow_count
                  << " output_underflows=" << interaction_capture.output_underflow_count
                  << " output_overflows=" << interaction_capture.output_overflow_count
                  << '\n';
    }

    if (correlation.delay_samples < 0 ||
        !is_finite_number(correlation.absolute_score))
    {
        std::cerr << "FATAL: could not infer a valid correlation delay.\n";
        std::cout << -1 << '\n';
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6)
              << "Correlation delay:"
              << " samples=" << correlation.delay_samples
              << " ms=" << correlation.delay_ms
              << " score=" << correlation.score
              << " abs_score=" << correlation.absolute_score
              << '\n';

    if (correlation.absolute_score < options.threshold)
    {
        std::cerr << "FATAL: max correlation " << correlation.absolute_score
                  << " is below threshold " << options.threshold << ".\n";
        std::cout << -1 << '\n';
        return 1;
    }

    std::cout << correlation.delay_samples << '\n';
    return 0;
}
