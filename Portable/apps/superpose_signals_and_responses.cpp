/*
Build from repo root:

  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=superpose_signals_and_responses -DPORTABLE_USE_MOCK=OFF
  cmake --build Portable/build --target portable_superpose_signals_and_responses --parallel

If `Portable/build` does not exist yet, came from another machine, or you changed
`PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` line before
the `cmake --build ...` line.

Run:
  ./Portable/build/portable_superpose_signals_and_responses

Optional:
  ./Portable/build/portable_superpose_signals_and_responses --max-repetitions 10
  ./Portable/build/portable_superpose_signals_and_responses --max-repetitions 10 --no-plots

This app reuses the shared sound-device config for device name, channel count,
and sample format, but it forces:
  SAMPLE_RATE = 44100
  FRAMES_PER_BUFFER = 512

Behavior:
  - forever repeat one 0.1 s interaction
  - each interaction plays 10 times:
      0.005 s of silence
      then 0.005 s of a 10 kHz sine on output channel 5 only
  - records input channel 19 only during that 0.1 s window
  - internally primes the stream with a short silence before each interaction
  - waits 0.3 s between interactions
  - saves CSV data every interaction
  - refreshes the plots every 10 interactions
  - adds the normalized response to a consolidated overlay plot
*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
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

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 44100
#endif

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 512
#endif

#include "portable/sound_device_query_config.h"

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef TONE_FREQUENCY_HZ
#define TONE_FREQUENCY_HZ 10000.0
#endif

#ifndef TONE_AMPLITUDE
#define TONE_AMPLITUDE 0.8
#endif

#ifndef BURST_COUNT
#define BURST_COUNT 10
#endif

#ifndef BURST_TONE_SECONDS
#define BURST_TONE_SECONDS 0.005
#endif

#ifndef BURST_SILENCE_SECONDS
#define BURST_SILENCE_SECONDS 0.005
#endif

#ifndef OUTPUT_CHANNEL_INDEX
#define OUTPUT_CHANNEL_INDEX 4
#endif

#ifndef INPUT_CHANNEL_INDEX
#define INPUT_CHANNEL_INDEX 18
#endif

#ifndef PLOT_EVERY_INTERACTIONS
#define PLOT_EVERY_INTERACTIONS 10
#endif

#ifndef INTERACTION_PAUSE_SECONDS
#define INTERACTION_PAUSE_SECONDS 0.3
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
    static_cast<double>(BURST_COUNT) *
    (BURST_TONE_SECONDS + BURST_SILENCE_SECONDS);
constexpr double kPreRollSeconds = 0.05;

std::atomic<bool> g_keep_running{true};

struct SuperposeCaptureData
{
    const float *reference_output = nullptr;
    float *captured_input = nullptr;
    int frame_index = 0;
    int max_frames = 0;
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
};

struct InteractionCapture
{
    std::vector<float> full_reference_output;
    std::vector<float> full_captured_input;
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
};

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
              << " OUTPUT_CHANNEL_1BASED=" << (OUTPUT_CHANNEL_INDEX + 1)
              << " INPUT_CHANNEL_1BASED=" << (INPUT_CHANNEL_INDEX + 1)
              << " BURST_COUNT=" << BURST_COUNT
              << " BURST_TONE_SECONDS=" << BURST_TONE_SECONDS
              << " BURST_SILENCE_SECONDS=" << BURST_SILENCE_SECONDS
              << " INTERACTION_PAUSE_SECONDS=" << INTERACTION_PAUSE_SECONDS
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

float burst_sample_at_time(double time_seconds)
{
    if (time_seconds < 0.0 || time_seconds >= kInteractionSeconds)
    {
        return 0.0f;
    }

    const double burst_period_seconds =
        BURST_TONE_SECONDS + BURST_SILENCE_SECONDS;
    const double local_burst_time =
        std::fmod(time_seconds, burst_period_seconds);
    if (local_burst_time < 0.0 || local_burst_time < BURST_SILENCE_SECONDS)
    {
        return 0.0f;
    }

    if (local_burst_time >= BURST_SILENCE_SECONDS + BURST_TONE_SECONDS)
    {
        return 0.0f;
    }

    const double tone_time = local_burst_time - BURST_SILENCE_SECONDS;
    return static_cast<float>(
        TONE_AMPLITUDE *
        std::sin(2.0 * kPi * TONE_FREQUENCY_HZ * tone_time));
}

int superpose_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    SuperposeCaptureData *data = static_cast<SuperposeCaptureData *>(userData);
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
            input ? input[i * CHANNELS + INPUT_CHANNEL_INDEX] : 0.0f;

        for (int channel = 0; channel < CHANNELS; ++channel)
        {
            output[i * CHANNELS + channel] =
                channel == OUTPUT_CHANNEL_INDEX ? output_sample : 0.0f;
        }
    }

    for (int i = frames_to_process; i < static_cast<int>(framesPerBuffer); ++i)
    {
        for (int channel = 0; channel < CHANNELS; ++channel)
        {
            output[i * CHANNELS + channel] = 0.0f;
        }
    }

    data->frame_index += frames_to_process;
    return data->frame_index >= data->max_frames ? paComplete : paContinue;
}

std::vector<float> build_reference_output(int interaction_frames)
{
    std::vector<float> reference_output(
        static_cast<size_t>(interaction_frames),
        0.0f);
    for (int frame = 0; frame < interaction_frames; ++frame)
    {
        const double time_seconds =
            static_cast<double>(frame) / static_cast<double>(SAMPLE_RATE);
        reference_output[static_cast<size_t>(frame)] =
            burst_sample_at_time(time_seconds);
    }
    return reference_output;
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
            burst_sample_at_time(time_seconds);
    }
    return reference_output;
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

std::vector<float> normalize_signal(const std::vector<float> &signal)
{
    const double peak = max_absolute_value(signal);
    if (peak <= 0.0)
    {
        return std::vector<float>(signal.size(), 0.0f);
    }

    std::vector<float> normalized(signal.size(), 0.0f);
    for (size_t i = 0; i < signal.size(); ++i)
    {
        normalized[i] = static_cast<float>(
            static_cast<double>(signal[i]) / peak);
    }
    return normalized;
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

double samples_to_milliseconds(int sample_count)
{
    return 1000.0 *
           static_cast<double>(sample_count) /
           static_cast<double>(SAMPLE_RATE);
}

std::filesystem::path plot_script_path()
{
    return std::filesystem::path(PORTABLE_OUTPUT_DIR).parent_path() /
           "scripts" /
           "plot_superpose_signals_and_responses.py";
}

bool plot_csv_with_superpose_script(const std::string &csv_path)
{
    const std::filesystem::path script_path = plot_script_path();

    int rc = run_plot_command("python3", script_path.string(), csv_path);
    if (rc == 0)
    {
        return true;
    }

    rc = run_plot_command("python", script_path.string(), csv_path);
    if (rc == 0)
    {
        return true;
    }

    std::cerr
        << "Could not generate superpose_signals_and_responses plot automatically. "
        << "The CSV is still available at: " << csv_path << '\n';
    return false;
}

bool save_latest_interaction_csv(
    const std::string &csv_path,
    const std::vector<float> &normalized_reference,
    const std::vector<float> &normalized_response)
{
    if (normalized_reference.size() != normalized_response.size())
    {
        return false;
    }

    std::vector<float> time_ms(normalized_reference.size(), 0.0f);
    for (size_t i = 0; i < normalized_reference.size(); ++i)
    {
        time_ms[i] = static_cast<float>(
            samples_to_milliseconds(static_cast<int>(i)));
    }

    return save_arrays_to_csv(
        csv_path,
        {
            "time_ms",
            "normalized_reference",
            "normalized_response"},
        {
            &time_ms,
            &normalized_reference,
            &normalized_response});
}

bool save_consolidated_csv(
    const std::string &csv_path,
    const std::vector<float> &normalized_reference,
    const std::vector<std::vector<float>> &normalized_responses)
{
    std::vector<float> time_ms(normalized_reference.size(), 0.0f);
    for (size_t i = 0; i < normalized_reference.size(); ++i)
    {
        time_ms[i] = static_cast<float>(
            samples_to_milliseconds(static_cast<int>(i)));
    }

    std::vector<std::string> headers = {
        "time_ms",
        "normalized_reference"};
    std::vector<const std::vector<float> *> columns = {
        &time_ms,
        &normalized_reference};

    for (size_t i = 0; i < normalized_responses.size(); ++i)
    {
        headers.push_back("response_rep_" + std::to_string(i + 1));
        columns.push_back(&normalized_responses[i]);
    }

    return save_arrays_to_csv(csv_path, headers, columns);
}

bool capture_single_interaction(
    PaStream *stream,
    SuperposeCaptureData *capture_data,
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
            Pa_AbortStream(stream);
            return false;
        }
        Pa_Sleep(2);
    }

    if (!g_keep_running.load())
    {
        Pa_AbortStream(stream);
        return false;
    }

    const PaError stop_error = Pa_StopStream(stream);
    if (stop_error != paNoError && stop_error != paStreamIsStopped)
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

int main(int argc, char **argv)
{
    std::signal(SIGINT, handle_interrupt_signal);

    long long max_repetitions = 0;
    bool plots_enabled = true;
    for (int argi = 1; argi < argc; ++argi)
    {
        const std::string arg = argv[argi] ? std::string(argv[argi]) : std::string();
        if (arg == "--no-plots")
        {
            plots_enabled = false;
            continue;
        }
        if (arg == "--max-repetitions" && argi + 1 < argc)
        {
            try
            {
                max_repetitions = std::stoll(argv[++argi]);
            }
            catch (const std::exception &)
            {
                std::cerr << "Invalid value for --max-repetitions.\n";
                return 1;
            }
            if (max_repetitions < 0)
            {
                std::cerr << "--max-repetitions must be non-negative.\n";
                return 1;
            }
            continue;
        }

        std::cerr << "Unknown argument: " << arg << '\n';
        return 1;
    }

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
    if (OUTPUT_CHANNEL_INDEX < 0 || OUTPUT_CHANNEL_INDEX >= CHANNELS)
    {
        std::cerr << "OUTPUT_CHANNEL_INDEX must be in [0, CHANNELS).\n";
        return 1;
    }
    if (INPUT_CHANNEL_INDEX < 0 || INPUT_CHANNEL_INDEX >= CHANNELS)
    {
        std::cerr << "INPUT_CHANNEL_INDEX must be in [0, CHANNELS).\n";
        return 1;
    }

    const int interaction_frames = static_cast<int>(std::llround(
        kInteractionSeconds * static_cast<double>(SAMPLE_RATE)));
    const int pre_roll_frames = static_cast<int>(std::llround(
        kPreRollSeconds * static_cast<double>(SAMPLE_RATE)));
    const int total_capture_frames = pre_roll_frames + interaction_frames;
    if (interaction_frames <= 0)
    {
        std::cerr << "Invalid interaction duration configuration.\n";
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
    input_parameters.suggestedLatency = device_info->defaultHighInputLatency;
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = CHANNELS;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = device_info->defaultHighOutputLatency;
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    const PaError format_error =
        Pa_IsFormatSupported(&input_parameters, &output_parameters, SAMPLE_RATE);
    if (format_error != paFormatIsSupported)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested full-duplex stream configuration is not supported",
            format_error);
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

    SuperposeCaptureData capture_data{};
    capture_data.max_frames = total_capture_frames;

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        superpose_callback,
        &capture_data);
    if (open_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_OpenStream failed for the requested superpose configuration",
            open_error);
        Pa_Terminate();
        return 1;
    }

    const std::filesystem::path output_dir(PORTABLE_OUTPUT_DIR);
    const std::string latest_csv =
        (output_dir / "superpose_signals_and_responses_latest.csv").string();
    const std::string consolidated_csv =
        (output_dir / "superpose_signals_and_responses_consolidated.csv").string();

    const std::vector<float> normalized_reference =
        normalize_signal(slice_signal(
            interaction_capture.full_reference_output,
            pre_roll_frames,
            interaction_frames));
    std::vector<std::vector<float>> consolidated_responses;
    long long repetition = 0;

    while (g_keep_running.load() &&
           (max_repetitions == 0 || repetition < max_repetitions))
    {
        repetition++;
        if (!capture_single_interaction(stream, &capture_data, &interaction_capture))
        {
            break;
        }

        const std::vector<float> interaction_response = slice_signal(
            interaction_capture.full_captured_input,
            pre_roll_frames,
            interaction_frames);
        const std::vector<float> normalized_response =
            normalize_signal(interaction_response);
        consolidated_responses.push_back(normalized_response);

        double mean_square = 0.0;
        if (!interaction_response.empty())
        {
            double square_sum = 0.0;
            for (float sample : interaction_response)
            {
                const double value = static_cast<double>(sample);
                square_sum += value * value;
            }
            mean_square =
                square_sum / static_cast<double>(interaction_response.size());
        }

        std::cout << "rep=" << repetition
                  << " selected_input_channel=" << (INPUT_CHANNEL_INDEX + 1)
                  << " selected_input_mean_square=" << mean_square
                  << " raw_input_peak=" << max_absolute_value(interaction_response)
                  << " in_under=" << interaction_capture.input_underflow_count
                  << " in_over=" << interaction_capture.input_overflow_count
                  << " out_under=" << interaction_capture.output_underflow_count
                  << " out_over=" << interaction_capture.output_overflow_count
                  << '\n';

        const bool should_plot_now =
            plots_enabled && (repetition % PLOT_EVERY_INTERACTIONS) == 0;

        if (save_latest_interaction_csv(
                latest_csv,
                normalized_reference,
                normalized_response))
        {
            if (should_plot_now)
            {
                plot_csv_with_superpose_script(latest_csv);
            }
        }

        if (save_consolidated_csv(
                consolidated_csv,
                normalized_reference,
                consolidated_responses))
        {
            if (should_plot_now)
            {
                plot_csv_with_superpose_script(consolidated_csv);
            }
        }

        if (g_keep_running.load())
        {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(INTERACTION_PAUSE_SECONDS));
        }
    }

    if (stream)
    {
        Pa_CloseStream(stream);
    }
    Pa_Terminate();
    return 0;
}
