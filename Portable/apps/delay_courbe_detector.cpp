/*
Build from repo root:

  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=delay_courbe_detector -DPORTABLE_USE_MOCK=OFF
  cmake --build Portable/build --target portable_delay_courbe_detector --parallel

If `Portable/build` does not exist yet, came from another machine, or you changed
`PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` line before
the `cmake --build ...` line.

Run:
  ./Portable/build/portable_delay_courbe_detector

Optional:
  ./Portable/build/portable_delay_courbe_detector --max-repetitions 10
  ./Portable/build/portable_delay_courbe_detector --max-repetitions 10 --no-plots

This app keeps its own local device/channel/sample-format defaults and forces:
  SAMPLE_RATE = 44100
  FRAMES_PER_BUFFER = 512

Behavior:
  - play a 1000 Hz sine for 0.1 s on all output channels
  - record 0.2 s from input channel 1
  - infer delay as the first sample where abs(input) >= max(abs(input)) / 2
  - wait 0.1 s
  - append delay to the running list
  - whenever a new biggest/smallest delay appears, save a plot-ready CSV and PNG
  - every 500 repetitions, save a 100-bin delay distribution CSV and PNG
*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <limits>
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

#ifndef DEVICE_NAME
#define DEVICE_NAME "MADIface USB (24285073): Audio (hw:1,0)"
#endif

#ifndef CHANNELS
#define CHANNELS 32
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef TONE_FREQUENCY_HZ
#define TONE_FREQUENCY_HZ 1000.0
#endif

#ifndef TONE_AMPLITUDE
#define TONE_AMPLITUDE 0.8
#endif

#ifndef TONE_SECONDS
#define TONE_SECONDS 0.1
#endif

#ifndef CAPTURE_SECONDS
#define CAPTURE_SECONDS 0.2
#endif

#ifndef COOLDOWN_SECONDS
#define COOLDOWN_SECONDS 0.1
#endif

#ifndef INPUT_CHANNEL_INDEX
#define INPUT_CHANNEL_INDEX 0
#endif

#ifndef DELAY_THRESHOLD_FRACTION
#define DELAY_THRESHOLD_FRACTION 0.5
#endif

#ifndef MINIMUM_INPUT_PEAK
#define MINIMUM_INPUT_PEAK 1.0e-6
#endif

#ifndef DISTRIBUTION_SAVE_EVERY
#define DISTRIBUTION_SAVE_EVERY 100
#endif

#ifndef DISTRIBUTION_BIN_COUNT
#define DISTRIBUTION_BIN_COUNT 100
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
std::atomic<bool> g_keep_running{true};

struct DelayCourbeCaptureData
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

struct IterationCapture
{
    std::vector<float> reference_output;
    std::vector<float> captured_input;
    int delay_samples = -1;
    double delay_ms = -1.0;
    double input_peak = 0.0;
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
              << " TONE_SECONDS=" << TONE_SECONDS
              << " CAPTURE_SECONDS=" << CAPTURE_SECONDS
              << " COOLDOWN_SECONDS=" << COOLDOWN_SECONDS
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

int delay_courbe_detector_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    DelayCourbeCaptureData *data =
        static_cast<DelayCourbeCaptureData *>(userData);
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
            output[i * CHANNELS + channel] = output_sample;
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

std::vector<float> build_reference_output(int capture_frames, int tone_frames)
{
    std::vector<float> reference_output(
        static_cast<size_t>(capture_frames),
        0.0f);

    for (int frame = 0; frame < tone_frames && frame < capture_frames; ++frame)
    {
        const double time_seconds =
            static_cast<double>(frame) / static_cast<double>(SAMPLE_RATE);
        reference_output[static_cast<size_t>(frame)] = static_cast<float>(
            TONE_AMPLITUDE * std::sin(2.0 * kPi * TONE_FREQUENCY_HZ * time_seconds));
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

int infer_delay_samples(
    const std::vector<float> &captured_input,
    double *out_peak = nullptr)
{
    const double peak = max_absolute_value(captured_input);
    if (out_peak)
    {
        *out_peak = peak;
    }

    if (peak < MINIMUM_INPUT_PEAK)
    {
        return -1;
    }

    const double threshold = peak * DELAY_THRESHOLD_FRACTION;
    for (size_t index = 0; index < captured_input.size(); ++index)
    {
        if (std::fabs(static_cast<double>(captured_input[index])) >= threshold)
        {
            return static_cast<int>(index);
        }
    }

    return -1;
}

double samples_to_milliseconds(int sample_count)
{
    return 1000.0 *
           static_cast<double>(sample_count) /
           static_cast<double>(SAMPLE_RATE);
}

std::filesystem::path delay_courbe_plot_script_path()
{
    return std::filesystem::path(PORTABLE_OUTPUT_DIR).parent_path() /
           "scripts" /
           "plot_delay_courbe_detector.py";
}

bool plot_delay_courbe_csv(const std::string &csv_path)
{
    const std::filesystem::path script_path = delay_courbe_plot_script_path();
    const std::vector<std::string> script_args = {csv_path};

    if (run_plot_command("python3", script_path.string(), script_args, false) == 0)
    {
        std::cout << "Interactive plot command: "
                  << build_plot_command(
                         "python3",
                         script_path.string(),
                         script_args)
                  << '\n';
        return true;
    }

    if (run_plot_command("python", script_path.string(), script_args, false) == 0)
    {
        std::cout << "Interactive plot command: "
                  << build_plot_command(
                         "python",
                         script_path.string(),
                         script_args)
                  << '\n';
        return true;
    }

    std::cerr
        << "Could not generate delay_courbe_detector plot automatically. "
        << "Try plotting interactively with: "
        << build_plot_command("python3", script_path.string(), script_args)
        << '\n'
        << "The CSV is still available at: " << csv_path << '\n';
    return false;
}

bool save_signal_plot_csv(
    const std::string &csv_path,
    const std::vector<float> &reference_output,
    const std::vector<float> &captured_input,
    double delay_ms)
{
    if (reference_output.size() != captured_input.size())
    {
        return false;
    }

    std::vector<float> time_ms(reference_output.size(), 0.0f);
    std::vector<float> delay_marker(reference_output.size(), static_cast<float>(delay_ms));
    for (size_t i = 0; i < reference_output.size(); ++i)
    {
        time_ms[i] = static_cast<float>(samples_to_milliseconds(static_cast<int>(i)));
    }

    return save_arrays_to_csv(
        csv_path,
        {"time_ms", "reference_output", "input_ch1", "delay_ms_marker"},
        {&time_ms, &reference_output, &captured_input, &delay_marker});
}

bool save_delay_distribution_csv(
    const std::string &csv_path,
    const std::vector<float> &delays_ms)
{
    std::vector<float> repetitions(delays_ms.size(), 0.0f);
    for (size_t i = 0; i < delays_ms.size(); ++i)
    {
        repetitions[i] = static_cast<float>(i + 1);
    }

    return save_arrays_to_csv(
        csv_path,
        {"repetition", "delay_ms"},
        {&repetitions, &delays_ms});
}

bool capture_single_iteration(
    PaStream *stream,
    DelayCourbeCaptureData *capture_data,
    IterationCapture *iteration_capture)
{
    if (!stream || !capture_data || !iteration_capture)
    {
        return false;
    }

    std::fill(
        iteration_capture->captured_input.begin(),
        iteration_capture->captured_input.end(),
        0.0f);
    capture_data->reference_output = iteration_capture->reference_output.data();
    capture_data->captured_input = iteration_capture->captured_input.data();
    capture_data->frame_index = 0;
    capture_data->max_frames =
        static_cast<int>(iteration_capture->reference_output.size());
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

        Pa_Sleep(5);
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

    iteration_capture->input_underflow_count =
        capture_data->input_underflow_count;
    iteration_capture->input_overflow_count =
        capture_data->input_overflow_count;
    iteration_capture->output_underflow_count =
        capture_data->output_underflow_count;
    iteration_capture->output_overflow_count =
        capture_data->output_overflow_count;

    iteration_capture->delay_samples = infer_delay_samples(
        iteration_capture->captured_input,
        &iteration_capture->input_peak);
    iteration_capture->delay_ms =
        iteration_capture->delay_samples >= 0
            ? samples_to_milliseconds(iteration_capture->delay_samples)
            : -1.0;
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
    if (INPUT_CHANNEL_INDEX < 0 || INPUT_CHANNEL_INDEX >= CHANNELS)
    {
        std::cerr << "INPUT_CHANNEL_INDEX must be in [0, CHANNELS).\n";
        return 1;
    }
    const int capture_frames = static_cast<int>(std::llround(
        CAPTURE_SECONDS * static_cast<double>(SAMPLE_RATE)));
    const int tone_frames = static_cast<int>(std::llround(
        TONE_SECONDS * static_cast<double>(SAMPLE_RATE)));
    if (capture_frames <= 0 || tone_frames <= 0 || tone_frames > capture_frames)
    {
        std::cerr << "Invalid tone/capture duration configuration.\n";
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

    IterationCapture iteration_capture;
    iteration_capture.reference_output =
        build_reference_output(capture_frames, tone_frames);
    iteration_capture.captured_input.assign(
        static_cast<size_t>(capture_frames),
        0.0f);

    DelayCourbeCaptureData capture_data{};
    capture_data.max_frames = capture_frames;

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        delay_courbe_detector_callback,
        &capture_data);
    if (open_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_OpenStream failed for the requested delay detector configuration",
            open_error);
        Pa_Terminate();
        return 1;
    }

    const std::filesystem::path output_dir(PORTABLE_OUTPUT_DIR);
    const std::string biggest_delay_csv =
        (output_dir / "delay_courbe_detector_biggest_delay.csv").string();
    const std::string smallest_delay_csv =
        (output_dir / "delay_courbe_detector_smallest_delay.csv").string();
    const std::string distribution_csv =
        (output_dir / "delay_courbe_detector_delay_distribuition.csv").string();

    std::vector<float> delays_ms;
    double biggest_delay_ms = -std::numeric_limits<double>::infinity();
    double smallest_delay_ms = std::numeric_limits<double>::infinity();
    long long repetition = 0;

    while (g_keep_running.load() &&
           (max_repetitions == 0 || repetition < max_repetitions))
    {
        repetition++;
        if (!capture_single_iteration(
                stream,
                &capture_data,
                &iteration_capture))
        {
            break;
        }

        std::cout << "rep=" << repetition
                  << " delay_samples=" << iteration_capture.delay_samples
                  << " delay_ms=" << iteration_capture.delay_ms
                  << " input_peak=" << iteration_capture.input_peak
                  << " in_under=" << iteration_capture.input_underflow_count
                  << " in_over=" << iteration_capture.input_overflow_count
                  << " out_under=" << iteration_capture.output_underflow_count
                  << " out_over=" << iteration_capture.output_overflow_count
                  << '\n';

        std::this_thread::sleep_for(
            std::chrono::duration<double>(COOLDOWN_SECONDS));

        if (iteration_capture.delay_samples < 0)
        {
            std::cerr << "No valid delay was detected for repetition "
                      << repetition << ".\n";
            continue;
        }

        const float delay_ms_value =
            static_cast<float>(iteration_capture.delay_ms);
        delays_ms.push_back(delay_ms_value);

        if (iteration_capture.delay_ms > biggest_delay_ms)
        {
            biggest_delay_ms = iteration_capture.delay_ms;
            if (save_signal_plot_csv(
                    biggest_delay_csv,
                    iteration_capture.reference_output,
                    iteration_capture.captured_input,
                    iteration_capture.delay_ms))
            {
                std::cout << "Saved biggest delay CSV: "
                          << biggest_delay_csv << '\n';
                if (plots_enabled)
                {
                    plot_delay_courbe_csv(biggest_delay_csv);
                }
            }
        }

        if (iteration_capture.delay_ms < smallest_delay_ms)
        {
            smallest_delay_ms = iteration_capture.delay_ms;
            if (save_signal_plot_csv(
                    smallest_delay_csv,
                    iteration_capture.reference_output,
                    iteration_capture.captured_input,
                    iteration_capture.delay_ms))
            {
                std::cout << "Saved smallest delay CSV: "
                          << smallest_delay_csv << '\n';
                if (plots_enabled)
                {
                    plot_delay_courbe_csv(smallest_delay_csv);
                }
            }
        }

        if ((repetition % DISTRIBUTION_SAVE_EVERY) == 0 &&
            !delays_ms.empty() &&
            save_delay_distribution_csv(distribution_csv, delays_ms))
        {
            std::cout << "Saved delay distribution CSV: "
                      << distribution_csv << '\n';
            if (plots_enabled)
            {
                plot_delay_courbe_csv(distribution_csv);
            }
        }
    }

    if (stream)
    {
        Pa_CloseStream(stream);
    }
    Pa_Terminate();

    return 0;
}
