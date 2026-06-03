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

This app uses the same device/channel/sample-format defaults as
infer_topology_and_save_it and forces:
  SAMPLE_RATE = 44100
  FRAMES_PER_BUFFER = 128

Behavior:
  - forever repeat one interaction
  - each interaction plays 5 times:
      0.05 s of silence
      then 0.005 s of a 10 kHz sine on output channel 2 only
  - records input channel 19 only during that 0.1 s window
  - internally primes the stream with a short silence before each interaction
  - waits 0.3 s between interactions
  - saves CSV data every 100 interactions and once more on exit
  - never auto-plots; it prints the 3 manual plot commands after each save
  - adds the normalized response to a consolidated overlay plot
  - detects the first response sample after the first output burst
  - compares raw delay against the PortAudio-timestamp-corrected delay
  - consolidates a running delay comparison plot
*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
#define FRAMES_PER_BUFFER 128
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
#define TONE_FREQUENCY_HZ 10000.0
#endif

#ifndef TONE_AMPLITUDE
#define TONE_AMPLITUDE 0.8
#endif

#ifndef BURST_COUNT
#define BURST_COUNT 5
#endif

#ifndef BURST_TONE_SECONDS
#define BURST_TONE_SECONDS 0.005
#endif

#ifndef BURST_SILENCE_SECONDS
#define BURST_SILENCE_SECONDS 0.05
#endif

#ifndef OUTPUT_CHANNEL_INDEX
#define OUTPUT_CHANNEL_INDEX 1
#endif

#ifndef INPUT_CHANNEL_INDEX
#define INPUT_CHANNEL_INDEX 18
#endif

#ifndef DELAY_THRESHOLD_FRACTION
#define DELAY_THRESHOLD_FRACTION 0.75
#endif

#ifndef MINIMUM_INPUT_PEAK
#define MINIMUM_INPUT_PEAK 1.0e-6
#endif

#ifndef SAVE_EVERY_INTERACTIONS
#define SAVE_EVERY_INTERACTIONS 100
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

struct CallbackTraceRow
{
    long long repetition = 0;
    int callback_index = 0;
    int absolute_frame_start = 0;
    int frames_per_buffer = 0;
    int frames_to_process = 0;
    unsigned long status_flags = 0;
    double input_buffer_adc_time = std::numeric_limits<double>::quiet_NaN();
    double current_time = std::numeric_limits<double>::quiet_NaN();
    double output_buffer_dac_time = std::numeric_limits<double>::quiet_NaN();
};

struct SuperposeCaptureData
{
    const float *reference_output = nullptr;
    float *captured_input = nullptr;
    double *input_adc_times = nullptr;
    double *output_dac_times = nullptr;
    long long repetition = 0;
    int callback_index = 0;
    int frame_index = 0;
    int max_frames = 0;
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
    std::vector<CallbackTraceRow> *callback_trace_rows = nullptr;
};

struct InteractionCapture
{
    std::vector<float> full_reference_output;
    std::vector<float> full_captured_input;
    std::vector<double> full_input_adc_times;
    std::vector<double> full_output_dac_times;
    int delay_samples = -1;
    double delay_ms = -1.0;
    double input_peak = 0.0;
    double mean_square = 0.0;
    int detected_input_frame_absolute = -1;
    double detected_input_adc_time = std::numeric_limits<double>::quiet_NaN();
    int first_tone_output_frame_absolute = -1;
    double first_tone_output_dac_time = std::numeric_limits<double>::quiet_NaN();
    double timestamp_corrected_delay_samples = std::numeric_limits<double>::quiet_NaN();
    double timestamp_corrected_delay_ms = std::numeric_limits<double>::quiet_NaN();
    double delay_correction_delta_samples = std::numeric_limits<double>::quiet_NaN();
    double delay_correction_delta_ms = std::numeric_limits<double>::quiet_NaN();
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
};

struct InteractionTimingSummary
{
    long long repetition = 0;
    int raw_delay_samples = -1;
    double raw_delay_ms = -1.0;
    double input_peak = 0.0;
    double mean_square = 0.0;
    int detected_input_frame_absolute = -1;
    double detected_input_adc_time = std::numeric_limits<double>::quiet_NaN();
    int first_tone_output_frame_absolute = -1;
    double first_tone_output_dac_time = std::numeric_limits<double>::quiet_NaN();
    double timestamp_corrected_delay_samples = std::numeric_limits<double>::quiet_NaN();
    double timestamp_corrected_delay_ms = std::numeric_limits<double>::quiet_NaN();
    double delay_correction_delta_samples = std::numeric_limits<double>::quiet_NaN();
    double delay_correction_delta_ms = std::numeric_limits<double>::quiet_NaN();
    int callback_count = 0;
    int priming_callback_count = 0;
    double first_output_minus_input_samples = std::numeric_limits<double>::quiet_NaN();
    double min_output_minus_input_samples = std::numeric_limits<double>::quiet_NaN();
    double max_output_minus_input_samples = std::numeric_limits<double>::quiet_NaN();
    double avg_output_minus_input_samples = std::numeric_limits<double>::quiet_NaN();
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
    double stream_input_latency_seconds = std::numeric_limits<double>::quiet_NaN();
    double stream_output_latency_seconds = std::numeric_limits<double>::quiet_NaN();
    double stream_sample_rate = std::numeric_limits<double>::quiet_NaN();
};

struct RankedResponse
{
    double raw_delay_ms = std::numeric_limits<double>::quiet_NaN();
    double corrected_delay_ms = std::numeric_limits<double>::quiet_NaN();
    std::vector<float> normalized_response;
};

struct CallbackRangeStats
{
    int callback_count = 0;
    int priming_callback_count = 0;
    double first_output_minus_input_samples = std::numeric_limits<double>::quiet_NaN();
    double min_output_minus_input_samples = std::numeric_limits<double>::quiet_NaN();
    double max_output_minus_input_samples = std::numeric_limits<double>::quiet_NaN();
    double avg_output_minus_input_samples = std::numeric_limits<double>::quiet_NaN();
};

struct StreamInfoSnapshot
{
    double input_latency_seconds = std::numeric_limits<double>::quiet_NaN();
    double output_latency_seconds = std::numeric_limits<double>::quiet_NaN();
    double sample_rate = std::numeric_limits<double>::quiet_NaN();
};

void handle_interrupt_signal(int)
{
    g_keep_running.store(false);
}

double seconds_to_samples(double seconds)
{
    return seconds * static_cast<double>(SAMPLE_RATE);
}

double sample_offset_to_milliseconds(double sample_count)
{
    return 1000.0 *
           sample_count /
           static_cast<double>(SAMPLE_RATE);
}

double quiet_nan()
{
    return std::numeric_limits<double>::quiet_NaN();
}

bool is_finite_number(double value)
{
    return std::isfinite(value);
}

PaError abort_or_stop_stream(PaStream *stream)
{
#if MOCK
    return Pa_StopStream(stream);
#else
    return Pa_AbortStream(stream);
#endif
}

bool stop_error_is_acceptable(PaError error)
{
#if MOCK
    return error == paNoError;
#else
    return error == paNoError || error == paStreamIsStopped;
#endif
}

StreamInfoSnapshot get_stream_info_snapshot(PaStream *stream)
{
    StreamInfoSnapshot snapshot;

    if (!stream)
    {
        return snapshot;
    }

#if MOCK
    snapshot.input_latency_seconds = stream->inputParams.suggestedLatency;
    snapshot.output_latency_seconds = stream->outputParams.suggestedLatency;
    snapshot.sample_rate = stream->sampleRate;
#else
    const PaStreamInfo *stream_info = Pa_GetStreamInfo(stream);
    if (stream_info)
    {
        snapshot.input_latency_seconds = stream_info->inputLatency;
        snapshot.output_latency_seconds = stream_info->outputLatency;
        snapshot.sample_rate = stream_info->sampleRate;
    }
#endif

    return snapshot;
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
    const PaStreamCallbackTimeInfo *timeInfo,
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

    if (data->callback_trace_rows)
    {
        CallbackTraceRow row;
        row.repetition = data->repetition;
        row.callback_index = data->callback_index;
        row.absolute_frame_start = data->frame_index;
        row.frames_per_buffer = static_cast<int>(framesPerBuffer);
        row.frames_to_process = frames_to_process;
        row.status_flags = statusFlags;
        if (timeInfo)
        {
            row.input_buffer_adc_time = timeInfo->inputBufferAdcTime;
            row.current_time = timeInfo->currentTime;
            row.output_buffer_dac_time = timeInfo->outputBufferDacTime;
        }
        data->callback_trace_rows->push_back(row);
    }

    for (int i = 0; i < frames_to_process; ++i)
    {
        const int absolute_frame = data->frame_index + i;
        const float output_sample = data->reference_output[absolute_frame];
        data->captured_input[absolute_frame] =
            input ? input[i * CHANNELS + INPUT_CHANNEL_INDEX] : 0.0f;
        if (data->input_adc_times)
        {
            data->input_adc_times[absolute_frame] =
                timeInfo
                    ? timeInfo->inputBufferAdcTime +
                          static_cast<double>(i) /
                              static_cast<double>(SAMPLE_RATE)
                    : quiet_nan();
        }
        if (data->output_dac_times)
        {
            data->output_dac_times[absolute_frame] =
                timeInfo
                    ? timeInfo->outputBufferDacTime +
                          static_cast<double>(i) /
                              static_cast<double>(SAMPLE_RATE)
                    : quiet_nan();
        }

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

    data->callback_index++;
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

int infer_delay_samples(
    const std::vector<float> &captured_input,
    int search_start_frame,
    double *out_peak = nullptr,
    int *out_detection_frame = nullptr)
{
    const int clamped_start = std::max(
        0,
        std::min(search_start_frame, static_cast<int>(captured_input.size())));

    const double peak = max_absolute_value(captured_input);
    if (out_peak)
    {
        *out_peak = 0.0;
    }
    if (out_detection_frame)
    {
        *out_detection_frame = -1;
    }

    double search_peak = 0.0;
    for (int index = clamped_start;
         index < static_cast<int>(captured_input.size());
         ++index)
    {
        search_peak = std::max(
            search_peak,
            std::fabs(static_cast<double>(captured_input[static_cast<size_t>(index)])));
    }

    if (out_peak)
    {
        *out_peak = search_peak;
    }

    if (peak < MINIMUM_INPUT_PEAK || search_peak < MINIMUM_INPUT_PEAK)
    {
        return -1;
    }

    const double threshold = search_peak * DELAY_THRESHOLD_FRACTION;
    for (int index = clamped_start;
         index < static_cast<int>(captured_input.size());
         ++index)
    {
        if (std::fabs(static_cast<double>(captured_input[static_cast<size_t>(index)])) >= threshold)
        {
            if (out_detection_frame)
            {
                *out_detection_frame = index;
            }
            return index - clamped_start;
        }
    }

    return -1;
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

int first_tone_start_frame_in_interaction()
{
    return static_cast<int>(std::llround(
        BURST_SILENCE_SECONDS * static_cast<double>(SAMPLE_RATE)));
}

double output_minus_input_samples(const CallbackTraceRow &row)
{
    if (!is_finite_number(row.output_buffer_dac_time) ||
        !is_finite_number(row.input_buffer_adc_time))
    {
        return quiet_nan();
    }

    return seconds_to_samples(
        row.output_buffer_dac_time - row.input_buffer_adc_time);
}

CallbackRangeStats summarize_callback_rows(
    const std::vector<CallbackTraceRow> &rows,
    size_t begin_index,
    size_t end_index)
{
    CallbackRangeStats stats;
    double sum_output_minus_input_samples = 0.0;
    int finite_output_minus_input_count = 0;

    for (size_t index = begin_index; index < end_index; ++index)
    {
        const CallbackTraceRow &row = rows[index];
        stats.callback_count++;

        if ((row.status_flags & paPrimingOutput) != 0)
        {
            stats.priming_callback_count++;
        }

        const double row_output_minus_input_samples =
            output_minus_input_samples(row);
        if (!is_finite_number(row_output_minus_input_samples))
        {
            continue;
        }

        if (finite_output_minus_input_count == 0)
        {
            stats.first_output_minus_input_samples =
                row_output_minus_input_samples;
            stats.min_output_minus_input_samples =
                row_output_minus_input_samples;
            stats.max_output_minus_input_samples =
                row_output_minus_input_samples;
        }
        else
        {
            stats.min_output_minus_input_samples = std::min(
                stats.min_output_minus_input_samples,
                row_output_minus_input_samples);
            stats.max_output_minus_input_samples = std::max(
                stats.max_output_minus_input_samples,
                row_output_minus_input_samples);
        }

        sum_output_minus_input_samples += row_output_minus_input_samples;
        finite_output_minus_input_count++;
    }

    if (finite_output_minus_input_count > 0)
    {
        stats.avg_output_minus_input_samples =
            sum_output_minus_input_samples /
            static_cast<double>(finite_output_minus_input_count);
    }

    return stats;
}

bool save_callback_trace_csv(
    const std::string &csv_path,
    const std::vector<CallbackTraceRow> &rows)
{
    if (!ensure_parent_directory(csv_path))
    {
        return false;
    }

    std::ofstream out(csv_path);
    if (!out.is_open())
    {
        return false;
    }

    out << std::setprecision(17);
    out << "repetition"
        << ';' << "callback_index"
        << ';' << "absolute_frame_start"
        << ';' << "frames_per_buffer"
        << ';' << "frames_to_process"
        << ';' << "status_flags"
        << ';' << "pa_priming_output"
        << ';' << "pa_input_underflow"
        << ';' << "pa_input_overflow"
        << ';' << "pa_output_underflow"
        << ';' << "pa_output_overflow"
        << ';' << "input_buffer_adc_time_seconds"
        << ';' << "current_time_seconds"
        << ';' << "output_buffer_dac_time_seconds"
        << ';' << "output_minus_input_samples"
        << ';' << "current_minus_input_samples"
        << ';' << "output_minus_current_samples"
        << '\n';

    for (const CallbackTraceRow &row : rows)
    {
        const double output_minus_input =
            output_minus_input_samples(row);
        const double current_minus_input =
            is_finite_number(row.current_time) &&
                    is_finite_number(row.input_buffer_adc_time)
                ? seconds_to_samples(
                      row.current_time - row.input_buffer_adc_time)
                : quiet_nan();
        const double output_minus_current =
            is_finite_number(row.output_buffer_dac_time) &&
                    is_finite_number(row.current_time)
                ? seconds_to_samples(
                      row.output_buffer_dac_time - row.current_time)
                : quiet_nan();

        out << row.repetition
            << ';' << row.callback_index
            << ';' << row.absolute_frame_start
            << ';' << row.frames_per_buffer
            << ';' << row.frames_to_process
            << ';' << row.status_flags
            << ';' << (((row.status_flags & paPrimingOutput) != 0) ? 1 : 0)
            << ';' << (((row.status_flags & paInputUnderflow) != 0) ? 1 : 0)
            << ';' << (((row.status_flags & paInputOverflow) != 0) ? 1 : 0)
            << ';' << (((row.status_flags & paOutputUnderflow) != 0) ? 1 : 0)
            << ';' << (((row.status_flags & paOutputOverflow) != 0) ? 1 : 0)
            << ';' << row.input_buffer_adc_time
            << ';' << row.current_time
            << ';' << row.output_buffer_dac_time
            << ';' << output_minus_input
            << ';' << current_minus_input
            << ';' << output_minus_current
            << '\n';
    }

    return true;
}

bool save_timing_summary_csv(
    const std::string &csv_path,
    const std::vector<InteractionTimingSummary> &rows)
{
    if (!ensure_parent_directory(csv_path))
    {
        return false;
    }

    std::ofstream out(csv_path);
    if (!out.is_open())
    {
        return false;
    }

    out << std::setprecision(17);
    out << "repetition"
        << ';' << "raw_delay_samples"
        << ';' << "raw_delay_ms"
        << ';' << "input_peak"
        << ';' << "mean_square"
        << ';' << "detected_input_frame_absolute"
        << ';' << "detected_input_adc_time_seconds"
        << ';' << "first_tone_output_frame_absolute"
        << ';' << "first_tone_output_dac_time_seconds"
        << ';' << "timestamp_corrected_delay_samples"
        << ';' << "timestamp_corrected_delay_ms"
        << ';' << "delay_correction_delta_samples"
        << ';' << "delay_correction_delta_ms"
        << ';' << "callback_count"
        << ';' << "priming_callback_count"
        << ';' << "first_output_minus_input_samples"
        << ';' << "min_output_minus_input_samples"
        << ';' << "max_output_minus_input_samples"
        << ';' << "avg_output_minus_input_samples"
        << ';' << "input_underflow_count"
        << ';' << "input_overflow_count"
        << ';' << "output_underflow_count"
        << ';' << "output_overflow_count"
        << ';' << "stream_input_latency_seconds"
        << ';' << "stream_output_latency_seconds"
        << ';' << "stream_sample_rate"
        << '\n';

    for (const InteractionTimingSummary &row : rows)
    {
        out << row.repetition
            << ';' << row.raw_delay_samples
            << ';' << row.raw_delay_ms
            << ';' << row.input_peak
            << ';' << row.mean_square
            << ';' << row.detected_input_frame_absolute
            << ';' << row.detected_input_adc_time
            << ';' << row.first_tone_output_frame_absolute
            << ';' << row.first_tone_output_dac_time
            << ';' << row.timestamp_corrected_delay_samples
            << ';' << row.timestamp_corrected_delay_ms
            << ';' << row.delay_correction_delta_samples
            << ';' << row.delay_correction_delta_ms
            << ';' << row.callback_count
            << ';' << row.priming_callback_count
            << ';' << row.first_output_minus_input_samples
            << ';' << row.min_output_minus_input_samples
            << ';' << row.max_output_minus_input_samples
            << ';' << row.avg_output_minus_input_samples
            << ';' << row.input_underflow_count
            << ';' << row.input_overflow_count
            << ';' << row.output_underflow_count
            << ';' << row.output_overflow_count
            << ';' << row.stream_input_latency_seconds
            << ';' << row.stream_output_latency_seconds
            << ';' << row.stream_sample_rate
            << '\n';
    }

    return true;
}

std::filesystem::path plot_script_path()
{
    return std::filesystem::path(PORTABLE_OUTPUT_DIR).parent_path() /
           "scripts" /
           "plot_superpose_signals_and_responses.py";
}

std::filesystem::path delay_distribution_plot_script_path()
{
    return std::filesystem::path(PORTABLE_OUTPUT_DIR).parent_path() /
           "scripts" /
           "plot_delay_courbe_detector.py";
}

bool run_save_plot_with_manual_hint(
    const std::filesystem::path &script_path,
    const std::string &csv_path,
    const std::string &plot_name)
{
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
        << "Could not generate " << plot_name << " plot automatically. "
        << "Try plotting interactively with: "
        << build_plot_command("python3", script_path.string(), script_args)
        << '\n'
        << "The CSV is still available at: " << csv_path << '\n';
    return false;
}

std::string build_superpose_show_command(const std::string &csv_path)
{
    return build_plot_command(
        "python3",
        plot_script_path().string(),
        {csv_path});
}

std::string build_delay_distribution_show_command(const std::string &csv_path)
{
    return build_plot_command(
        "python3",
        delay_distribution_plot_script_path().string(),
        {csv_path});
}

void print_manual_plot_commands(
    const std::string &selected_delays_csv,
    const std::string &consolidated_csv,
    const std::string &delay_distribution_csv)
{
    std::cout << "Manual plot commands:\n"
              << "  " << build_superpose_show_command(selected_delays_csv) << '\n'
              << "  " << build_superpose_show_command(consolidated_csv) << '\n'
              << "  "
              << build_delay_distribution_show_command(delay_distribution_csv)
              << '\n';
}

bool save_selected_delays_csv(
    const std::string &csv_path,
    const std::vector<float> &normalized_reference,
    double tone_start_ms,
    const RankedResponse &smallest_delay_response,
    const RankedResponse &median_delay_response,
    const RankedResponse &biggest_delay_response)
{
    if (normalized_reference.size() != smallest_delay_response.normalized_response.size() ||
        normalized_reference.size() != median_delay_response.normalized_response.size() ||
        normalized_reference.size() != biggest_delay_response.normalized_response.size())
    {
        return false;
    }

    std::vector<float> time_ms(normalized_reference.size(), 0.0f);
    const float tone_start_marker_value =
        static_cast<float>(tone_start_ms);
    const float smallest_delay_marker_value =
        is_finite_number(smallest_delay_response.raw_delay_ms)
            ? static_cast<float>(tone_start_ms + smallest_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float smallest_corrected_delay_marker_value =
        is_finite_number(smallest_delay_response.corrected_delay_ms)
            ? static_cast<float>(tone_start_ms + smallest_delay_response.corrected_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float smallest_delay_value =
        is_finite_number(smallest_delay_response.raw_delay_ms)
            ? static_cast<float>(smallest_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float smallest_corrected_delay_value =
        is_finite_number(smallest_delay_response.corrected_delay_ms)
            ? static_cast<float>(smallest_delay_response.corrected_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float smallest_delay_delta_value =
        is_finite_number(smallest_delay_response.raw_delay_ms) &&
                is_finite_number(smallest_delay_response.corrected_delay_ms)
            ? static_cast<float>(
                  smallest_delay_response.corrected_delay_ms -
                  smallest_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float median_delay_marker_value =
        is_finite_number(median_delay_response.raw_delay_ms)
            ? static_cast<float>(tone_start_ms + median_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float median_corrected_delay_marker_value =
        is_finite_number(median_delay_response.corrected_delay_ms)
            ? static_cast<float>(tone_start_ms + median_delay_response.corrected_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float median_delay_value =
        is_finite_number(median_delay_response.raw_delay_ms)
            ? static_cast<float>(median_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float median_corrected_delay_value =
        is_finite_number(median_delay_response.corrected_delay_ms)
            ? static_cast<float>(median_delay_response.corrected_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float median_delay_delta_value =
        is_finite_number(median_delay_response.raw_delay_ms) &&
                is_finite_number(median_delay_response.corrected_delay_ms)
            ? static_cast<float>(
                  median_delay_response.corrected_delay_ms -
                  median_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float biggest_delay_marker_value =
        is_finite_number(biggest_delay_response.raw_delay_ms)
            ? static_cast<float>(tone_start_ms + biggest_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float biggest_corrected_delay_marker_value =
        is_finite_number(biggest_delay_response.corrected_delay_ms)
            ? static_cast<float>(tone_start_ms + biggest_delay_response.corrected_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float biggest_delay_value =
        is_finite_number(biggest_delay_response.raw_delay_ms)
            ? static_cast<float>(biggest_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float biggest_corrected_delay_value =
        is_finite_number(biggest_delay_response.corrected_delay_ms)
            ? static_cast<float>(biggest_delay_response.corrected_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    const float biggest_delay_delta_value =
        is_finite_number(biggest_delay_response.raw_delay_ms) &&
                is_finite_number(biggest_delay_response.corrected_delay_ms)
            ? static_cast<float>(
                  biggest_delay_response.corrected_delay_ms -
                  biggest_delay_response.raw_delay_ms)
            : std::numeric_limits<float>::quiet_NaN();
    std::vector<float> tone_start_ms_marker(
        normalized_reference.size(),
        tone_start_marker_value);
    std::vector<float> smallest_delay_ms_marker(
        normalized_reference.size(),
        smallest_delay_marker_value);
    std::vector<float> smallest_corrected_delay_ms_marker(
        normalized_reference.size(),
        smallest_corrected_delay_marker_value);
    std::vector<float> smallest_delay_ms_value_column(
        normalized_reference.size(),
        smallest_delay_value);
    std::vector<float> smallest_corrected_delay_ms_value_column(
        normalized_reference.size(),
        smallest_corrected_delay_value);
    std::vector<float> smallest_delay_delta_ms_value_column(
        normalized_reference.size(),
        smallest_delay_delta_value);
    std::vector<float> median_delay_ms_marker(
        normalized_reference.size(),
        median_delay_marker_value);
    std::vector<float> median_corrected_delay_ms_marker(
        normalized_reference.size(),
        median_corrected_delay_marker_value);
    std::vector<float> median_delay_ms_value_column(
        normalized_reference.size(),
        median_delay_value);
    std::vector<float> median_corrected_delay_ms_value_column(
        normalized_reference.size(),
        median_corrected_delay_value);
    std::vector<float> median_delay_delta_ms_value_column(
        normalized_reference.size(),
        median_delay_delta_value);
    std::vector<float> biggest_delay_ms_marker(
        normalized_reference.size(),
        biggest_delay_marker_value);
    std::vector<float> biggest_corrected_delay_ms_marker(
        normalized_reference.size(),
        biggest_corrected_delay_marker_value);
    std::vector<float> biggest_delay_ms_value_column(
        normalized_reference.size(),
        biggest_delay_value);
    std::vector<float> biggest_corrected_delay_ms_value_column(
        normalized_reference.size(),
        biggest_corrected_delay_value);
    std::vector<float> biggest_delay_delta_ms_value_column(
        normalized_reference.size(),
        biggest_delay_delta_value);
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
            "tone_start_ms_marker",
            "normalized_response_smallest_delay",
            "raw_delay_ms_marker_smallest",
            "timestamp_corrected_delay_ms_marker_smallest",
            "raw_delay_ms_value_smallest",
            "timestamp_corrected_delay_ms_value_smallest",
            "delay_correction_delta_ms_smallest",
            "normalized_response_median_delay",
            "raw_delay_ms_marker_median",
            "timestamp_corrected_delay_ms_marker_median",
            "raw_delay_ms_value_median",
            "timestamp_corrected_delay_ms_value_median",
            "delay_correction_delta_ms_median",
            "normalized_response_biggest_delay",
            "raw_delay_ms_marker_biggest",
            "timestamp_corrected_delay_ms_marker_biggest",
            "raw_delay_ms_value_biggest",
            "timestamp_corrected_delay_ms_value_biggest",
            "delay_correction_delta_ms_biggest"},
        {
            &time_ms,
            &normalized_reference,
            &tone_start_ms_marker,
            &smallest_delay_response.normalized_response,
            &smallest_delay_ms_marker,
            &smallest_corrected_delay_ms_marker,
            &smallest_delay_ms_value_column,
            &smallest_corrected_delay_ms_value_column,
            &smallest_delay_delta_ms_value_column,
            &median_delay_response.normalized_response,
            &median_delay_ms_marker,
            &median_corrected_delay_ms_marker,
            &median_delay_ms_value_column,
            &median_corrected_delay_ms_value_column,
            &median_delay_delta_ms_value_column,
            &biggest_delay_response.normalized_response,
            &biggest_delay_ms_marker,
            &biggest_corrected_delay_ms_marker,
            &biggest_delay_ms_value_column,
            &biggest_corrected_delay_ms_value_column,
            &biggest_delay_delta_ms_value_column});
}

bool save_consolidated_csv(
    const std::string &csv_path,
    const std::vector<float> &normalized_reference,
    double tone_start_ms,
    const std::vector<std::vector<float>> &normalized_responses)
{
    std::vector<float> time_ms(normalized_reference.size(), 0.0f);
    std::vector<float> tone_start_ms_marker(
        normalized_reference.size(),
        static_cast<float>(tone_start_ms));
    for (size_t i = 0; i < normalized_reference.size(); ++i)
    {
        time_ms[i] = static_cast<float>(
            samples_to_milliseconds(static_cast<int>(i)));
    }

    std::vector<std::string> headers = {
        "time_ms",
        "normalized_reference",
        "tone_start_ms_marker"};
    std::vector<const std::vector<float> *> columns = {
        &time_ms,
        &normalized_reference,
        &tone_start_ms_marker};

    for (size_t i = 0; i < normalized_responses.size(); ++i)
    {
        headers.push_back("response_rep_" + std::to_string(i + 1));
        columns.push_back(&normalized_responses[i]);
    }

    return save_arrays_to_csv(csv_path, headers, columns);
}

bool save_delay_distribution_csv(
    const std::string &csv_path,
    const std::vector<InteractionTimingSummary> &timing_summaries)
{
    if (!ensure_parent_directory(csv_path))
    {
        return false;
    }

    std::ofstream out(csv_path);
    if (!out.is_open())
    {
        return false;
    }

    out << std::setprecision(17);
    out << "repetition"
        << ';' << "raw_delay_ms"
        << ';' << "timestamp_corrected_delay_ms"
        << ';' << "delay_correction_delta_ms"
        << '\n';

    for (const InteractionTimingSummary &summary : timing_summaries)
    {
        out << summary.repetition
            << ';' << summary.raw_delay_ms
            << ';' << summary.timestamp_corrected_delay_ms
            << ';' << summary.delay_correction_delta_ms
            << '\n';
    }

    return true;
}

bool capture_single_interaction(
    PaStream *stream,
    SuperposeCaptureData *capture_data,
    InteractionCapture *interaction_capture,
    long long repetition)
{
    if (!stream || !capture_data || !interaction_capture)
    {
        return false;
    }

    std::fill(
        interaction_capture->full_captured_input.begin(),
        interaction_capture->full_captured_input.end(),
        0.0f);
    std::fill(
        interaction_capture->full_input_adc_times.begin(),
        interaction_capture->full_input_adc_times.end(),
        quiet_nan());
    std::fill(
        interaction_capture->full_output_dac_times.begin(),
        interaction_capture->full_output_dac_times.end(),
        quiet_nan());
    capture_data->reference_output =
        interaction_capture->full_reference_output.data();
    capture_data->captured_input =
        interaction_capture->full_captured_input.data();
    capture_data->input_adc_times =
        interaction_capture->full_input_adc_times.data();
    capture_data->output_dac_times =
        interaction_capture->full_output_dac_times.data();
    capture_data->repetition = repetition;
    capture_data->callback_index = 0;
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
            abort_or_stop_stream(stream);
            return false;
        }
        Pa_Sleep(2);
    }

    if (!g_keep_running.load())
    {
        abort_or_stop_stream(stream);
        return false;
    }

    const PaError stop_error = Pa_StopStream(stream);
    if (!stop_error_is_acceptable(stop_error))
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
    for (int argi = 1; argi < argc; ++argi)
    {
        const std::string arg = argv[argi] ? std::string(argv[argi]) : std::string();
        if (arg == "--no-plots")
        {
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
    input_parameters.suggestedLatency = device_info->defaultLowInputLatency;
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = CHANNELS;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = device_info->defaultLowOutputLatency;
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
    interaction_capture.full_input_adc_times.assign(
        static_cast<size_t>(total_capture_frames),
        quiet_nan());
    interaction_capture.full_output_dac_times.assign(
        static_cast<size_t>(total_capture_frames),
        quiet_nan());

    SuperposeCaptureData capture_data{};
    capture_data.max_frames = total_capture_frames;
    std::vector<CallbackTraceRow> callback_trace_rows;
    capture_data.callback_trace_rows = &callback_trace_rows;

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

    const StreamInfoSnapshot stream_info = get_stream_info_snapshot(stream);
    if (is_finite_number(stream_info.sample_rate))
    {
        std::cout << "Opened stream info:"
                  << " inputLatency=" << stream_info.input_latency_seconds
                  << "s outputLatency=" << stream_info.output_latency_seconds
                  << "s sampleRate=" << stream_info.sample_rate
                  << '\n';
    }

    const std::filesystem::path output_dir(PORTABLE_OUTPUT_DIR);
    const std::string selected_delays_csv =
        (output_dir / "superpose_signals_and_responses_selected_delays.csv").string();
    const std::string consolidated_csv =
        (output_dir / "superpose_signals_and_responses_consolidated.csv").string();
    const std::string delay_distribution_csv =
        (output_dir / "superpose_signals_and_responses_delay_distribuition.csv").string();
    const std::string timing_summary_csv =
        (output_dir / "superpose_signals_and_responses_timing_summary.csv").string();
    const std::string callback_trace_csv =
        (output_dir / "superpose_signals_and_responses_callback_trace.csv").string();

    const std::vector<float> normalized_reference =
        normalize_signal(slice_signal(
            interaction_capture.full_reference_output,
            pre_roll_frames,
            interaction_frames));
    const int tone_start_frame_in_interaction =
        first_tone_start_frame_in_interaction();
    if (tone_start_frame_in_interaction < 0 ||
        tone_start_frame_in_interaction >= interaction_frames)
    {
        std::cerr << "The first tone start must be inside the interaction window.\n";
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    const double tone_start_ms =
        samples_to_milliseconds(tone_start_frame_in_interaction);
    const int first_tone_output_frame_absolute =
        pre_roll_frames + tone_start_frame_in_interaction;
    std::vector<std::vector<float>> consolidated_responses;
    std::vector<RankedResponse> delay_ranked_responses;
    std::vector<InteractionTimingSummary> timing_summaries;
    bool selected_delays_dirty = false;
    bool consolidated_dirty = false;
    bool distribution_dirty = false;
    bool timing_summary_dirty = false;
    bool callback_trace_dirty = false;
    long long repetition = 0;

    const auto flush_saved_csvs = [&](bool print_commands) {
        bool saved_any_csv = false;

        if (selected_delays_dirty &&
            !delay_ranked_responses.empty())
        {
            const size_t median_index = delay_ranked_responses.size() / 2;
            const auto &smallest = delay_ranked_responses.front();
            const auto &median = delay_ranked_responses[median_index];
            const auto &biggest = delay_ranked_responses.back();
            if (save_selected_delays_csv(
                    selected_delays_csv,
                    normalized_reference,
                    tone_start_ms,
                    smallest,
                    median,
                    biggest))
            {
                selected_delays_dirty = false;
                saved_any_csv = true;
            }
        }

        if (consolidated_dirty &&
            save_consolidated_csv(
                consolidated_csv,
                normalized_reference,
                tone_start_ms,
                consolidated_responses))
        {
            consolidated_dirty = false;
            saved_any_csv = true;
        }

        if (distribution_dirty &&
            !timing_summaries.empty() &&
            save_delay_distribution_csv(delay_distribution_csv, timing_summaries))
        {
            distribution_dirty = false;
            saved_any_csv = true;
        }

        if (timing_summary_dirty &&
            save_timing_summary_csv(timing_summary_csv, timing_summaries))
        {
            timing_summary_dirty = false;
            saved_any_csv = true;
        }

        if (callback_trace_dirty &&
            save_callback_trace_csv(callback_trace_csv, callback_trace_rows))
        {
            callback_trace_dirty = false;
            saved_any_csv = true;
        }

        if (saved_any_csv && print_commands)
        {
            print_manual_plot_commands(
                selected_delays_csv,
                consolidated_csv,
                delay_distribution_csv);
            std::cout << "Timing diagnostics CSVs:\n"
                      << "  " << timing_summary_csv << '\n'
                      << "  " << callback_trace_csv << '\n';
        }
    };

    while (g_keep_running.load() &&
           (max_repetitions == 0 || repetition < max_repetitions))
    {
        repetition++;
        const size_t callback_trace_begin = callback_trace_rows.size();
        if (!capture_single_interaction(
                stream,
                &capture_data,
                &interaction_capture,
                repetition))
        {
            break;
        }
        const size_t callback_trace_end = callback_trace_rows.size();

        const std::vector<float> interaction_response = slice_signal(
            interaction_capture.full_captured_input,
            pre_roll_frames,
            interaction_frames);
        const std::vector<float> normalized_response =
            normalize_signal(interaction_response);
        int detected_input_frame_in_interaction = -1;
        interaction_capture.delay_samples = infer_delay_samples(
            interaction_response,
            tone_start_frame_in_interaction,
            &interaction_capture.input_peak,
            &detected_input_frame_in_interaction);
        interaction_capture.delay_ms =
            interaction_capture.delay_samples >= 0
                ? samples_to_milliseconds(interaction_capture.delay_samples)
                : -1.0;
        consolidated_responses.push_back(normalized_response);
        consolidated_dirty = true;

        interaction_capture.mean_square = 0.0;
        if (!interaction_response.empty())
        {
            double square_sum = 0.0;
            for (float sample : interaction_response)
            {
                const double value = static_cast<double>(sample);
                square_sum += value * value;
            }
            interaction_capture.mean_square =
                square_sum / static_cast<double>(interaction_response.size());
        }

        interaction_capture.first_tone_output_frame_absolute =
            first_tone_output_frame_absolute;
        interaction_capture.first_tone_output_dac_time =
            first_tone_output_frame_absolute >= 0 &&
                    first_tone_output_frame_absolute <
                        static_cast<int>(interaction_capture.full_output_dac_times.size())
                ? interaction_capture.full_output_dac_times[static_cast<size_t>(
                      first_tone_output_frame_absolute)]
                : quiet_nan();

        interaction_capture.detected_input_frame_absolute =
            detected_input_frame_in_interaction >= 0
                ? pre_roll_frames + detected_input_frame_in_interaction
                : -1;
        interaction_capture.detected_input_adc_time =
            interaction_capture.detected_input_frame_absolute >= 0 &&
                    interaction_capture.detected_input_frame_absolute <
                        static_cast<int>(interaction_capture.full_input_adc_times.size())
                ? interaction_capture.full_input_adc_times[static_cast<size_t>(
                      interaction_capture.detected_input_frame_absolute)]
                : quiet_nan();

        interaction_capture.timestamp_corrected_delay_samples = quiet_nan();
        interaction_capture.timestamp_corrected_delay_ms = quiet_nan();
        interaction_capture.delay_correction_delta_samples = quiet_nan();
        interaction_capture.delay_correction_delta_ms = quiet_nan();
        if (is_finite_number(interaction_capture.detected_input_adc_time) &&
            is_finite_number(interaction_capture.first_tone_output_dac_time))
        {
            const double corrected_delay_seconds =
                interaction_capture.detected_input_adc_time -
                interaction_capture.first_tone_output_dac_time;
            interaction_capture.timestamp_corrected_delay_samples =
                seconds_to_samples(corrected_delay_seconds);
            interaction_capture.timestamp_corrected_delay_ms =
                1000.0 * corrected_delay_seconds;
        }
        if (interaction_capture.delay_samples >= 0 &&
            is_finite_number(interaction_capture.timestamp_corrected_delay_samples))
        {
            interaction_capture.delay_correction_delta_samples =
                interaction_capture.timestamp_corrected_delay_samples -
                static_cast<double>(interaction_capture.delay_samples);
            interaction_capture.delay_correction_delta_ms =
                interaction_capture.timestamp_corrected_delay_ms -
                interaction_capture.delay_ms;
        }

        const CallbackRangeStats callback_stats =
            summarize_callback_rows(
                callback_trace_rows,
                callback_trace_begin,
                callback_trace_end);

        InteractionTimingSummary timing_summary;
        timing_summary.repetition = repetition;
        timing_summary.raw_delay_samples = interaction_capture.delay_samples;
        timing_summary.raw_delay_ms = interaction_capture.delay_ms;
        timing_summary.input_peak = interaction_capture.input_peak;
        timing_summary.mean_square = interaction_capture.mean_square;
        timing_summary.detected_input_frame_absolute =
            interaction_capture.detected_input_frame_absolute;
        timing_summary.detected_input_adc_time =
            interaction_capture.detected_input_adc_time;
        timing_summary.first_tone_output_frame_absolute =
            interaction_capture.first_tone_output_frame_absolute;
        timing_summary.first_tone_output_dac_time =
            interaction_capture.first_tone_output_dac_time;
        timing_summary.timestamp_corrected_delay_samples =
            interaction_capture.timestamp_corrected_delay_samples;
        timing_summary.timestamp_corrected_delay_ms =
            interaction_capture.timestamp_corrected_delay_ms;
        timing_summary.delay_correction_delta_samples =
            interaction_capture.delay_correction_delta_samples;
        timing_summary.delay_correction_delta_ms =
            interaction_capture.delay_correction_delta_ms;
        timing_summary.callback_count = callback_stats.callback_count;
        timing_summary.priming_callback_count =
            callback_stats.priming_callback_count;
        timing_summary.first_output_minus_input_samples =
            callback_stats.first_output_minus_input_samples;
        timing_summary.min_output_minus_input_samples =
            callback_stats.min_output_minus_input_samples;
        timing_summary.max_output_minus_input_samples =
            callback_stats.max_output_minus_input_samples;
        timing_summary.avg_output_minus_input_samples =
            callback_stats.avg_output_minus_input_samples;
        timing_summary.input_underflow_count =
            interaction_capture.input_underflow_count;
        timing_summary.input_overflow_count =
            interaction_capture.input_overflow_count;
        timing_summary.output_underflow_count =
            interaction_capture.output_underflow_count;
        timing_summary.output_overflow_count =
            interaction_capture.output_overflow_count;
        timing_summary.stream_input_latency_seconds =
            stream_info.input_latency_seconds;
        timing_summary.stream_output_latency_seconds =
            stream_info.output_latency_seconds;
        timing_summary.stream_sample_rate =
            stream_info.sample_rate;
        timing_summaries.push_back(timing_summary);
        timing_summary_dirty = true;
        callback_trace_dirty = true;

        std::cout << "rep=" << repetition
                  << " selected_input_channel=" << (INPUT_CHANNEL_INDEX + 1)
                  << " delay_samples=" << interaction_capture.delay_samples
                  << " delay_ms=" << interaction_capture.delay_ms
                  << " input_peak=" << interaction_capture.input_peak
                  << " selected_input_mean_square="
                  << interaction_capture.mean_square
                  << " raw_input_peak=" << max_absolute_value(interaction_response)
                  << " corrected_delay_samples="
                  << interaction_capture.timestamp_corrected_delay_samples
                  << " corrected_delay_ms="
                  << interaction_capture.timestamp_corrected_delay_ms
                  << " delay_correction_delta_ms="
                  << interaction_capture.delay_correction_delta_ms
                  << " cb_out_minus_in_first_samples="
                  << callback_stats.first_output_minus_input_samples
                  << " cb_out_minus_in_avg_samples="
                  << callback_stats.avg_output_minus_input_samples
                  << " priming_callbacks=" << callback_stats.priming_callback_count
                  << " in_under=" << interaction_capture.input_underflow_count
                  << " in_over=" << interaction_capture.input_overflow_count
                  << " out_under=" << interaction_capture.output_underflow_count
                  << " out_over=" << interaction_capture.output_overflow_count
                  << '\n';

        if (interaction_capture.delay_samples < 0)
        {
            std::cerr << "No valid delay was detected for repetition "
                      << repetition << ".\n";
        }
        else
        {
            RankedResponse ranked_response;
            ranked_response.raw_delay_ms = interaction_capture.delay_ms;
            ranked_response.corrected_delay_ms =
                interaction_capture.timestamp_corrected_delay_ms;
            ranked_response.normalized_response = normalized_response;
            delay_ranked_responses.push_back(std::move(ranked_response));
            std::sort(
                delay_ranked_responses.begin(),
                delay_ranked_responses.end(),
                [](const auto &lhs, const auto &rhs) {
                    return lhs.raw_delay_ms < rhs.raw_delay_ms;
                });
            selected_delays_dirty = true;
            distribution_dirty = true;
        }

        if ((repetition % SAVE_EVERY_INTERACTIONS) == 0)
        {
            flush_saved_csvs(true);
        }

        if (g_keep_running.load())
        {
            std::this_thread::sleep_for(
            std::chrono::duration<double>(INTERACTION_PAUSE_SECONDS));
        }
    }

    flush_saved_csvs(true);

    if (stream)
    {
        Pa_CloseStream(stream);
    }
    Pa_Terminate();
    return 0;
}
