/*
Build from repo root:

  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=test_installation -DPORTABLE_USE_MOCK=OFF
  cmake --build Portable/build --target portable_test_installation --parallel

Run:
  ./Portable/build/portable_test_installation

Important config:
  local device config in this file:
  #define DEVICE_NAME ...
  #define CHANNELS ...
  #define SAMPLE_RATE ...
  #define FRAMES_PER_BUFFER ...
  #define SAMPLE_FORMAT ...

This app performs one sweep over i = 1..8. For each i it plays an 800 Hz beep
with amplitude 0.8 for 0.5 seconds, repeated i times with 0.5 seconds of
silence between beeps, on output channels i, i+8, i+16, and i+24. For each
output-channel test it measures the mean square on all input channels during the
beep windows and prints the three quietest inputs as a percentage of the
loudest input in that same test.
*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
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



// Local sound-device config for this app.
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

#ifndef BEEP_SECONDS
#define BEEP_SECONDS 0.2
#endif

#ifndef SILENCE_SECONDS
#define SILENCE_SECONDS 0.3
#endif

#ifndef TONE_FREQUENCY_HZ
#define TONE_FREQUENCY_HZ 800.0
#endif

#ifndef TONE_AMPLITUDE
#define TONE_AMPLITUDE 0.4
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
constexpr int kInputChannelCount = 32;
constexpr int kOutputChannelCount = 32;
constexpr int kLoopStart = 1;
constexpr int kLoopEnd = 8;
constexpr int kOutputStride = 8;
constexpr int kReportedQuietInputs = 3;

struct SegmentPlan
{
    int loop_index_one_based = 0;
    int output_channel_zero_based = 0;
    int repetition_count = 0;
    int total_frames = 0;
};

struct SegmentAccum
{
    std::vector<double> sum_squares;
    int active_beep_frames = 0;
};

struct TestInstallationCallbackData
{
    const std::vector<SegmentPlan> *plans = nullptr;
    std::vector<SegmentAccum> *results = nullptr;
    int current_segment_index = 0;
    int frame_offset_in_segment = 0;
    int total_frames_processed = 0;
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
};

struct RankedInput
{
    int channel_zero_based = 0;
    double mean_square = 0.0;
};

void print_requested_config()
{
    std::cout << "Requested config:"
              << " DEVICE_NAME=" << DEVICE_NAME
              << " CHANNELS=" << CHANNELS
              << " SAMPLE_RATE=" << SAMPLE_RATE
              << " FRAMES_PER_BUFFER=" << FRAMES_PER_BUFFER
              << " BEEP_SECONDS=" << BEEP_SECONDS
              << " SILENCE_SECONDS=" << SILENCE_SECONDS
              << " TONE_FREQUENCY_HZ=" << TONE_FREQUENCY_HZ
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

std::vector<SegmentPlan> build_segment_plan(
    int beep_frames,
    int silence_frames)
{
    std::vector<SegmentPlan> plans;
        for (int i = kLoopStart; i <= kLoopEnd; ++i)
    {
        for (int base_channel = i - 1;
             base_channel < kOutputChannelCount;
             base_channel += kOutputStride)
        {
            SegmentPlan plan;
            plan.loop_index_one_based = i;
            plan.output_channel_zero_based = base_channel;
            plan.repetition_count = i;
            plan.total_frames = i * (beep_frames + silence_frames);
            plans.push_back(plan);
        }
    }
    return plans;
}

int test_installation_callback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    TestInstallationCallbackData *data =
        static_cast<TestInstallationCallbackData *>(userData);
    const float *input = static_cast<const float *>(inputBuffer);
    float *output = static_cast<float *>(outputBuffer);

    if (!data || !data->plans || !data->results || !output)
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

    const int beep_frames = static_cast<int>(std::llround(
        BEEP_SECONDS * static_cast<double>(SAMPLE_RATE)));
    const int silence_frames = static_cast<int>(std::llround(
        SILENCE_SECONDS * static_cast<double>(SAMPLE_RATE)));
    const int cycle_frames = beep_frames + silence_frames;

    for (unsigned long frame = 0; frame < framesPerBuffer; ++frame)
    {
        std::fill(
            output + frame * kOutputChannelCount,
            output + (frame + 1) * kOutputChannelCount,
            0.0f);

        if (data->current_segment_index >= static_cast<int>(data->plans->size()))
        {
            continue;
        }

        const SegmentPlan &plan =
            (*data->plans)[static_cast<size_t>(data->current_segment_index)];
        SegmentAccum &result =
            (*data->results)[static_cast<size_t>(data->current_segment_index)];

        const int frame_in_cycle =
            cycle_frames > 0 ? data->frame_offset_in_segment % cycle_frames : 0;
        const bool is_beep_frame = frame_in_cycle < beep_frames;

        float tone_sample = 0.0f;
        if (is_beep_frame)
        {
            const double t =
                static_cast<double>(data->total_frames_processed) /
                static_cast<double>(SAMPLE_RATE);
            tone_sample = static_cast<float>(
                TONE_AMPLITUDE * std::sin(2.0 * kPi * TONE_FREQUENCY_HZ * t));
            output[frame * kOutputChannelCount + plan.output_channel_zero_based] =
                tone_sample;

            for (int ch = 0; ch < kInputChannelCount; ++ch)
            {
                const float sample =
                    input ? input[frame * kInputChannelCount + ch] : 0.0f;
                result.sum_squares[static_cast<size_t>(ch)] +=
                    static_cast<double>(sample) * static_cast<double>(sample);
            }
            result.active_beep_frames++;
        }

        data->frame_offset_in_segment++;
        data->total_frames_processed++;

        if (data->frame_offset_in_segment >= plan.total_frames)
        {
            data->current_segment_index++;
            data->frame_offset_in_segment = 0;
        }
    }

    return data->current_segment_index >= static_cast<int>(data->plans->size())
               ? paComplete
               : paContinue;
}

std::vector<RankedInput> rank_inputs_by_mean_square(
    const SegmentAccum &result)
{
    std::vector<RankedInput> ranked;
    ranked.reserve(kInputChannelCount);

    for (int ch = 0; ch < kInputChannelCount; ++ch)
    {
        double mean_square = 0.0;
        if (result.active_beep_frames > 0)
        {
            mean_square =
                result.sum_squares[static_cast<size_t>(ch)] /
                static_cast<double>(result.active_beep_frames);
        }

        ranked.push_back(RankedInput{
            ch,
            mean_square
        });
    }

    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const RankedInput &lhs, const RankedInput &rhs)
        {
            if (lhs.mean_square != rhs.mean_square)
            {
                return lhs.mean_square < rhs.mean_square;
            }
            return lhs.channel_zero_based < rhs.channel_zero_based;
        });

    return ranked;
}

double highest_mean_square(const SegmentAccum &result)
{
    double highest = 0.0;
    for (double sum_square : result.sum_squares)
    {
        const double mean_square =
            result.active_beep_frames > 0
                ? sum_square / static_cast<double>(result.active_beep_frames)
                : 0.0;
        highest = std::max(highest, mean_square);
    }
    return highest;
}

} // namespace

int main()
{
    if (CHANNELS < kInputChannelCount || CHANNELS < kOutputChannelCount)
    {
        std::cerr << "CHANNELS must be at least " << kInputChannelCount
                  << " for this installation test.\n";
        return 1;
    }
    if (FRAMES_PER_BUFFER <= 0)
    {
        std::cerr << "FRAMES_PER_BUFFER must be positive.\n";
        return 1;
    }

    const int beep_frames = static_cast<int>(std::llround(
        BEEP_SECONDS * static_cast<double>(SAMPLE_RATE)));
    const int silence_frames = static_cast<int>(std::llround(
        SILENCE_SECONDS * static_cast<double>(SAMPLE_RATE)));
    if (beep_frames <= 0)
    {
        std::cerr << "BEEP_SECONDS * SAMPLE_RATE must be positive.\n";
        return 1;
    }
    if (silence_frames < 0)
    {
        std::cerr << "SILENCE_SECONDS * SAMPLE_RATE must not be negative.\n";
        return 1;
    }

    const std::vector<SegmentPlan> plans =
        build_segment_plan(beep_frames, silence_frames);
    if (plans.empty())
    {
        std::cerr << "No output-channel tests were generated.\n";
        return 1;
    }

    std::vector<SegmentAccum> results;
    results.reserve(plans.size());
    for (size_t index = 0; index < plans.size(); ++index)
    {
        SegmentAccum result;
        result.sum_squares.assign(static_cast<size_t>(kInputChannelCount), 0.0);
        results.push_back(result);
    }

    int total_frames = 0;
    for (const SegmentPlan &plan : plans)
    {
        total_frames += plan.total_frames;
    }

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

    if (device_info->maxInputChannels < kInputChannelCount)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested installation test requires 32 input channels",
            paInvalidChannelCount);
        Pa_Terminate();
        return 1;
    }
    if (device_info->maxOutputChannels < kOutputChannelCount)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested installation test requires 32 output channels",
            paInvalidChannelCount);
        Pa_Terminate();
        return 1;
    }

    PaStreamParameters input_parameters{};
    input_parameters.device = device_index;
    input_parameters.channelCount = kInputChannelCount;
    input_parameters.sampleFormat = SAMPLE_FORMAT;
    input_parameters.suggestedLatency = std::max(
        device_info->defaultLowInputLatency,
        static_cast<double>(FRAMES_PER_BUFFER) /
            static_cast<double>(SAMPLE_RATE));
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = kOutputChannelCount;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = std::max(
        device_info->defaultLowOutputLatency,
        static_cast<double>(FRAMES_PER_BUFFER) /
            static_cast<double>(SAMPLE_RATE));
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    const PaError format_error = Pa_IsFormatSupported(
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE);
    if (format_error != paFormatIsSupported)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested test configuration is not supported",
            format_error);
        Pa_Terminate();
        return 1;
    }

    TestInstallationCallbackData data{};
    data.plans = &plans;
    data.results = &results;

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        &input_parameters,
        &output_parameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff,
        test_installation_callback,
        &data);
    if (open_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_OpenStream failed for the installation test",
            open_error);
        Pa_Terminate();
        return 1;
    }

    std::cout << "Running installation test over " << plans.size()
              << " output cases (" << total_frames << " frames total, about "
              << std::fixed << std::setprecision(1)
              << static_cast<double>(total_frames) / static_cast<double>(SAMPLE_RATE)
              << " seconds)...\n";

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_StartStream failed for the installation test",
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

    const PaError stop_error = Pa_StopStream(stream);
    if (stop_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_StopStream failed after the installation test",
            stop_error);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << std::defaultfloat;
    std::cout << "Results:\n";
    for (size_t index = 0; index < plans.size(); ++index)
    {
        const SegmentPlan &plan = plans[index];
        const SegmentAccum &result = results[index];
        const std::vector<RankedInput> ranked =
            rank_inputs_by_mean_square(result);
        const double highest = highest_mean_square(result);

        std::cout << "i=" << plan.loop_index_one_based
                  << " output_ch=" << (plan.output_channel_zero_based + 1)
                  << " quietest_inputs=";

        for (int rank = 0; rank < kReportedQuietInputs; ++rank)
        {
            const RankedInput &entry = ranked[static_cast<size_t>(rank)];
            const double percentage =
                highest > 0.0 ? 100.0 * entry.mean_square / highest : 0.0;
            if (rank > 0)
            {
                std::cout << ", ";
            }
            std::cout << "ch" << (entry.channel_zero_based + 1)
                      << " meanSq=" << std::scientific << std::setprecision(6)
                      << entry.mean_square
                      << " (" << std::fixed << std::setprecision(3)
                      << percentage << "% of max)";
        }

        std::cout << std::defaultfloat
                  << " maxMeanSq=" << std::scientific << std::setprecision(6)
                  << highest
                  << std::defaultfloat
                  << '\n';
    }

    std::cout << "Captured frames=" << data.total_frames_processed
              << " inputUnderflows=" << data.input_underflow_count
              << " inputOverflows=" << data.input_overflow_count
              << " outputUnderflows=" << data.output_underflow_count
              << " outputOverflows=" << data.output_overflow_count
              << '\n';

    return 0;
}
