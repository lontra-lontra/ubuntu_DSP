#include "portable/mockportaudio.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace
{

bool g_initialized = false;
std::vector<MockDevice> g_devices;

PaDeviceIndex stream_device_index(
    const PaStream *stream)
{
    if (!stream)
    {
        return paNoDevice;
    }

    return stream->hasOutputParams ? stream->outputParams.device
                                   : stream->inputParams.device;
}

PaError validate_stream_parameters(
    const PaStreamParameters *params,
    bool is_input)
{
    if (!params)
    {
        return paNoError;
    }

    if (!g_initialized)
    {
        return paNotInitialized;
    }

    if (params->device < 0 ||
        params->device >= static_cast<PaDeviceIndex>(g_devices.size()))
    {
        return paInvalidDevice;
    }

    const MockDevice &device = g_devices[static_cast<size_t>(params->device)];
    const int max_channels =
        is_input ? device.info.maxInputChannels : device.info.maxOutputChannels;

    if (params->channelCount <= 0 || params->channelCount > max_channels)
    {
        return paInvalidChannelCount;
    }

    if (params->sampleFormat != device.supportedFormat)
    {
        return paInvalidSampleFormat;
    }

    return paNoError;
}

void capture_output_if_needed(
    const PaStream *stream,
    const std::vector<float> &output_buffer,
    unsigned long frames_per_buffer)
{
    if (!stream->hasOutputParams)
    {
        return;
    }

    MockDevice &device = g_devices[static_cast<size_t>(stream->outputParams.device)];
    const int channel_count = stream->outputParams.channelCount;
    if (device.outputHistory.size() < static_cast<size_t>(channel_count))
    {
        device.outputHistory.assign(
            static_cast<size_t>(channel_count),
            std::vector<float>());
    }

    for (unsigned long frame = 0; frame < frames_per_buffer; ++frame)
    {
        for (int ch = 0; ch < channel_count; ++ch)
        {
            device.outputHistory[static_cast<size_t>(ch)].push_back(
                output_buffer[static_cast<size_t>(frame * channel_count + ch)]);
        }
    }
}

void reset_output_history_for_device(
    MockDevice &device,
    int channel_count,
    const char *phase)
{
    const int safe_channel_count = std::max(0, channel_count);
    device.outputHistory.assign(
        static_cast<size_t>(safe_channel_count),
        std::vector<float>());

    std::cout << "[mock] " << phase
              << " stream for device "
              << (device.info.name ? device.info.name : "(null)")
              << ": reset output history\n";
}

void flush_pending_stream_closed_callback_if_needed(
    MockDevice &device)
{
    if (!device.hasPendingStreamClosed)
    {
        return;
    }

    if (device.onStreamClosed)
    {
        device.onStreamClosed(device, device.pendingStreamClosedSampleRate);
    }

    device.hasPendingStreamClosed = false;
    device.pendingStreamClosedSampleRate = 0.0;
    reset_output_history_for_device(
        device,
        static_cast<int>(device.outputHistory.size()),
        "ending");
}

void mark_stream_closed_callback_pending_if_needed(
    const PaStream *stream)
{
    const PaDeviceIndex device_index = stream_device_index(stream);
    if (device_index < 0 ||
        device_index >= static_cast<PaDeviceIndex>(g_devices.size()))
    {
        return;
    }

    MockDevice &device = g_devices[static_cast<size_t>(device_index)];
    if (!device.onStreamClosed)
    {
        if (stream->hasOutputParams)
        {
            reset_output_history_for_device(
                device,
                stream->outputParams.channelCount,
                "ending");
        }
        return;
    }

    device.hasPendingStreamClosed = true;
    device.pendingStreamClosedSampleRate = stream->sampleRate;
}

void reset_output_history_if_needed(
    const PaStream *stream,
    const char *phase)
{
    if (!stream)
    {
        return;
    }

    if (stream->hasOutputParams)
    {
        MockDevice &device = g_devices[static_cast<size_t>(stream->outputParams.device)];
        flush_pending_stream_closed_callback_if_needed(device);
        reset_output_history_for_device(
            device,
            stream->outputParams.channelCount,
            phase);
    }
}

} // namespace

std::vector<MockDevice> *mock_pa_get_devices_list()
{
    return &g_devices;
}

PaError Pa_Initialize()
{
    g_initialized = true;
    return paNoError;
}

PaError Pa_Terminate()
{
    bool ran_deferred_handlers = false;
    for (MockDevice &device : g_devices)
    {
        if (!device.hasPendingStreamClosed)
        {
            continue;
        }

        if (!ran_deferred_handlers)
        {
            std::cout << "[mock] running deferred stream-closed handlers\n";
            ran_deferred_handlers = true;
        }

        flush_pending_stream_closed_callback_if_needed(device);
    }

    if (ran_deferred_handlers)
    {
        std::cout << "[mock] deferred stream-closed handlers finished\n";
    }

    g_initialized = false;
    return paNoError;
}

PaDeviceIndex Pa_GetDeviceCount()
{
    if (!g_initialized)
    {
        return paNotInitialized;
    }

    return static_cast<PaDeviceIndex>(g_devices.size());
}

const PaDeviceInfo *Pa_GetDeviceInfo(PaDeviceIndex device)
{
    if (!g_initialized)
    {
        return nullptr;
    }

    if (device < 0 || device >= static_cast<PaDeviceIndex>(g_devices.size()))
    {
        return nullptr;
    }

    return &g_devices[static_cast<size_t>(device)].info;
}

const char *Pa_GetErrorText(PaError error)
{
    switch (error)
    {
    case paNoError: return "No error";
    case paNotInitialized: return "PortAudio mock not initialized";
    case paInvalidDevice: return "Invalid device";
    case paInvalidChannelCount: return "Invalid channel count";
    case paInvalidSampleRate: return "Invalid sample rate";
    case paInvalidSampleFormat: return "Invalid sample format";
    case paBadStreamPtr: return "Bad stream pointer";
    case paInvalidFlag: return "Invalid flag";
    default: return "Unknown PortAudio mock error";
    }
}

PaDeviceIndex Pa_GetDefaultInputDevice()
{
    if (!g_initialized)
    {
        return paNotInitialized;
    }

    for (size_t i = 0; i < g_devices.size(); ++i)
    {
        if (g_devices[i].info.maxInputChannels > 0)
        {
            return static_cast<PaDeviceIndex>(i);
        }
    }

    return paNoDevice;
}

PaDeviceIndex Pa_GetDefaultOutputDevice()
{
    if (!g_initialized)
    {
        return paNotInitialized;
    }

    for (size_t i = 0; i < g_devices.size(); ++i)
    {
        if (g_devices[i].info.maxOutputChannels > 0)
        {
            return static_cast<PaDeviceIndex>(i);
        }
    }

    return paNoDevice;
}

PaError Pa_IsFormatSupported(
    const PaStreamParameters *inputParameters,
    const PaStreamParameters *outputParameters,
    double sampleRate)
{
    if (!g_initialized)
    {
        return paNotInitialized;
    }

    if (sampleRate <= 0.0)
    {
        return paInvalidSampleRate;
    }

    const PaError input_err = validate_stream_parameters(inputParameters, true);
    if (input_err != paNoError)
    {
        return input_err;
    }

    const PaError output_err = validate_stream_parameters(outputParameters, false);
    if (output_err != paNoError)
    {
        return output_err;
    }

    auto sample_rate_is_supported =
        [sampleRate](const PaStreamParameters *params) -> bool
    {
        if (!params)
        {
            return true;
        }
        const MockDevice &device = g_devices[static_cast<size_t>(params->device)];
        for (double rate : device.supportedSampleRates)
        {
            if (std::fabs(rate - sampleRate) < 1e-6)
            {
                return true;
            }
        }
        return device.supportedSampleRates.empty();
    };

    if (!sample_rate_is_supported(inputParameters) ||
        !sample_rate_is_supported(outputParameters))
    {
        return paInvalidSampleRate;
    }

    return paFormatIsSupported;
}

PaError Pa_OpenStream(
    PaStream **stream,
    const PaStreamParameters *inputParameters,
    const PaStreamParameters *outputParameters,
    double sampleRate,
    unsigned long framesPerBuffer,
    PaStreamFlags,
    PaStreamCallback *streamCallback,
    void *userData)
{
    if (!g_initialized)
    {
        return paNotInitialized;
    }

    if (!stream || !streamCallback)
    {
        return paBadStreamPtr;
    }

    const PaError format_err =
        Pa_IsFormatSupported(inputParameters, outputParameters, sampleRate);
    if (format_err != paFormatIsSupported)
    {
        return format_err;
    }

    PaStream *opened_stream = new PaStream();
    opened_stream->hasInputParams = (inputParameters != nullptr);
    opened_stream->hasOutputParams = (outputParameters != nullptr);
    if (inputParameters)
    {
        opened_stream->inputParams = *inputParameters;
    }
    if (outputParameters)
    {
        opened_stream->outputParams = *outputParameters;
    }
    opened_stream->sampleRate = sampleRate;
    opened_stream->framesPerBuffer = framesPerBuffer == 0 ? 64ul : framesPerBuffer;
    opened_stream->callback = streamCallback;
    opened_stream->userData = userData;
    opened_stream->active = false;

    *stream = opened_stream;
    return paNoError;
}

PaError Pa_StartStream(PaStream *stream)
{
    if (!stream)
    {
        return paBadStreamPtr;
    }

    stream->active = true;
    reset_output_history_if_needed(stream, "starting");

    const int input_channels =
        stream->hasInputParams ? stream->inputParams.channelCount : 0;
    const int output_channels =
        stream->hasOutputParams ? stream->outputParams.channelCount : 0;

    std::vector<float> input_buffer(
        static_cast<size_t>(stream->framesPerBuffer) * static_cast<size_t>(input_channels),
        0.0f);
    std::vector<float> output_buffer(
        static_cast<size_t>(stream->framesPerBuffer) * static_cast<size_t>(output_channels),
        0.0f);

    PaStreamCallbackTimeInfo time_info{};
    unsigned long callback_count = 0;
    size_t previous_status_line_length = 0;

    while (stream->active)
    {
        const double current_time =
            static_cast<double>(callback_count * stream->framesPerBuffer) / stream->sampleRate;
        time_info.currentTime = current_time;
        time_info.inputBufferAdcTime = current_time;
        time_info.outputBufferDacTime = current_time;

        if (stream->hasInputParams)
        {
            const MockDevice &device =
                g_devices[static_cast<size_t>(stream->inputParams.device)];
            for (unsigned long frame = 0; frame < stream->framesPerBuffer; ++frame)
            {
                const double frame_time =
                    current_time +
                    static_cast<double>(frame) / stream->sampleRate;
                for (int ch = 0; ch < input_channels; ++ch)
                {
                    float sample = 0.0f;
                    if (ch < static_cast<int>(device.inputGenerators.size()) &&
                        device.inputGenerators[static_cast<size_t>(ch)])
                    {
                        sample = device.inputGenerators[static_cast<size_t>(ch)](frame_time);
                    }
                    input_buffer[static_cast<size_t>(frame * input_channels + ch)] = sample;
                }
            }
        }
        std::fill(output_buffer.begin(), output_buffer.end(), 0.0f);
        const auto callback_start = std::chrono::steady_clock::now();
        const int callback_result =
            stream->callback(
                input_buffer.empty() ? nullptr : input_buffer.data(),
                output_buffer.empty() ? nullptr : output_buffer.data(),
                stream->framesPerBuffer,
                &time_info,
                0,
                stream->userData);
        const auto callback_end = std::chrono::steady_clock::now();

        capture_output_if_needed(stream, output_buffer, stream->framesPerBuffer);
        callback_count++;
        const double callback_delay_seconds =
            std::chrono::duration<double>(callback_end - callback_start).count();
        const double allowed_delay_seconds =
            static_cast<double>(stream->framesPerBuffer) / stream->sampleRate;
        const double delay_ratio_percent =
            allowed_delay_seconds > 0.0
                ? 100.0 * callback_delay_seconds / allowed_delay_seconds
                : 0.0;

        std::ostringstream status_builder;
        status_builder << std::fixed << std::setprecision(6)
                       << "[mock] callback " << callback_count
                       << " t=" << current_time << "s"
                       << " delay=" << callback_delay_seconds << "s"
                       << " allowed=" << allowed_delay_seconds << "s"
                       << " load=" << delay_ratio_percent << "%";
        std::string status_line = status_builder.str();
        if (status_line.size() < previous_status_line_length)
        {
            status_line.append(previous_status_line_length - status_line.size(), ' ');
        }

        std::cout << '\r' << status_line << std::flush;
        previous_status_line_length = status_line.size();

        if (callback_result == paAbort || callback_result == paComplete)
        {
            break;
        }
    }

    if (callback_count > 0)
    {
        std::cout << '\n';
    }

    std::cout << "[mock] callbacks finished\n";
    mark_stream_closed_callback_pending_if_needed(stream);
    stream->active = false;
    return paNoError;
}

PaError Pa_StopStream(PaStream *stream)
{
    if (!stream)
    {
        return paBadStreamPtr;
    }

    stream->active = false;
    return paNoError;
}

PaError Pa_CloseStream(PaStream *stream)
{
    if (!stream)
    {
        return paBadStreamPtr;
    }

    delete stream;
    return paNoError;
}

int Pa_IsStreamActive(PaStream *stream)
{
    if (!stream)
    {
        return 0;
    }
    return stream->active ? 1 : 0;
}

void Pa_Sleep(long milliseconds)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}
