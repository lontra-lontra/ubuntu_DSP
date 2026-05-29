#pragma once

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef MOCK
#error "MOCK must be defined by the build system before including portable/audio_helpers.h"
#endif

#if MOCK
#include "portable/mockportaudio.h"
#else
#include <portaudio.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

inline std::string portable_ascii_lower(std::string s)
{
    std::transform(
        s.begin(),
        s.end(),
        s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline std::vector<double> get_possible_sample_rates(
    int device_index,
    bool input = true)
{
    std::vector<double> rates;
    const PaDeviceInfo *device_info = Pa_GetDeviceInfo(device_index);
    if (!device_info)
    {
        return rates;
    }

    const std::vector<double> standard_rates = {
        8000.0,
        11025.0,
        16000.0,
        22050.0,
        32000.0,
        44100.0,
        48000.0,
        96000.0,
        192000.0
    };

    PaStreamParameters params{};
    params.device = device_index;
    params.channelCount = std::max(
        1,
        std::min(
            input ? device_info->maxInputChannels : device_info->maxOutputChannels,
            2));
    params.sampleFormat = paFloat32;
    params.suggestedLatency =
        input ? device_info->defaultLowInputLatency : device_info->defaultLowOutputLatency;

    for (double rate : standard_rates)
    {
        const PaError err =
            input ? Pa_IsFormatSupported(&params, nullptr, rate)
                  : Pa_IsFormatSupported(nullptr, &params, rate);
        if (err == paFormatIsSupported)
        {
            rates.push_back(rate);
        }
    }

    return rates;
}

inline void list_all_devices()
{
    const int device_count = Pa_GetDeviceCount();
    if (device_count < 0)
    {
        std::cerr << "Pa_GetDeviceCount failed: " << device_count << '\n';
        return;
    }

    std::cout << "Available audio devices:\n";
    if (device_count == 0)
    {
        std::cout << "  (none)\n";
        return;
    }

    for (int i = 0; i < device_count; ++i)
    {
        const PaDeviceInfo *device_info = Pa_GetDeviceInfo(i);
        if (!device_info)
        {
            continue;
        }
#if !MOCK
        const PaHostApiInfo *host_api_info = Pa_GetHostApiInfo(device_info->hostApi);
#endif

        std::cout << "  [" << i << "] "
                  << (device_info->name ? device_info->name : "(null)")
                  << " | in=" << device_info->maxInputChannels
                  << " out=" << device_info->maxOutputChannels
                  << " defaultSR=" << device_info->defaultSampleRate
#if !MOCK
                  << " hostApi="
                  << (host_api_info && host_api_info->name ? host_api_info->name : "(null)")
#endif
                  << " | rates=";

        const std::vector<double> rates = get_possible_sample_rates(i, true);
        for (double rate : rates)
        {
            std::cout << rate << ' ';
        }
        std::cout << '\n';

    }
}

inline int select_device_by_name(
    const char *name,
    bool prefer_input_device = true)
{
    const int device_count = Pa_GetDeviceCount();
    if (device_count < 0)
    {
        return device_count;
    }

    const std::string target_name = name ? std::string(name) : std::string();
    if (target_name.empty() || target_name == "default")
    {
        const int default_device = prefer_input_device ? Pa_GetDefaultInputDevice()
                                                       : Pa_GetDefaultOutputDevice();
        if (default_device == paNoDevice)
        {
            std::cerr << "No default "
                      << (prefer_input_device ? "input" : "output")
                      << " device is available.\n";
            list_all_devices();
        }
        return default_device;
    }

    for (int i = 0; i < device_count; ++i)
    {
        const PaDeviceInfo *device_info = Pa_GetDeviceInfo(i);
        if (!device_info || !device_info->name)
        {
            continue;
        }
        if (std::strcmp(device_info->name, target_name.c_str()) == 0)
        {
            return i;
        }
    }

    std::cerr << "Device not found: " << target_name << '\n';
    list_all_devices();
    return -1;
}

inline bool check_if_device_respects_input_and_output_stream_specs(
    const PaStreamParameters *input_stream_parameters,
    const PaStreamParameters *output_stream_parameters,
    double sample_rate)
{
    const PaError err =
        Pa_IsFormatSupported(
            input_stream_parameters,
            output_stream_parameters,
            sample_rate);
    if (err == paFormatIsSupported)
    {
        return true;
    }

    std::cerr << "Pa_IsFormatSupported failed: " << Pa_GetErrorText(err) << '\n';
    return false;
}
