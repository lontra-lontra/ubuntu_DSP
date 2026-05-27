#pragma once

#include "portable/mock_device_registry.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#ifndef CHANNELS
#define CHANNELS 2
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 48000
#endif

#ifndef PORTABLE_OUTPUT_DIR
#define PORTABLE_OUTPUT_DIR "output"
#endif

#ifndef CSV_TO_INFER_TOPOLOGY_FROM
#define CSV_TO_INFER_TOPOLOGY_FROM "topology_to_infer_from"
#endif

inline void register_mock_devices()
{
    clear_mock_devices();

    const auto build_arbitrary_signal_list = []() -> std::vector<MockInputGenerator>
    {
        std::vector<MockInputGenerator> signals;
        signals.reserve(CHANNELS);

        constexpr double kPi = 3.14159265358979323846;
        for (int channel = 0; channel < CHANNELS; ++channel)
        {
            const double f0 = 110.0 + 11.0 * static_cast<double>(channel);
            const double f1 = 30.0 + 3.0 * static_cast<double>(channel % 5);
            const double phase = 0.2 * static_cast<double>(channel);
            const double amplitude = 0.4 + 0.01 * static_cast<double>(channel);
            signals.push_back([f0, f1, phase, amplitude, kPi](double t) -> float
            {
                return static_cast<float>(
                    amplitude * std::sin(2.0 * kPi * f0 * t + phase) +
                    0.15 * std::sin(2.0 * kPi * f1 * t));
            });
        }

        return signals;
    };

    const std::vector<double> supported_sample_rates = {
        8000.0, 16000.0, 44100.0, 48000.0};

    register_mock_signal_generator_device(
        "portable_mock_device",
        CHANNELS,
        CHANNELS,
        SAMPLE_RATE,
        supported_sample_rates,
        build_arbitrary_signal_list());

    const std::string topology_csv_path =
        std::string(PORTABLE_OUTPUT_DIR) + "/" +
        CSV_TO_INFER_TOPOLOGY_FROM + ".csv";
    try
    {
        register_mock_device_inferred_from_topology(
            "infered_from_topology",
            topology_csv_path.c_str(),
            CHANNELS,
            CHANNELS,
            SAMPLE_RATE,
            supported_sample_rates);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Could not register inferred topology mock device from "
                  << topology_csv_path << ": " << ex.what() << '\n';
    }
}
