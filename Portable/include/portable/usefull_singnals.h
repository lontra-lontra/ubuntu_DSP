#pragma once

#include <cmath>

inline float portable_classic_chirp(
    float t,
    float total_time,
    float padding_time,
    float start_frequency = 20.0f,
    float end_frequency = 20000.0f,
    float window_time = 0.0f)
{
    constexpr float kPi = 3.14159265358979323846f;

    if (total_time <= 0.0f ||
        start_frequency < 0.0f ||
        end_frequency < 0.0f ||
        window_time < 0.0f)
    {
        return 0.0f;
    }

    if (padding_time < 0.0f || 2.0f * padding_time >= total_time)
    {
        return 0.0f;
    }

    if (t < 0.0f || t >= total_time)
    {
        return 0.0f;
    }

    if (t < padding_time || t >= total_time - padding_time)
    {
        return 0.0f;
    }

    const float active_time = total_time - 2.0f * padding_time;
    if (2.0f * window_time > active_time)
    {
        return 0.0f;
    }

    const float local_t = t - padding_time;
    const float sweep_rate = (end_frequency - start_frequency) / active_time;
    const float phase =
        2.0f * kPi *
        (start_frequency * local_t + 0.5f * sweep_rate * local_t * local_t);
    float amplitude = 1.0f;

    if (window_time > 0.0f)
    {
        if (local_t < window_time)
        {
            amplitude =
                0.5f - 0.5f * std::cos(kPi * local_t / window_time);
        }
        else if (local_t > active_time - window_time)
        {
            const float fade_out_t = (active_time - local_t) / window_time;
            amplitude =
                0.5f - 0.5f * std::cos(kPi * fade_out_t);
        }
    }

    return amplitude * std::sin(phase);
}
