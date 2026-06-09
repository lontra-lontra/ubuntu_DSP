#pragma once

#ifndef MOCK
#error "MOCK must be defined by the build system before including portable/stream_status.h"
#endif

#if MOCK
#include "portable/mockportaudio.h"
#else
#include <portaudio.h>
#endif

#include <sstream>
#include <string>

struct PortableStreamStatusSummary
{
    int input_underflow_count = 0;
    int input_overflow_count = 0;
    int output_underflow_count = 0;
    int output_overflow_count = 0;
    int priming_output_count = 0;
    int callback_warning_count = 0;

    void observe(PaStreamCallbackFlags status_flags)
    {
        if (status_flags == 0)
        {
            return;
        }

        callback_warning_count++;

        if ((status_flags & paInputUnderflow) != 0)
        {
            input_underflow_count++;
        }
        if ((status_flags & paInputOverflow) != 0)
        {
            input_overflow_count++;
        }
        if ((status_flags & paOutputUnderflow) != 0)
        {
            output_underflow_count++;
        }
        if ((status_flags & paOutputOverflow) != 0)
        {
            output_overflow_count++;
        }
        if ((status_flags & paPrimingOutput) != 0)
        {
            priming_output_count++;
        }
    }

    void accumulate_from(const PortableStreamStatusSummary &other)
    {
        input_underflow_count += other.input_underflow_count;
        input_overflow_count += other.input_overflow_count;
        output_underflow_count += other.output_underflow_count;
        output_overflow_count += other.output_overflow_count;
        priming_output_count += other.priming_output_count;
        callback_warning_count += other.callback_warning_count;
    }

    bool has_any_warnings() const
    {
        return callback_warning_count != 0;
    }
};

inline std::string portable_describe_status_flags(PaStreamCallbackFlags status_flags)
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

    if ((status_flags & paInputUnderflow) != 0)
    {
        append_flag("input underflow");
    }
    if ((status_flags & paInputOverflow) != 0)
    {
        append_flag("input overflow");
    }
    if ((status_flags & paOutputUnderflow) != 0)
    {
        append_flag("output underflow");
    }
    if ((status_flags & paOutputOverflow) != 0)
    {
        append_flag("output overflow");
    }
    if ((status_flags & paPrimingOutput) != 0)
    {
        append_flag("output priming");
    }

    if (description.empty())
    {
        description = "unknown callback status flag issue";
    }

    return description;
}

inline std::string portable_format_status_summary(
    const PortableStreamStatusSummary &summary)
{
    std::ostringstream stream;
    stream << "callbackWarnings=" << summary.callback_warning_count
           << " inputUnderflows=" << summary.input_underflow_count
           << " inputOverflows=" << summary.input_overflow_count
           << " outputUnderflows=" << summary.output_underflow_count
           << " outputOverflows=" << summary.output_overflow_count
           << " outputPriming=" << summary.priming_output_count;
    return stream.str();
}
