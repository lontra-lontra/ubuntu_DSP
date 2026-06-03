/*
Build from repo root:

  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=chega_de_saudade_stems_32_outputs -DPORTABLE_USE_MOCK=OFF
  cmake --build Portable/build --target portable_chega_de_saudade_stems_32_outputs --parallel

If `Portable/build` does not exist yet, came from another machine, or you changed
`PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` line before
the `cmake --build ...` line.

Run:
  ./Portable/build/portable_chega_de_saudade_stems_32_outputs
  ./Portable/build/portable_chega_de_saudade_stems_32_outputs --dry-run

This local-only app expects Demucs stems for:
  chega_de_saudade_waves_stems/raw/htdemucs_6s/Chega De Saudade/

It decodes each stem to stereo with `ffmpeg`, then routes:
  floor(32 / stem_count) consecutive outputs to each stem.
Within each stem block, left/right stereo pairs repeat across the assigned outputs.
Any remainder outputs stay silent.
*/

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <system_error>
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

#ifndef CHEGA_DE_SAUDADE_DEMUCS_MODEL
#define CHEGA_DE_SAUDADE_DEMUCS_MODEL "htdemucs_6s"
#endif

#ifndef CHEGA_DE_SAUDADE_TRACK_STEM_DIR
#define CHEGA_DE_SAUDADE_TRACK_STEM_DIR "Chega De Saudade"
#endif

#ifndef CHEGA_DE_SAUDADE_STEMS_RELATIVE_DIR
#define CHEGA_DE_SAUDADE_STEMS_RELATIVE_DIR "chega_de_saudade_waves_stems/raw"
#endif

#ifndef STEM_GAIN
#define STEM_GAIN 0.9f
#endif

#ifndef MIN_STEM_STREAM_FRAMES_PER_BUFFER
#define MIN_STEM_STREAM_FRAMES_PER_BUFFER 1024
#endif

#if MOCK
#include "portable/mockportaudio.h"
#include "portable/mock_devices.h"
#else
#include <portaudio.h>
#endif

#include "portable/audio_helpers.h"

namespace
{

std::atomic<bool> g_keep_running{true};

struct StemData
{
    std::string name;
    std::filesystem::path path;
    std::vector<float> samples;
    size_t frame_count = 0;
    int first_output_channel = -1;
    int last_output_channel = -1;
};

void handle_interrupt_signal(int)
{
    g_keep_running.store(false);
}

std::string shell_single_quote(const std::string &value)
{
    std::string quoted = "'";
    for (char c : value)
    {
        if (c == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted.push_back(c);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::filesystem::path normalize_candidate(
    const std::filesystem::path &base,
    const std::filesystem::path &relative)
{
    std::error_code ec;
    const std::filesystem::path combined = base / relative;
    const std::filesystem::path normalized =
        std::filesystem::weakly_canonical(combined, ec);
    if (!ec)
    {
        return normalized;
    }

    return combined.lexically_normal();
}

std::optional<std::filesystem::path> find_stem_directory(
    const std::filesystem::path &argv0)
{
    const std::filesystem::path relative_dir =
        std::filesystem::path(CHEGA_DE_SAUDADE_STEMS_RELATIVE_DIR) /
        CHEGA_DE_SAUDADE_DEMUCS_MODEL /
        CHEGA_DE_SAUDADE_TRACK_STEM_DIR;

    std::error_code ec;
    const std::filesystem::path current_candidate =
        normalize_candidate(std::filesystem::current_path(ec), relative_dir);
    if (!ec && std::filesystem::is_directory(current_candidate))
    {
        return current_candidate;
    }

    const std::filesystem::path exe_dir =
        std::filesystem::absolute(argv0, ec).parent_path();
    if (!ec)
    {
        const std::filesystem::path repo_root_candidate =
            normalize_candidate(exe_dir, std::filesystem::path("..") / ".." / relative_dir);
        if (std::filesystem::is_directory(repo_root_candidate))
        {
            return repo_root_candidate;
        }
    }

    return std::nullopt;
}

std::vector<std::filesystem::path> collect_stem_paths(
    const std::filesystem::path &stem_dir)
{
    std::vector<std::filesystem::path> paths;
    std::set<std::string> used_filenames;

    const std::array<const char *, 6> preferred_names = {
        "bass",
        "drums",
        "guitar",
        "piano",
        "vocals",
        "other"
    };

    for (const char *name : preferred_names)
    {
        for (const char *extension : {".wav", ".mp3"})
        {
            const std::filesystem::path candidate =
                stem_dir / (std::string(name) + extension);
            if (std::filesystem::is_regular_file(candidate))
            {
                paths.push_back(candidate);
                used_filenames.insert(candidate.filename().string());
                break;
            }
        }
    }

    std::vector<std::filesystem::path> extras;
    for (const auto &entry : std::filesystem::directory_iterator(stem_dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        const std::filesystem::path path = entry.path();
        const std::string extension =
            portable_ascii_lower(path.extension().string());
        if (extension != ".mp3" && extension != ".wav")
        {
            continue;
        }

        const std::string filename = path.filename().string();
        if (used_filenames.count(filename) != 0)
        {
            continue;
        }

        extras.push_back(path);
    }

    std::sort(
        extras.begin(),
        extras.end(),
        [](const std::filesystem::path &a, const std::filesystem::path &b)
        {
            return a.filename().string() < b.filename().string();
        });

    paths.insert(paths.end(), extras.begin(), extras.end());
    return paths;
}

int choose_playback_sample_rate(const PaDeviceInfo *device_info)
{
    if (device_info && device_info->defaultSampleRate > 0.0)
    {
        const long long rounded_rate =
            std::llround(device_info->defaultSampleRate);
        if (rounded_rate > 0 && rounded_rate <= static_cast<long long>(INT_MAX))
        {
            return static_cast<int>(rounded_rate);
        }
    }

    return SAMPLE_RATE;
}

unsigned long choose_stream_frames_per_buffer()
{
    return static_cast<unsigned long>(
        std::max(FRAMES_PER_BUFFER, MIN_STEM_STREAM_FRAMES_PER_BUFFER));
}

std::vector<float> decode_stem_to_stereo_samples(
    const std::filesystem::path &stem_path,
    int playback_sample_rate)
{
    const std::string command =
        "ffmpeg -v error -nostdin -i " +
        shell_single_quote(stem_path.string()) +
        " -f f32le -acodec pcm_f32le -ac 2 -ar " +
        std::to_string(playback_sample_rate) +
        " pipe:1";

    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
    {
        std::cerr << "popen failed for ffmpeg on " << stem_path
                  << ": errno=" << errno << '\n';
        return {};
    }

    std::vector<float> samples;
    std::array<float, 4096> chunk{};

    while (true)
    {
        const size_t sample_count =
            std::fread(chunk.data(), sizeof(float), chunk.size(), pipe);
        if (sample_count > 0)
        {
            samples.insert(
                samples.end(),
                chunk.begin(),
                chunk.begin() + static_cast<std::ptrdiff_t>(sample_count));
        }

        if (sample_count < chunk.size())
        {
            break;
        }
    }

    const int exit_code = pclose(pipe);
    if (exit_code != 0)
    {
        std::cerr << "ffmpeg failed while decoding " << stem_path
                  << " with exit code " << exit_code << '\n';
        return {};
    }

    return samples;
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

    print_selected_device_summary(device_index, device_info);
    std::cerr << "Requested config:"
              << " DEVICE_NAME=" << DEVICE_NAME
              << " CHANNELS=" << CHANNELS
              << " SAMPLE_RATE=" << SAMPLE_RATE
              << " FRAMES_PER_BUFFER=" << FRAMES_PER_BUFFER
              << '\n';

    std::cerr << "Available devices for comparison:\n";
    list_all_devices();
}

void print_routing_summary(
    const std::vector<StemData> &stems,
    int outputs_per_stem,
    int silent_outputs,
    int playback_sample_rate,
    unsigned long stream_frames_per_buffer)
{
    std::cout << "Found " << stems.size()
              << " stems. Routing " << outputs_per_stem
              << " outputs per stem.\n"
              << "Playback config: sample_rate=" << playback_sample_rate
              << " frames_per_buffer=" << stream_frames_per_buffer
              << '\n';

    for (const StemData &stem : stems)
    {
        std::cout << "  " << stem.name
                  << " -> outputs "
                  << (stem.first_output_channel + 1)
                  << "-"
                  << (stem.last_output_channel + 1)
                  << " | frames=" << stem.frame_count
                  << '\n';
    }

    if (silent_outputs > 0)
    {
        const int first_silent_output =
            stems.empty() ? 1 : stems.back().last_output_channel + 2;
        std::cout << "  silent -> outputs "
                  << first_silent_output
                  << "-"
                  << CHANNELS
                  << '\n';
    }
}

} // namespace

int main(int argc, char **argv)
{
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

    bool dry_run = false;
    for (int i = 1; i < argc; ++i)
    {
        if (argv[i] && std::string(argv[i]) == "--dry-run")
        {
            dry_run = true;
        }
    }

    const std::filesystem::path argv0 =
        argc > 0 && argv && argv[0] ? std::filesystem::path(argv[0])
                                    : std::filesystem::path();
    const auto stem_dir = find_stem_directory(argv0);
    if (!stem_dir)
    {
        std::cerr
            << "Could not find the local Chega De Saudade stem directory.\n"
            << "Expected something like:\n"
            << "  chega_de_saudade_waves_stems/raw/" << CHEGA_DE_SAUDADE_DEMUCS_MODEL
            << "/" << CHEGA_DE_SAUDADE_TRACK_STEM_DIR << "/\n";
        return 1;
    }

    const std::vector<std::filesystem::path> stem_paths =
        collect_stem_paths(*stem_dir);
    if (stem_paths.empty())
    {
        std::cerr << "No stem audio files were found under:\n  "
                  << *stem_dir << '\n';
        return 1;
    }

    bool portaudio_initialized = false;
    int device_index = -1;
    const PaDeviceInfo *device_info = nullptr;
    int playback_sample_rate = SAMPLE_RATE;
    const unsigned long stream_frames_per_buffer =
        choose_stream_frames_per_buffer();

    if (!dry_run)
    {
        const PaError init_error = Pa_Initialize();
        if (init_error != paNoError)
        {
            std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(init_error)
                      << '\n';
            return 1;
        }
        portaudio_initialized = true;

        device_index = select_device_by_name(DEVICE_NAME, false);
        if (device_index < 0)
        {
            Pa_Terminate();
            return 1;
        }

        device_info = Pa_GetDeviceInfo(device_index);
        if (!device_info)
        {
            std::cerr << "Pa_GetDeviceInfo failed for device " << device_index
                      << '\n';
            Pa_Terminate();
            return 1;
        }

        print_selected_device_summary(device_index, device_info);
        if (device_info->maxOutputChannels < CHANNELS)
        {
            show_failure_context(
                device_index,
                device_info,
                "Selected device does not expose the requested 32 output channels",
                paInvalidChannelCount);
            Pa_Terminate();
            return 1;
        }

        playback_sample_rate = choose_playback_sample_rate(device_info);
    }

    std::vector<StemData> stems;
    stems.reserve(stem_paths.size());
    for (const std::filesystem::path &stem_path : stem_paths)
    {
        StemData stem;
        stem.name = stem_path.stem().string();
        stem.path = stem_path;
        stem.samples = decode_stem_to_stereo_samples(
            stem_path,
            playback_sample_rate);
        if (stem.samples.empty() || (stem.samples.size() % 2) != 0)
        {
            std::cerr << "Failed to decode stem: " << stem_path << '\n';
            Pa_Terminate();
            return 1;
        }
        stem.frame_count = stem.samples.size() / 2;
        stems.push_back(std::move(stem));
    }

    const int outputs_per_stem = CHANNELS / static_cast<int>(stems.size());
    if (outputs_per_stem <= 0)
    {
        std::cerr << "CHANNELS=" << CHANNELS
                  << " is too small for " << stems.size()
                  << " stems.\n";
        Pa_Terminate();
        return 1;
    }

    for (size_t stem_index = 0; stem_index < stems.size(); ++stem_index)
    {
        stems[stem_index].first_output_channel =
            static_cast<int>(stem_index) * outputs_per_stem;
        stems[stem_index].last_output_channel =
            stems[stem_index].first_output_channel + outputs_per_stem - 1;
    }

    const int silent_outputs =
        CHANNELS - outputs_per_stem * static_cast<int>(stems.size());
    print_routing_summary(
        stems,
        outputs_per_stem,
        silent_outputs,
        playback_sample_rate,
        stream_frames_per_buffer);

    size_t total_frames = 0;
    for (const StemData &stem : stems)
    {
        total_frames = std::max(total_frames, stem.frame_count);
    }
    if (total_frames == 0)
    {
        std::cerr << "Decoded stems are empty.\n";
        Pa_Terminate();
        return 1;
    }

    if (dry_run)
    {
        std::cout << "Dry run only. Longest decoded stem has "
                  << total_frames << " frames at " << playback_sample_rate
                  << " Hz.\n";
        if (portaudio_initialized)
        {
            Pa_Terminate();
        }
        return 0;
    }

    PaStreamParameters output_parameters{};
    output_parameters.device = device_index;
    output_parameters.channelCount = CHANNELS;
    output_parameters.sampleFormat = SAMPLE_FORMAT;
    output_parameters.suggestedLatency = std::max(
        device_info->defaultHighOutputLatency,
        2.0 * static_cast<double>(stream_frames_per_buffer) /
            static_cast<double>(playback_sample_rate));
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    const PaError format_error =
        Pa_IsFormatSupported(nullptr, &output_parameters, playback_sample_rate);
    if (format_error != paFormatIsSupported)
    {
        show_failure_context(
            device_index,
            device_info,
            "Requested 32-output playback configuration is not supported",
            format_error);
        Pa_Terminate();
        return 1;
    }

    PaStream *stream = nullptr;
    const PaError open_error = Pa_OpenStream(
        &stream,
        nullptr,
        &output_parameters,
        playback_sample_rate,
        stream_frames_per_buffer,
        paClipOff,
        nullptr,
        nullptr);
    if (open_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_OpenStream failed for the requested playback configuration",
            open_error);
        Pa_Terminate();
        return 1;
    }

    const PaError start_error = Pa_StartStream(stream);
    if (start_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_StartStream failed for the requested playback configuration",
            start_error);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    std::signal(SIGINT, handle_interrupt_signal);
    std::vector<float> output_buffer(
        static_cast<size_t>(stream_frames_per_buffer * CHANNELS),
        0.0f);

    std::cout << "Playing Chega De Saudade stems once across 32 outputs. Press Ctrl+C to stop.\n";

    const PaError prime_error = Pa_WriteStream(
        stream,
        output_buffer.data(),
        stream_frames_per_buffer);
    if (prime_error != paNoError && prime_error != paOutputUnderflowed)
    {
        show_failure_context(
            device_index,
            device_info,
            "Initial stream prime failed",
            prime_error);
        Pa_AbortStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    for (size_t frame_offset = 0;
         frame_offset < total_frames && g_keep_running.load();
         frame_offset += static_cast<size_t>(stream_frames_per_buffer))
    {
        const size_t frames_this_chunk = std::min(
            static_cast<size_t>(stream_frames_per_buffer),
            total_frames - frame_offset);
        std::fill(output_buffer.begin(), output_buffer.end(), 0.0f);

        for (const StemData &stem : stems)
        {
            for (size_t local_frame = 0; local_frame < frames_this_chunk; ++local_frame)
            {
                const size_t stem_frame = frame_offset + local_frame;
                if (stem_frame >= stem.frame_count)
                {
                    continue;
                }

                const size_t sample_offset = stem_frame * 2;
                const float left_sample = STEM_GAIN * stem.samples[sample_offset];
                const float right_sample = STEM_GAIN * stem.samples[sample_offset + 1];
                const float mono_sample = 0.5f * (left_sample + right_sample);
                const size_t row_offset =
                    local_frame * static_cast<size_t>(CHANNELS);
                for (int output_channel = stem.first_output_channel;
                     output_channel <= stem.last_output_channel;
                     ++output_channel)
                {
                    const int relative_output_channel =
                        output_channel - stem.first_output_channel;
                    float sample = mono_sample;
                    if (outputs_per_stem > 1)
                    {
                        const bool is_last_unpaired_output =
                            (outputs_per_stem % 2) != 0 &&
                            relative_output_channel == outputs_per_stem - 1;
                        if (!is_last_unpaired_output)
                        {
                            sample = (relative_output_channel % 2) == 0
                                         ? left_sample
                                         : right_sample;
                        }
                    }
                    output_buffer[row_offset + static_cast<size_t>(output_channel)] =
                        sample;
                }
            }
        }

        const PaError write_error = Pa_WriteStream(
            stream,
            output_buffer.data(),
            static_cast<unsigned long>(frames_this_chunk));
        if (write_error == paOutputUnderflowed)
        {
            static bool warned_about_underflow = false;
            if (!warned_about_underflow)
            {
                std::cerr
                    << "Warning: PortAudio reported an output underflow. "
                    << "Continuing playback.\n";
                warned_about_underflow = true;
            }
            continue;
        }
        if (write_error != paNoError)
        {
            show_failure_context(
                device_index,
                device_info,
                "Pa_WriteStream failed while playing the stem grid",
                write_error);
            Pa_AbortStream(stream);
            Pa_CloseStream(stream);
            Pa_Terminate();
            return 1;
        }
    }

    const PaError stop_error = Pa_StopStream(stream);
    if (stop_error != paNoError)
    {
        show_failure_context(
            device_index,
            device_info,
            "Pa_StopStream failed after playback",
            stop_error);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    Pa_CloseStream(stream);
    if (portaudio_initialized)
    {
        Pa_Terminate();
    }

    std::cout << "Finished Chega De Saudade stem playback.\n";
    return 0;
}
