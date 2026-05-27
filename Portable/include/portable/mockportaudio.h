#pragma once

#include <functional>
#include <string>
#include <vector>

using PaError = int;
using PaDeviceIndex = int;
using PaTime = double;
using PaSampleFormat = unsigned long;
using PaStreamFlags = unsigned long;
using PaStreamCallbackFlags = unsigned long;

struct PaStream;
struct MockDevice;

using PaStreamCallback = int(
    const void *input,
    void *output,
    unsigned long frameCount,
    const struct PaStreamCallbackTimeInfo *timeInfo,
    PaStreamCallbackFlags statusFlags,
    void *userData);

static constexpr PaError paNoError = 0;
static constexpr PaError paNotInitialized = -10000;
static constexpr PaError paInvalidDevice = -9996;
static constexpr PaError paInvalidChannelCount = -9998;
static constexpr PaError paInvalidSampleRate = -9997;
static constexpr PaError paInvalidSampleFormat = -9994;
static constexpr PaError paBadStreamPtr = -9988;
static constexpr PaError paInvalidFlag = -9992;
static constexpr PaError paFormatIsSupported = 0;

static constexpr PaSampleFormat paFloat32 = 0x00000001ul;
static constexpr PaSampleFormat paInt16 = 0x00000008ul;

static constexpr PaDeviceIndex paNoDevice = -1;

static constexpr PaStreamCallbackFlags paInputUnderflow = 0x00000001ul;
static constexpr PaStreamCallbackFlags paInputOverflow = 0x00000002ul;
static constexpr PaStreamCallbackFlags paOutputUnderflow = 0x00000004ul;
static constexpr PaStreamCallbackFlags paOutputOverflow = 0x00000008ul;
static constexpr PaStreamCallbackFlags paPrimingOutput = 0x00000010ul;

static constexpr int paContinue = 0;
static constexpr int paComplete = 1;
static constexpr int paAbort = 2;

static constexpr PaStreamFlags paClipOff = 0x00000001ul;

struct PaStreamCallbackTimeInfo
{
    double inputBufferAdcTime = 0.0;
    double currentTime = 0.0;
    double outputBufferDacTime = 0.0;
};

struct PaStreamParameters
{
    PaDeviceIndex device = paNoDevice;
    int channelCount = 0;
    PaSampleFormat sampleFormat = paFloat32;
    PaTime suggestedLatency = 0.0;
    void *hostApiSpecificStreamInfo = nullptr;
};

struct PaDeviceInfo
{
    int structVersion = 2;
    const char *name = nullptr;
    int hostApi = 0;
    int maxInputChannels = 0;
    int maxOutputChannels = 0;
    double defaultLowInputLatency = 0.01;
    double defaultLowOutputLatency = 0.01;
    double defaultHighInputLatency = 0.1;
    double defaultHighOutputLatency = 0.1;
    double defaultSampleRate = 48000.0;
};

using MockInputGenerator = std::function<float(double)>;
using MockStreamClosedCallback = std::function<void(MockDevice &, double)>;

struct MockDevice
{
    PaDeviceInfo info{};
    PaSampleFormat supportedFormat = paFloat32;
    std::vector<double> supportedSampleRates;
    std::vector<MockInputGenerator> inputGenerators;
    std::vector<std::vector<float>> outputHistory;
    void *internalState = nullptr;
    MockStreamClosedCallback onStreamClosed;
    bool hasPendingStreamClosed = false;
    double pendingStreamClosedSampleRate = 0.0;
    std::string ownedName;
};

struct PaStream
{
    bool hasInputParams = false;
    bool hasOutputParams = false;
    PaStreamParameters inputParams{};
    PaStreamParameters outputParams{};
    double sampleRate = 0.0;
    unsigned long framesPerBuffer = 0;
    PaStreamCallback *callback = nullptr;
    void *userData = nullptr;
    bool active = false;
};

PaError Pa_Initialize();
PaError Pa_Terminate();
PaDeviceIndex Pa_GetDeviceCount();
const PaDeviceInfo *Pa_GetDeviceInfo(PaDeviceIndex device);
const char *Pa_GetErrorText(PaError error);
PaDeviceIndex Pa_GetDefaultInputDevice();
PaDeviceIndex Pa_GetDefaultOutputDevice();
PaError Pa_IsFormatSupported(
    const PaStreamParameters *inputParameters,
    const PaStreamParameters *outputParameters,
    double sampleRate);
PaError Pa_OpenStream(
    PaStream **stream,
    const PaStreamParameters *inputParameters,
    const PaStreamParameters *outputParameters,
    double sampleRate,
    unsigned long framesPerBuffer,
    PaStreamFlags streamFlags,
    PaStreamCallback *streamCallback,
    void *userData);
PaError Pa_StartStream(PaStream *stream);
PaError Pa_StopStream(PaStream *stream);
PaError Pa_CloseStream(PaStream *stream);
int Pa_IsStreamActive(PaStream *stream);
void Pa_Sleep(long milliseconds);

std::vector<MockDevice> *mock_pa_get_devices_list();
