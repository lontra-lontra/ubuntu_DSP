# Portable on Linux

## Dependencies

Required build tools:

- `cmake`
- `ninja`
- `pkg-config`
- a C compiler and a C++17 compiler such as `gcc`/`g++` or `clang`/`clang++`

Required runtime/build dependencies for real audio (`MOCK=FALSE`):

- FFTW single-precision development files such as `libfftw3-dev`
- ALSA development headers and libraries such as `libasound2-dev`

Required runtime/build dependencies for CUDA apps:

- an NVIDIA GPU with a working Linux driver
- the NVIDIA CUDA Toolkit, including `nvcc`

Optional local tools for plots and viewers:

- `python3`
- `python3 -m pip install matplotlib`
- `python3 -m pip install numpy`

Vendored in this repo:

- PortAudio source (`Portable/third_party/portaudio`)
- CUDA sample helper headers used by `portable_cuda_device_query`

## Build

CPU targets:

```bash
cmake -S Portable -B Portable/build -G Ninja
cmake --build Portable/build
```

CUDA targets too:

```bash
cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_ENABLE_CUDA_APPS=ON
cmake --build Portable/build
```

## MOCK vs non-MOCK

Each app in `Portable/apps` defines its own `MOCK` value near the top of the file.

- `MOCK=TRUE`: uses the local mock audio path and does not need ALSA hardware access
- `MOCK=FALSE`: uses vendored PortAudio with ALSA on Linux

If an app is in non-MOCK mode, `DEVICE_NAME "default"` uses the default ALSA device selected by PortAudio. Run `portable_sound_device_query` first if you want to inspect the visible device list before changing a device name.

## Typical commands

```bash
./Portable/build/portable_sound_device_query
./Portable/build/portable_multi_conv_benchmarking
./Portable/build/portable_infer_topology_and_save_it
./Portable/build/portable_detect_timming
```

With CUDA enabled:

```bash
./Portable/build/portable_cuda_device_query
./Portable/build/portable_simple_cuda_portaudio
./Portable/build/portable_simple_cuda_naive_convolution
./Portable/build/portable_simple_cuda_less_naive_convolution
```
