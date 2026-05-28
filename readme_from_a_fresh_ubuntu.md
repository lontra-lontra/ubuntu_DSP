# Portable DSP from a fresh Ubuntu install

## Short answer

Yes, one solution can cover both Ubuntu Desktop and Ubuntu Server.

The clean split is:

- let Ubuntu own the NVIDIA kernel driver
- let Nix own the user-space build environment
- let this repo's `CMakeLists.txt` decide whether you build CPU-only or CUDA targets

That is not ugly. The only real differences between Desktop and Server are runtime concerns:

- Desktop is nicer for the `matplotlib` viewers because they open windows
- Server is fine for building and for CUDA compute, but real-audio apps need an accessible ALSA device

## What this repo currently expects

From the current `Portable/CMakeLists.txt` and app sources:

- Linux only
- CMake + Ninja
- FFTW single precision (`fftw3f`)
- ALSA headers/libs for apps that default to `MOCK=FALSE`
- optional CUDA toolkit for CUDA targets
- optional Python + `numpy` + `matplotlib` for viewers

Important default behavior:

- these apps currently default to real audio (`MOCK=FALSE`):
  - `portable_sound_device_query`
  - `portable_multi_conv_benchmarking`
  - `portable_infer_topology_and_save_it`
  - `portable_detect_timming`
- these are mock-friendly by default:
  - `portable_infer_topology_from_infered_topology_and_save_it`
  - `portable_simple_cuda_portaudio`
  - `portable_simple_cuda_naive_convolution`
  - `portable_simple_cuda_less_naive_convolution`
- `portable_cuda_device_query` does not need audio hardware

## Recommended approach

### 1. Pre-install: download Ubuntu and write the USB stick

Pick the Ubuntu image you want:

- Ubuntu Desktop if you want the easiest plotting and audio-device inspection experience
- Ubuntu Server if this box is mainly for CUDA and command-line work

On another machine:

1. Download the Ubuntu `.iso` for the target architecture from Ubuntu.
2. Install Balena Etcher.
3. Insert a USB stick large enough for the image.
4. In Balena Etcher, choose the Ubuntu `.iso`, choose the USB stick, then click Flash.
5. Safely eject the USB stick, boot the target machine from it, and start the installer.

If this machine is headless, Ubuntu Server is viable.

### 2. Install Ubuntu

During the installer:

- use the normal Ubuntu install flow
- let Ubuntu use the whole target disk unless you already have a partitioning plan
- enable network access during install if possible
- reboot into the fresh system when the installer finishes

## 3. Clone this repo

```bash
git clone https://github.com/lontra-lontra/ubuntu_DSP.git
cd ubuntu_DSP
```

## 4. Install the NVIDIA driver with Ubuntu, not with Nix

Do this before touching the CUDA apps.

Update the machine first:

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

After reboot:

- on Ubuntu Desktop:

```bash
sudo ubuntu-drivers install
sudo reboot
```

- on Ubuntu Server / headless compute box:

```bash
sudo ubuntu-drivers install --gpgpu
sudo reboot
```

Verify after reboot:

```bash
nvidia-smi
```

If `nvidia-smi` works, the system driver side is in good shape.

## 5. Install Nix

Recommended on Linux: multi-user install.

```bash
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon
```

Then start a fresh login shell or log out and back in.

## 6. Enter the Nix shell

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

The `flake.nix` in this repo gives you:

- `cmake`
- `ninja`
- `pkg-config`
- Python with `numpy` and `matplotlib`
- `fftw3f`
- ALSA development files
- a CUDA toolkit path suitable for `find_package(CUDAToolkit)`

Note:

- the flake intentionally keeps the NVIDIA driver outside Nix
- the flake uses `cudaPackages.cudatoolkit` because this repo's CMake expects a conventional toolkit layout

## 7. Configure and build

### CPU-only

```bash
cmake -S Portable -B Portable/build -G Ninja
cmake --build Portable/build
```

### CPU + CUDA

```bash
cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_ENABLE_CUDA_APPS=ON
cmake --build Portable/build
```

## 8. Build only the app you care about

Examples:

```bash
cmake --build Portable/build --target portable_sound_device_query
cmake --build Portable/build --target portable_multi_conv_benchmarking
cmake --build Portable/build --target portable_cuda_device_query
cmake --build Portable/build --target portable_simple_cuda_less_naive_convolution
```

## 9. Run apps

Examples:

```bash
./Portable/build/portable_sound_device_query
./Portable/build/portable_multi_conv_benchmarking
./Portable/build/portable_infer_topology_and_save_it
./Portable/build/portable_detect_timming
./Portable/build/portable_cuda_device_query
./Portable/build/portable_simple_cuda_portaudio
./Portable/build/portable_simple_cuda_naive_convolution
./Portable/build/portable_simple_cuda_less_naive_convolution
```

## 10. Headless-server caveats

This is the part that matters most for "one solution for both".

### What works well on both Desktop and Server

- installing the Ubuntu NVIDIA driver
- installing Nix
- `nix --extra-experimental-features 'nix-command flakes' develop`
- building CPU targets
- building CUDA targets
- running `portable_cuda_device_query`
- running mock-mode apps

### What is less clean on a pure headless server

- Python viewer scripts call `matplotlib.pyplot.show()`
- real-audio apps need a usable ALSA device
- device names like `"default"` only help if the server actually exposes a sound device to the process

So the single-solution story is:

- yes for build environment
- yes for CUDA
- yes for mock audio
- maybe for real audio, depending on whether the server has a real ALSA device
- awkward for interactive plotting unless you use X forwarding or open the generated CSVs elsewhere

The generated files land under `Portable/output`.

## 11. If you want to switch an app between mock and real audio

Each app keeps its main knobs near the top of the source file in `Portable/apps`.

Typical knobs are:

- `#define MOCK TRUE` or `FALSE`
- `#define DEVICE_NAME ...`
- sample rate / buffer size / channel count constants

Examples:

- [Portable/apps/sound_device_query.cpp](/home/ian/DSP/Portable/apps/sound_device_query.cpp:1)
- [Portable/apps/multi_conv_benchmarking.cpp](/home/ian/DSP/Portable/apps/multi_conv_benchmarking.cpp:1)
- [Portable/apps/simple_cuda_naive_convolution.cu](/home/ian/DSP/Portable/apps/simple_cuda_naive_convolution.cu:1)

If you have no real audio hardware on the target machine, prefer apps that already use `MOCK=TRUE`, or flip the app you want to test into mock mode and rebuild it.

## 12. Troubleshooting

### `nvidia-smi` fails

The Ubuntu driver install is not finished or not matched to the running kernel yet. Fix that before debugging Nix or CMake.

### `cmake` cannot find CUDA

Run the shell through the flake first:

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

Then verify:

```bash
echo "$CUDAToolkit_ROOT"
echo "$CUDACXX"
nvcc --version
```

### Runtime error about `libcuda.so.1`

That usually means the host driver library is not visible at runtime even though the toolkit is.

Find the host driver library:

```bash
ldconfig -p | grep libcuda.so.1
```

If needed, export the directory containing that library into `LD_LIBRARY_PATH` before running the CUDA app.

### Real-audio app opens but cannot use the device

Run:

```bash
./Portable/build/portable_sound_device_query
```

and confirm that the ALSA device you expect is visible through PortAudio.
