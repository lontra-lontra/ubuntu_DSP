# Portable DSP from a fresh Ubuntu install

## What this repo expects

- Linux
- CMake + Ninja
- FFTW single precision (`fftw3f`)
- ALSA headers and libraries for apps that default to `MOCK=FALSE`
- optional CUDA toolkit for CUDA targets
- optional Python with `numpy` and `matplotlib` for viewer scripts

## 1. Pre-install: download Ubuntu and write the USB stick

For a normal laptop or workstation:

- Ubuntu Desktop ISO: https://ubuntu.com/download/desktop
- Ubuntu Desktop install tutorial: https://documentation.ubuntu.com/desktop/en/24.04/tutorial/install-ubuntu-desktop/

For a headless machine or mostly-CLI CUDA box:

- Ubuntu Server ISO: https://ubuntu.com/download/server
- Ubuntu Server install tutorial: https://ubuntu.com/tutorials/tutorial-install-ubuntu-server

To write the USB installer:

- Balena Etcher download: https://etcher.balena.io/
- Balena Etcher docs: https://etcher-docs.balena.io/

Concrete flow:

1. On another machine, download the Ubuntu Desktop or Ubuntu Server ISO from the links above.
2. Download and open Balena Etcher.
3. Insert an 8 GB or larger USB stick.
4. In Etcher, click `Flash from file`, pick the Ubuntu ISO, pick the USB stick, then click `Flash`.
5. Safely eject the USB stick, boot the target machine from it, and follow the matching official Ubuntu install tutorial.

## 2. Install Ubuntu

Follow the official Ubuntu installer tutorial you chose in step 1.

## 3. Install base packages and clone this repo

```bash
cd ~
sudo apt update
sudo apt install -y git curl
git clone https://github.com/lontra-lontra/ubuntu_DSP.git
cd ubuntu_DSP
```

If you copied this repo from another machine instead of cloning it, remove any old CMake build directory now:

```bash
rm -rf Portable/build
```

## 4. Install the NVIDIA driver with Ubuntu, not with Nix

Do this before touching the CUDA apps.

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

After reboot:

On Ubuntu Desktop:

```bash
sudo ubuntu-drivers install
sudo reboot
```

On Ubuntu Server or a headless compute box:

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

## 7. Validate the installations

Run these after entering the Nix shell:

```bash
cmake --version
ninja --version
pkg-config --modversion fftw3f
python3 -c "import numpy, matplotlib; print(numpy.__version__)"
echo "$CUDAToolkit_ROOT"
echo "$CUDACXX"
nvcc --version
nvidia-smi
```

If those commands work, your shell, CUDA toolkit path, Python packages, FFTW, and driver stack are in the expected state.

## 8. Build one app at a time

Important: CMake build directories are host-specific. There is no elegant way to reuse the same `Portable/build` directory across different machines or different repo paths. The normal fix is to delete that build directory and configure again on the current machine.

If you see an error like `CMakeCache.txt directory ... is different than the directory ... where CMake was created`, run:

```bash
rm -rf Portable/build
```

Then configure again on the current machine.

The simplest rule is: open the source file for the app you want under `Portable/apps` and copy the build and run commands written at the top of that file.

Important:

- `cmake -S Portable -B Portable/build ...` is the configure step
- `cmake --build Portable/build ...` is only the build step
- if `Portable/build` does not exist yet, if it came from another machine, or if you changed `PORTABLE_APP` / `PORTABLE_USE_MOCK`, rerun the first `cmake -S ...` command before the `cmake --build ...` command
- the first command creates or refreshes `Portable/build/CMakeCache.txt`

Exact recovery commands:

```bash
rm -rf Portable/build
cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=sound_device_test -DPORTABLE_USE_MOCK=OFF
cmake --build Portable/build --target portable_sound_device_test --parallel
```

Examples:

- [Portable/apps/sound_device_query.cpp](/home/ian/ubuntu_DSP/Portable/apps/sound_device_query.cpp:1)
- [Portable/apps/sound_device_test.cpp](/home/ian/ubuntu_DSP/Portable/apps/sound_device_test.cpp:1)
- [Portable/apps/multi_conv_benchmarking.cpp](/home/ian/ubuntu_DSP/Portable/apps/multi_conv_benchmarking.cpp:1)
- [Portable/apps/infer_topology_and_save_it.cpp](/home/ian/ubuntu_DSP/Portable/apps/infer_topology_and_save_it.cpp:1)
- [Portable/apps/detect_timming.cpp](/home/ian/ubuntu_DSP/Portable/apps/detect_timming.cpp:1)
- [Portable/apps/cuda_device_query.cu](/home/ian/ubuntu_DSP/Portable/apps/cuda_device_query.cu:1)
- [Portable/apps/simple_cuda_portaudio.cu](/home/ian/ubuntu_DSP/Portable/apps/simple_cuda_portaudio.cu:1)
- [Portable/apps/simple_cuda_naive_convolution.cu](/home/ian/ubuntu_DSP/Portable/apps/simple_cuda_naive_convolution.cu:1)
- [Portable/apps/simple_cuda_less_naive_convolution.cu](/home/ian/ubuntu_DSP/Portable/apps/simple_cuda_less_naive_convolution.cu:1)

That is still CMake, but only because this repo already uses CMake as its build system. The readme does not need to teach the whole build system if the per-app commands at the top of each file are enough.

## 9. Run apps

Use the run command written at the top of the app source file you chose in step 8.

## 10. Switching between mock and real audio

Each app keeps its main knobs near the top of the source file in `Portable/apps`.

Typical knobs are:

- mock/real audio selection in the first `cmake` command at the top of each app file:
  `-DPORTABLE_USE_MOCK=ON` or `-DPORTABLE_USE_MOCK=OFF`
- `#define DEVICE_NAME ...`
- sample rate, buffer size, and channel count constants

Examples:

- [Portable/apps/sound_device_query.cpp](/home/ian/ubuntu_DSP/Portable/apps/sound_device_query.cpp:1)
- [Portable/apps/sound_device_test.cpp](/home/ian/ubuntu_DSP/Portable/apps/sound_device_test.cpp:1)
- [Portable/apps/multi_conv_benchmarking.cpp](/home/ian/ubuntu_DSP/Portable/apps/multi_conv_benchmarking.cpp:1)

If you have no real audio hardware on the target machine, use the app command that already has `-DPORTABLE_USE_MOCK=ON`, or change only that flag and rerun the first `cmake` command before rebuilding.

## 11. Troubleshooting

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

Use the non-`/stubs/` path from that output. If needed, export the directory containing that real driver library into `LD_LIBRARY_PATH` before running the CUDA app.

### `CUDA driver is a stub library`

That means the program found the CUDA toolkit's placeholder `libcuda.so.1` instead of the real NVIDIA driver library from the host system.

Check both locations:

```bash
ldconfig -p | grep libcuda.so.1
find "$CUDAToolkit_ROOT" -path '*/lib/stubs/libcuda.so*' 2>/dev/null
```

Then force the real host driver directory to the front of `LD_LIBRARY_PATH`:

```bash
REAL_LIBCUDA_PATH="$(ldconfig -p | awk '/libcuda\.so\.1/ && $NF !~ /\/stubs(\/|$)/ { print $NF; exit }')"
REAL_LIBCUDA_DIR="$(dirname "$REAL_LIBCUDA_PATH")"
export LD_LIBRARY_PATH="$REAL_LIBCUDA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

If you want a blunt one-shot run command for a CUDA app, this is fine too:

```bash
LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$CUDAToolkit_ROOT/lib:$(pkg-config --variable=libdir fftw3f)" ./Portable/build/portable_cuda_device_query
```

If `LD_LIBRARY_PATH` already contains a CUDA `stubs` directory, start a fresh shell or remove that entry before running the app again.

If `ldconfig -p | grep libcuda.so.1` finds only `/stubs/` paths or finds nothing, the NVIDIA host driver is still not installed correctly even if the CUDA toolkit is present.

### Real-audio app opens but cannot use the device

Run:

```bash
cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=sound_device_query -DPORTABLE_USE_MOCK=OFF
cmake --build Portable/build --target portable_sound_device_query --parallel
./Portable/build/portable_sound_device_query
```

and confirm that the ALSA device you expect is visible through PortAudio.

## 12. Extra: update the repo later

```bash
git pull
```

## 13. Optional: rerun the Nix shell

If you pulled changes to `flake.nix`, changed machines, or your shell environment looks stale, exit the current shell and enter it again:

```bash
exit
nix --extra-experimental-features 'nix-command flakes' develop
```
