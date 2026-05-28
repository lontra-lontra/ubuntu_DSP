# Portable DSP from a fresh Ubuntu install


## What this repo currently expects

and app sources:

- Linux only
- CMake + Ninja
- FFTW single precision (`fftw3f`)
- ALSA headers/libs for apps that default to `MOCK=FALSE`
- optional CUDA toolkit for CUDA targets
- optional Python + `numpy` + `matplotlib` for viewers


### 1. Pre-install: download Ubuntu and write the USB stick

For a normal laptop or workstation, use Ubuntu Desktop:

- ISO download: https://ubuntu.com/download/desktop
- official install tutorial: https://documentation.ubuntu.com/desktop/en/24.04/tutorial/install-ubuntu-desktop/

For a headless machine or mostly-CLI CUDA box, use Ubuntu Server:

- ISO download: https://ubuntu.com/download/server
- official install tutorial: https://ubuntu.com/tutorials/tutorial-install-ubuntu-server

To write the installer USB:

- Balena Etcher download: https://etcher.balena.io/
- Balena Etcher docs: https://etcher-docs.balena.io/

Concrete flow:

1. On another machine, download the Ubuntu Desktop or Ubuntu Server ISO from the link above.
2. Download and open Balena Etcher.
3. Insert an 8 GB or larger USB stick.
4. In Etcher, click `Flash from file`, pick the Ubuntu ISO, pick the USB stick, then click `Flash`.
5. Safely eject the USB stick, boot the target machine from it, and follow the matching official Ubuntu install tutorial above.

### 2. Install Ubuntu

Follow the official Ubuntu installer tutorial you chose in step 1.

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


## RUN ON OF THESE : (THE COMMANDS ARE IN THE HEADER)
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
