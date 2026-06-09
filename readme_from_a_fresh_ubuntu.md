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

## 6. Give JACK and your user realtime priority

If `jackd` prints something like `Cannot use real-time scheduling (RR/10) (1: Operation not permitted)`, give your normal user permission to request realtime priority and lock memory:

```bash
sudo tee /etc/security/limits.d/95-jack-realtime.conf >/dev/null <<EOF
$(id -un) - rtprio 95
$(id -un) - memlock unlimited
EOF
```

This is a normal-user permission change for JACK. It is not a reason to run `jackd` as root.

Then do a full log out and log back in before testing JACK again.

Opening a new terminal tab is not enough. If you are using SSH, disconnect and reconnect. If you want the blunt reliable version, just reboot once.

Optional check after logging back in:

```bash
ulimit -r
ulimit -l
```

`ulimit -r` should now allow a positive realtime priority instead of `0`.

If `ulimit -r` is still `0`, the limits were not applied to that login session. Check:

```bash
cat /etc/security/limits.d/95-jack-realtime.conf
grep pam_limits /etc/pam.d/common-session /etc/pam.d/common-session-noninteractive
```

You want to see your username in the first file, and `pam_limits.so` referenced in the PAM session files.

If `ulimit -l` is still finite instead of `unlimited`, that is the same kind of problem: the login-session limits did not reload yet.

## 7. Enter the Nix shell

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

If you change `flake.nix` later, exit the old shell and run that command again before expecting new tools like `jackd` to appear.

The `flake.nix` in this repo gives you:

- `cmake`
- `ninja`
- `pkg-config`
- Python with `numpy` and `matplotlib`
- `fftw3f`
- ALSA development files
- JACK libraries plus the `jackd` server tool
- a CUDA toolkit path suitable for `find_package(CUDAToolkit)`

## 8. Validate the installations

Run these after entering the Nix shell:

```bash
cmake --version
ninja --version
pkg-config --modversion fftw3f
command -v jackd
python3 -c "import numpy, matplotlib; print(numpy.__version__)"
echo "$CUDAToolkit_ROOT"
echo "$CUDACXX"
nvcc --version
nvidia-smi
```

If those commands work, your shell, CUDA toolkit path, Python packages, FFTW, and driver stack are in the expected state.

## 9. Pick an app and follow the commands at the top of its source file

If `Portable/build` came from another machine, another repo path, or an old configuration, reset it first:

```bash
rm -rf Portable/build
```

Then open the app you want in `Portable/apps` and use the build, run, mock, real-audio, and JACK commands written at the top of that file.

Examples:

- [Portable/apps/sound_device_query.cpp](/home/ian/ubuntu_DSP/Portable/apps/sound_device_query.cpp:1)
- [Portable/apps/calculate_single_delay.cpp](/home/ian/ubuntu_DSP/Portable/apps/calculate_single_delay.cpp:1)
- [Portable/apps/infer_topology_and_save_it.cpp](/home/ian/ubuntu_DSP/Portable/apps/infer_topology_and_save_it.cpp:1)
- [Portable/apps/test_channels.cpp](/home/ian/ubuntu_DSP/Portable/apps/test_channels.cpp:1)
- [Portable/apps/400hz_on_all_channels.cpp](/home/ian/ubuntu_DSP/Portable/apps/400hz_on_all_channels.cpp:1)
- [Portable/apps/multi_conv_benchmarking.cpp](/home/ian/ubuntu_DSP/Portable/apps/multi_conv_benchmarking.cpp:1)
- [Portable/apps/cuda_device_query.cu](/home/ian/ubuntu_DSP/Portable/apps/cuda_device_query.cu:1)
- [Portable/apps/simple_cuda_portaudio.cu](/home/ian/ubuntu_DSP/Portable/apps/simple_cuda_portaudio.cu:1)
- [Portable/apps/simple_cuda_naive_convolution.cu](/home/ian/ubuntu_DSP/Portable/apps/simple_cuda_naive_convolution.cu:1)
- [Portable/apps/simple_cuda_less_naive_convolution.cu](/home/ian/ubuntu_DSP/Portable/apps/simple_cuda_less_naive_convolution.cu:1)

## 10. Troubleshooting

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

For Nix-built binaries, a safer one-shot test is to expose only `libcuda.so.1` through a tiny shim directory instead of prepending the whole system library directory:

```bash
mkdir -p /tmp/portable-real-libcuda-"$USER"
ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 /tmp/portable-real-libcuda-"$USER"/libcuda.so.1
ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 /tmp/portable-real-libcuda-"$USER"/libcuda.so
LD_LIBRARY_PATH="/tmp/portable-real-libcuda-$USER:$CUDAToolkit_ROOT/lib:$(pkg-config --variable=libdir fftw3f)" ./Portable/build/portable_cuda_device_query
```

Prepending `/lib/x86_64-linux-gnu` directly can pull in the host glibc and produce loader errors on a Nix-built binary, so prefer the shim-directory version above.

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

## 11. Extra: update the repo later

```bash
git fetch origin
git reset --hard origin/main
```

That discards local changes and makes the working tree match `origin/main`.

## 12. Optional: rerun the Nix shell

If you pulled changes to `flake.nix`, changed machines, or your shell environment looks stale, exit the current shell and enter it again:

```bash
exit
nix --extra-experimental-features 'nix-command flakes' develop
```
