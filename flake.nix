{
  description = "DSP development shell for the Portable Linux/CUDA apps";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      cudaToolkit = pkgs.cudaPackages.cudatoolkit;
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        matplotlib
        numpy
      ]);
    in {
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          pythonEnv
          cudaToolkit
        ];

        buildInputs = [
          pkgs.alsa-lib
          pkgs.fftwFloat
        ];

        shellHook = ''
          export CMAKE_GENERATOR=Ninja
          export CUDA_PATH="${cudaToolkit}"
          export CUDA_HOME="${cudaToolkit}"
          export CUDAToolkit_ROOT="${cudaToolkit}"
          export CUDACXX="${cudaToolkit}/bin/nvcc"

          BASE_LD_LIBRARY_PATH="${cudaToolkit}/lib:${pkgs.fftwFloat}/lib"
          HOST_LDCONFIG=""
          for candidate in "$(command -v ldconfig 2>/dev/null)" /usr/sbin/ldconfig /sbin/ldconfig; do
            if [ -n "$candidate" ] && [ -x "$candidate" ]; then
              HOST_LDCONFIG="$candidate"
              break
            fi
          done

          REAL_LIBCUDA_PATH=""
          if [ -n "$HOST_LDCONFIG" ]; then
            REAL_LIBCUDA_PATH="$("$HOST_LDCONFIG" -p 2>/dev/null | awk '/libcuda\.so\.1/ && $NF !~ /\/stubs(\/|$)/ { print $NF; exit }')"
          fi
          if [ -z "$REAL_LIBCUDA_PATH" ]; then
            for candidate in /lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
              if [ -e "$candidate" ]; then
                REAL_LIBCUDA_PATH="$candidate"
                break
              fi
            done
          fi

          CLEANED_LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | awk '$0 !~ /\/stubs(\/|$)/ && NF' | paste -sd: -)"
          if [ -n "$REAL_LIBCUDA_PATH" ]; then
            PORTABLE_REAL_LIBCUDA_DIR="/tmp/portable-real-libcuda-$USER"
            mkdir -p "$PORTABLE_REAL_LIBCUDA_DIR"
            ln -sfn "$REAL_LIBCUDA_PATH" "$PORTABLE_REAL_LIBCUDA_DIR/libcuda.so.1"
            ln -sfn "$REAL_LIBCUDA_PATH" "$PORTABLE_REAL_LIBCUDA_DIR/libcuda.so"
            if [ -n "$CLEANED_LD_LIBRARY_PATH" ]; then
              export LD_LIBRARY_PATH="$PORTABLE_REAL_LIBCUDA_DIR:$BASE_LD_LIBRARY_PATH:$CLEANED_LD_LIBRARY_PATH"
            else
              export LD_LIBRARY_PATH="$PORTABLE_REAL_LIBCUDA_DIR:$BASE_LD_LIBRARY_PATH"
            fi
          elif [ -n "$CLEANED_LD_LIBRARY_PATH" ]; then
            export LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH:$CLEANED_LD_LIBRARY_PATH"
          else
            export LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH"
          fi

          echo "Entered the DSP dev shell."
          echo "CPU configure:  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=sound_device_test -DPORTABLE_USE_MOCK=OFF"
          echo "CUDA configure: cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=cuda_device_query -DPORTABLE_USE_MOCK=ON"
          if [ -n "$REAL_LIBCUDA_PATH" ]; then
            echo "Using host libcuda from: $REAL_LIBCUDA_PATH"
            echo "Shim libcuda directory: $PORTABLE_REAL_LIBCUDA_DIR"
          else
            echo "Warning: could not find a non-stub host libcuda.so.1 with ldconfig."
          fi
          echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
        '';
      };
    };
}
