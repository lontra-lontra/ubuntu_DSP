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

          REAL_LIBCUDA_PATH="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/ && $NF !~ /\/stubs(\/|$)/ { print $NF; exit }')"
          if [ -n "$REAL_LIBCUDA_PATH" ]; then
            REAL_LIBCUDA_DIR="$(dirname "$REAL_LIBCUDA_PATH")"
            CLEANED_LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | awk '$0 !~ /\/stubs(\/|$)/ && NF' | paste -sd: -)"
            if [ -n "$CLEANED_LD_LIBRARY_PATH" ]; then
              export LD_LIBRARY_PATH="$REAL_LIBCUDA_DIR:$CLEANED_LD_LIBRARY_PATH"
            else
              export LD_LIBRARY_PATH="$REAL_LIBCUDA_DIR"
            fi
          fi

          echo "Entered the DSP dev shell."
          echo "CPU configure:  cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=sound_device_test -DPORTABLE_USE_MOCK=OFF"
          echo "CUDA configure: cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_APP=cuda_device_query -DPORTABLE_USE_MOCK=ON"
          if [ -n "$REAL_LIBCUDA_PATH" ]; then
            echo "Using host libcuda from: $REAL_LIBCUDA_PATH"
          else
            echo "Warning: could not find a non-stub host libcuda.so.1 with ldconfig."
          fi
        '';
      };
    };
}
