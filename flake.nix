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

          echo "Entered the DSP dev shell."
          echo "CPU configure:  cmake -S Portable -B Portable/build -G Ninja"
          echo "CUDA configure: cmake -S Portable -B Portable/build -G Ninja -DPORTABLE_ENABLE_CUDA_APPS=ON"
        '';
      };
    };
}
