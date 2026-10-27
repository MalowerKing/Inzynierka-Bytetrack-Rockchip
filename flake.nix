{
  description = "C++ development environment for ML inference and ByteTrack";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSystem = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forEachSystem (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            # Build tools (compilers, build systems, pkg-config)
            nativeBuildInputs = with pkgs; [
              cmake
              pkg-config
              ninja           # Faster alternative to Make
              clang-tools     # Provides clangd for IDE autocompletion
            ];

            # C++ Libraries
            buildInputs = with pkgs; [
              opencv          # Video/image processing, bounding boxes
              eigen           # Matrix operations (Kalman filters for ByteTrack)
              onnxruntime     # Loading and running pre-trained ML models
            ];

            # Environment variables to help CMake and compilers find Nix packages
            shellHook = ''
              echo "🚀 C++ ML Tracking Environment Loaded"
              echo "Libraries: OpenCV, Eigen3, ONNX Runtime"
              
              # Ensure CMake can find OpenCV
              export OpenCV_DIR=${pkgs.opencv}/lib/cmake/opencv4
              fish
            '';
          };
        }
      );
    };
}
