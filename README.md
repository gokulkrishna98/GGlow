# GGlow
Graph lowering (Glow) experimental implementation using MLIR

## Requirements
- Install the latest version of bazelisk: [details](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation)
- Install the python3.10 (preferebly using conda): [help](https://saturncloud.io/blog/how-to-create-a-conda-environment-with-a-specific-python-version/)
- Have Clang compiler tools installed

## Dependencies Added
Need to resolve them to be done automatically from build.
- "/usr/local/lib/libmlir_c_runner_utils.so"
- "/usr/local/lib/libmlir_runner_utils.so"

## Building Steps
- Clone the repo and go inside
```
git clone https://github.com/gokulkrishna98/GGlow.git
cd GGlow
```
- Clone the submodules
```
git submodule init
git submodule update
```
- Run the bazel build command
    - For building GlowDialect
    ```
    bazel build lib/Dialect/GGlow:GGlowDialect
    ```
    - For building gglow
    ```
    bazel build src:gglow
    ```
    - For building all of mlir binaries from llvm-project
    ```
    bazel build @llvm-project//mlir/...:all
    ```
- You can find the exectuables from bazil-bin path


## Current Goal
- Read the pytorch model: get graph and weights.
- Develop GGlow dialect to represent this model.
- Do simple lowering executing on my Intel cpu.

## Format Stype
Using WebKit stype of format using clang-format tool
```
clang-format --style=WebKit -i gglow.cpp
```

## Current TODO
-[x] Lower Conv2d to LLVM dialect
-[x] Generate resblock high level operation
-[ ] Convert pytorch model to IR in GGLow Dialect
-[ ] Lower to LLVM 
## References and Learning topics
- Torch-MLIR: https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md
