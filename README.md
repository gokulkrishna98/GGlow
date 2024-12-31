# GGlow
Graph lowering (Glow) experimental implementation using MLIR

## Requirements
- Install the latest version of bazelisk: [details](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation)
- Install the python3.10 (preferebly using conda): [help](https://saturncloud.io/blog/how-to-create-a-conda-environment-with-a-specific-python-version/)
- Have Clang compiler tools installed

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

## Layers Used in ResNet18
### Conv2d
### BatchNorm2d
### Linear
### ReLU
### MaxPool2d
### Sequential [Have to check this if this is useful]
It is a container operation (I have to ignore it during traversing ig)
### BasicBlock
Similar to sequential, have to figure out what this does

### AdaptiveAvgPool2d
## Current WIP
represent the whole graph using an IR using MLIR taking
inspiration from Glow IR.

## Current TODO
- Define Operations
- Define Types
- Create APIs to generate MLIR.

## References and Learning topics
- Torch-MLIR: https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md
