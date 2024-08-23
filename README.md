# GGlow
Graph lowering (Glow) experimental implementation using MLIR

## Requirements
- Install the latest version of bazelisk: [details](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation)
- Install the python3.10 (preferebly using conda): [help](https://saturncloud.io/blog/how-to-create-a-conda-environment-with-a-specific-python-version/)

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
parameters:
- in_channels (int) – Number of channels in the input image
- out_channels (int) – Number of channels produced by the convolution
- kernel_size (int or tuple) – Size of the convolving kernel
- stride (int or tuple, optional) – Stride of the convolution. Default: 1
- padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
- padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
- dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
- groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
- bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

### BatchNorm2d
parameters:
- num_features (int) – CC from an expected input of size (N,C,H,W)(N,C,H,W)
- eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
- momentum (Optional[float]) – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
- affine (bool) – a boolean value that when set to True, this module has learnable affine parameters. Default: True
- track_running_stats (bool) – a boolean value that when set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: True

### Linear
parameters
- in_features (int) – size of each input sample
- out_features (int) – size of each output sample
- bias (bool) – If set to False, the layer will not learn an additive bias. Default: True

### ReLU
parameters
- inplace (bool) – can optionally do the operation in-place. Default: False

### MaxPool2d
parameters
- kernel_size (Union[int, Tuple[int, int]]) – the size of the window to take a max over
- stride (Union[int, Tuple[int, int]]) – the stride of the window. Default value is kernel_size
- padding (Union[int, Tuple[int, int]]) – Implicit negative infinity padding to be added on both sides
- dilation (Union[int, Tuple[int, int]]) – a parameter that controls the stride of elements in the window
- return_indices (bool) – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
- ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape

### Sequential [Have to check this if this is useful]
It is a container operation (I have to ignore it during traversing ig)

### BasicBlock
Similar to sequential, have to figure out what this does

### AdaptiveAvgPool2d
parameters:
- output_size (Union[int, None, Tuple[Optional[int], Optional[int]]]) – the target output size of the image of the form H x W. Can be a tuple (H, W) or a single H for a square image H x H. H and W can be either a int, or None which means the size will be the same as that of the input.

## Current WIP
represent the whole graph using an IR using MLIR taking
inspiration from Glow IR.

## Current TODO
- Define Operations
- Define Types
- Create APIs to generate MLIR.

## References and Learning topics
- Torch-MLIR: https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md
