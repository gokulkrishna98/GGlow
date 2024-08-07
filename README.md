# GGlow
Graph lowering (Glow) experimental implementation using MLIR

Current Goal
- Read the pytorch model: get graph and weights.
- Develop GGlow dialect to represent this model.
- Do simple lowering executing on my Intel cpu.

## Format Stype
Using WebKit stype of format using clang-format tool
```
clang-format --style=WebKit -i gglow.cpp
```

## Layers Used in ResNet18
- Conv2d
- BatchNorm2d
- Linear
- ReLU
- MaxPool2d
- Sequential [Have to check this if this is useful]
- BasicBlock
- AdaptiveAvgPool2d

## Current WIP
represent the whole graph using an IR using MLIR taking
inspiration from Glow IR.
