#ifndef GGLOW_OPS_INTERFACE_H
#define GGLOW_OPS_INTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gglow{
    #include "lib/Dialect/GGlow/GGlowOpsInterface.h.inc"
    std::unique_ptr<mlir::Pass> createShapeInferencePass();
} // namespace gglow 
} // namespace mlir

#endif